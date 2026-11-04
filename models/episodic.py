"""Shared probability, sampling and conditional auxiliary-loss contracts.

Support decoders emit logits. getSelfPred in each model applies exactly one
sigmoid and returns foreground probabilities, never logits or two-class maps.
"""
import random
import torch
import torch.nn.functional as F


def binary_probabilities(foreground):
    """[N,1,H,W] foreground probabilities -> [N,2,H,W], order bg, fg."""
    if foreground.ndim != 4 or foreground.shape[1] != 1:
        raise ValueError('Expected foreground probability shape [N,1,H,W]')
    return torch.cat((1.0 - foreground, foreground), dim=1)


def sample_masked_features(features, mask, count):
    """Uniform sample without replacement; repeat complete pools if too small.

    Discrete indices/masks are not differentiated. Indexing the original feature
    tensor preserves gradients to every selected feature (including repeats).
    """
    if count < 0:
        raise ValueError('Prototype count must be nonnegative')
    if features.shape[0] != 1 or mask.shape[0] != 1:
        raise ValueError('Sampling is per episode, with batch dimension one')
    channels = features.shape[1]
    pool = features[0].permute(1, 2, 0).reshape(-1, channels)[mask.reshape(-1) == 1]
    size = len(pool)
    if count == 0 or size == 0:
        return features.new_zeros((count, channels))
    if size >= count:
        return pool[random.sample(range(size), count)]
    repeats, remainder = divmod(count, size)
    indices = list(range(size)) * repeats + random.sample(range(size), remainder)
    return pool[indices]


class EpisodicSupportMixin:
    @property
    def device(self):
        return next(self.parameters()).device

    def compute_support_loss(self, probabilities, masks):
        """Mean NLL over valid support way/shot pairs, once per episode.

        A pair is valid when at least one mask pixel is 0 or 1. Other pixels
        (including 255) are ignored. An all-invalid pair is skipped before NLL.
        The public forward supports one way/shot; nested probabilities also make
        the normalization explicit and testable for more than one support pair.
        """
        ways, shots = masks.shape[:2]
        if torch.is_tensor(probabilities):
            if (ways, shots) != (1, 1):
                raise ValueError('Provide one probability tensor per way/shot')
            probabilities = [[probabilities]]
        terms = []
        for way in range(ways):
            for shot in range(shots):
                target = masks[way, shot]
                valid = (target == 0) | (target == 1)
                if not bool(valid.any().item()):
                    continue
                prob = probabilities[way][shot]
                eps = torch.finfo(prob.dtype).eps
                label = target.masked_fill(~valid, 255).long().unsqueeze(0)
                terms.append(F.nll_loss(prob.clamp(eps, 1 - eps).log(), label, ignore_index=255))
        return torch.stack(terms).mean() if terms else masks.new_zeros((), dtype=torch.float32)

    def _query_self_probability(self, features, foreground, background):
        if self.context_self_prediction:
            return self.getSelfPred(features, foreground, background)
        return self.getSelfPred(features, foreground)

    def compute_auxiliary_losses(self, supp_fts, qry_fts, fore_mask, pred,
                                 sup_fg_pts, sup_bg_pts, self_pred):
        """SSP is independent; align/aux require a nonempty query seed.

        Final query hard mask -> optional support-based fallback -> pooled query
        fg/bg globals -> query self-prediction -> query error masks/prototypes ->
        reverse support logits. Invalid query branches are never evaluated.
        """
        ssp = self.compute_support_loss(self_pred, fore_mask)
        zero = pred.new_zeros(())
        align_terms, aux_terms = [], []
        ways, shots = fore_mask.shape[:2]
        hard_query = pred.argmax(dim=1)
        for way in range(ways):
            valid_shots = [shot for shot in range(shots)
                if bool(((fore_mask[way, shot] == 0) | (fore_mask[way, shot] == 1)).any().item())]
            if not valid_shots:
                continue
            query_mask = (hard_query == way + 1).float()
            if not bool((query_mask > 0).any().item()):
                if self.context_self_prediction:
                    seed_fg, seed_bg = sup_fg_pts[way], sup_bg_pts[way]
                else:
                    seed_fg = torch.stack([self.getFeatures(supp_fts[way, [shot]],
                        fore_mask[way, [shot]]) for shot in valid_shots]).mean(dim=0)
                    seed_bg = None
                seed = self._query_self_probability(qry_fts, seed_fg, seed_bg)
                seed = F.interpolate(seed, size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)
                query_mask = binary_probabilities(seed).argmax(dim=1).float()
                if not bool((query_mask > 0).any().item()):
                    continue  # SSP was already computed; do not evaluate align/aux.

            query_fg = self.getFeatures(qry_fts, query_mask)
            query_bg = self.getFeatures(qry_fts, 1 - query_mask)
            query_self = self._query_self_probability(qry_fts, query_fg, query_bg)
            query_self = F.interpolate(query_self, size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)
            query_self_mask = binary_probabilities(query_self).argmax(dim=1)
            query_fg_pts = self.get_fg_pts(qry_fts, query_mask, query_self_mask)
            query_bg_pts = self.get_bg_pts(qry_fts, query_mask, query_self_mask)
            aux_terms.append(self.get_aux_loss(sup_fg_pts[way], query_fg_pts,
                                               sup_bg_pts[way], query_bg_pts))
            shot_terms = []
            for shot in valid_shots:
                fg = self.get_fg_sim(supp_fts[way, [shot]], query_fg_pts)
                bg = self.get_bg_sim(supp_fts[way, [shot]], query_bg_pts)
                logits = F.interpolate(torch.cat((bg, fg), dim=1), size=fore_mask.shape[-2:],
                                       mode='bilinear', align_corners=True)
                target = fore_mask[way, shot]
                valid = (target == 0) | (target == 1)
                label = target.masked_fill(~valid, 255).long().unsqueeze(0)
                prob = logits.softmax(dim=1)
                eps = torch.finfo(prob.dtype).eps
                shot_terms.append(F.nll_loss(prob.clamp(eps, 1-eps).log(), label, ignore_index=255))
            align_terms.append(torch.stack(shot_terms).mean())
        # Divide by all ways (including skipped query ways); outer forward divides by batch.
        align = torch.stack(align_terms).sum() / ways if align_terms else zero
        aux = torch.stack(aux_terms).sum() / ways if aux_terms else zero
        return align, aux, zero, ssp
