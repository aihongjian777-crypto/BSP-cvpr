"""
FSS via CoW
Extended from ADNet code by Hansen et al.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder import Res101Encoder
from models.module_ct import MLP, Decoder, Supp_Decoder
import numpy as np
import random
import cv2
from boundary_loss import BoundaryLoss
from models.episodic import EpisodicSupportMixin, binary_probabilities, sample_masked_features


class FewShotSeg(EpisodicSupportMixin, nn.Module):
    def __init__(self, pretrained_weights="deeplabv3", prototype_size=257):
        super().__init__()
        # Encoder
        self.encoder = Res101Encoder(replace_stride_with_dilation=[True, True, False],
                                     pretrained_weights=pretrained_weights)
        self.prototype_size = prototype_size
        self.context_self_prediction = False
        self.scaler = 20.0
        self.criterion = nn.NLLLoss(ignore_index=255)
        self.criterion_b = BoundaryLoss(theta0=3, theta=5)
        self.criterion_MSE = nn.MSELoss()
        self.fg_num = 100
        self.bg_num = 600
        self.mlp1 = MLP(prototype_size, self.fg_num)
        self.mlp2 = MLP(prototype_size, self.bg_num)
        self.decoder1 = Decoder(self.fg_num)
        self.decoder2 = Decoder(self.bg_num)
        self.supp_decoder = Supp_Decoder()

    def forward(self, supp_imgs, supp_mask, qry_imgs, train=False):
        """One-way, one-shot, one-query episodes; leading tensor dimension is batch.

        Multi-shot fusion is not implemented. Reject it rather than reusing shot-zero
        masks/features for other shots. n_part evaluation calls remain single-shot.
        """
        if len(supp_imgs) != 1 or len(supp_imgs[0]) != 1 or len(qry_imgs) != 1:
            raise ValueError('This model supports exactly 1 way, 1 shot and 1 query per episode')
        self.n_ways = self.n_shots = self.n_queries = 1
        support, mask, query = supp_imgs[0][0], supp_mask[0][0], qry_imgs[0]
        if support.shape[0] != query.shape[0] or mask.shape != support.shape[:1]+support.shape[-2:]:
            raise ValueError('Support/query batch sizes and support mask shape must agree')
        batch, image_size = support.shape[0], mask.shape[-2:]
        features, thresholds = self.encoder(torch.cat((support, query), dim=0))
        raw = features['down2']
        fs, fq = raw[:batch], raw[batch:]
        ts, tq = thresholds[:batch], thresholds[batch:]
        outputs=[]
        losses=[raw.new_zeros(()) for _ in range(4)]
        for episode in range(batch):
            sf, qf, sm = fs[[episode]], fq[[episode]], mask[[episode]]
            self.thresh_pred = [tq[[episode]]]
            sf_bar = F.interpolate(sf, size=image_size, mode='bilinear', align_corners=True)
            global_fg = self.getFeatures(sf, sm)
            global_bg = self.getFeatures(sf, 1-sm)
            if self.context_self_prediction:
                region_bg, region_fg = self.CI(sf_bar, sm, self.kernel, self.stride,
                                             flag=int(bool((sm > 0).any().item())))
                bank_fg = torch.cat((region_fg, global_fg), dim=0)
                bank_bg = torch.cat((region_bg, global_bg), dim=0)
                support_probability = self.getSelfPred(sf_bar, bank_fg, bank_bg)
            else:
                support_probability = self.getSelfPred(sf, global_fg)
            support_probability = F.interpolate(support_probability, size=image_size,
                                                 mode='bilinear', align_corners=True)
            support_2c = binary_probabilities(support_probability)
            support_hard = support_2c.argmax(dim=1)
            fg_pts = [self.get_fg_pts(sf, sm, support_hard)]
            bg_pts = [self.get_bg_pts(sf, sm, support_hard)]
            if self.context_self_prediction:
                fg_input = self.IFR_FD(fg_pts[0], qf, sf, sm, tq[episode], ts[episode])
                bg_input = self.IFR_FD(bg_pts[0], qf, sf, 1-sm, tq[episode], ts[episode])
                fg_logit = self.ifr_decoder_fg(fg_input)
                bg_logit = self.ifr_decoder_bg(bg_input)
                logits = F.interpolate(torch.cat((bg_logit, fg_logit), dim=1), size=image_size,
                                       mode='bilinear', align_corners=True)
                prediction = logits.softmax(dim=1)
            elif not bool((sm > 0).any().item()):
                # Preserve the CT empty-support anomaly-score prediction branch.
                probability = self.getPred(qf, global_fg, tq[[episode]])
                probability = F.interpolate(probability.unsqueeze(1), size=image_size,
                                             mode='bilinear', align_corners=True)
                prediction = binary_probabilities(probability)
            else:
                fg_logit = self.get_fg_sim(qf, fg_pts[0])
                bg_logit = self.get_bg_sim(qf, bg_pts[0])
                logits = F.interpolate(torch.cat((bg_logit, fg_logit), dim=1), size=image_size,
                                       mode='bilinear', align_corners=True)
                prediction = logits.softmax(dim=1)
            outputs.append(prediction)
            if train:
                current = self.align_aux_Loss(sf.unsqueeze(0), qf, sm.unsqueeze(0), prediction,
                                              fg_pts, bg_pts, support_2c)
                losses = [total+value for total,value in zip(losses,current)]
        return (torch.cat(outputs, dim=0),) + tuple(value/batch for value in losses)

    def getPred(self, fts, prototype, thresh):
        """
        Args:
            fts: (1, 512, 64, 64)
            prototype: (1, 512)
            thresh: (1, 1)
        """
        sim = -F.cosine_similarity(fts, prototype[..., None, None], dim=1) * self.scaler
        pred = 1.0 - torch.sigmoid(0.5 * (sim - thresh))

        return pred

    def getSelfPred(self, supp_fts, supp_vec):
        """
        Args:
            supp_fts: 1 x 512 x 64 x 64
            supp_vec: 1 x 512
        """
        supp_vec = supp_vec[..., None, None].expand(-1, -1, supp_fts.shape[-2], supp_fts.shape[-1])
        supp_pred = torch.cat([supp_fts, supp_vec, supp_vec], dim=1)
        supp_logit = self.supp_decoder(supp_pred)

        return torch.sigmoid(supp_logit)

    def getFeatures(self, fts, mask):
        """
        Args:
            fts: (1, 512, 64, 64)
            mask: (1, 256, 256)
        """
        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear', align_corners=True)
        masked_fts = torch.sum(fts * mask[None, ...], dim=(-2, -1)) \
                     / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)

        return masked_fts

    def getPrototype(self, fg_fts):
        """
        Args:
            fg_fts: way x shot x [1 x 512]
        """
        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [torch.sum(torch.cat([tr for tr in way], dim=0), dim=0, keepdim=True) / n_shots
                         for way in fg_fts]

        return fg_prototypes

    def alignLoss(self, supp_fts, qry_fts, pred, fore_mask):
        """
        Args:
            supp_fts: (way, shot, 512, 64, 64)
            qry_fts: (N, 512, 64, 64)
            pred: (1, 2, H, W)
            fore_mask: (way, shot, 256, 256)
        """
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Get query mask
        pred_mask = pred.argmax(dim=1, keepdim=True).squeeze(1)
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=0).float()

        # Compute the support loss
        loss = torch.zeros(1).to(self.device)
        b_loss = torch.zeros(1).to(self.device)
        for way in range(n_ways):
            if way in skip_ways:
                continue
            for shot in range(n_shots):
                qry_fts_ = [[self.getFeatures(qry_fts, pred_mask[way + 1])]]
                fg_prototypes = self.getPrototype(qry_fts_)

                # Get predictions
                supp_pred = self.getPred(supp_fts[way, [shot]], fg_prototypes[way], self.thresh_pred[way])
                supp_pred = F.interpolate(supp_pred[None, ...], size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)

                pred_ups = torch.cat((1.0 - supp_pred, supp_pred), dim=1)

                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255, device=fore_mask.device)
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[fore_mask[way, shot] == 0] = 0

                # Compute Loss
                eps = torch.finfo(torch.float32).eps
                log_prob = torch.log(torch.clamp(pred_ups, eps, 1 - eps))
                loss += self.criterion(log_prob, supp_label[None, ...].long()) / n_shots / n_ways

        return loss, b_loss

    def align_aux_Loss(self, supp_fts, qry_fts, fore_mask, pred, sup_fg_pts, sup_bg_pts, self_pred):
        return self.compute_auxiliary_losses(supp_fts, qry_fts, fore_mask, pred,
                                             sup_fg_pts, sup_bg_pts, self_pred)

    def get_fg_pts(self, features, mask, pred_mask):
        """
        Args:
        features: (1, 512, 64, 64)
        mask: (1, 256, 256)
        pred_mask: (1, 256, 256)
        """
        target_size = (self.prototype_size, self.prototype_size)
        mask = F.interpolate(mask.unsqueeze(1).float(), size=target_size, mode='nearest').squeeze(1)
        pred_mask = F.interpolate(pred_mask.unsqueeze(1).float(), size=target_size, mode='nearest').squeeze(1)
        features_trans = F.interpolate(features, size=target_size, mode='bilinear', align_corners=True)

        ie_mask = mask.squeeze(0) - torch.tensor(cv2.erode(mask.squeeze(0).cpu().numpy(), np.ones((3, 3), dtype=np.uint8), iterations=2)).to(self.device)
        ie_mask = ie_mask.unsqueeze(0)
        ie_prototype = torch.sum(features_trans * ie_mask[None, ...], dim=(-2, -1)) \
                       / (ie_mask[None, ...].sum(dim=(-2, -1)) + 1e-5)
        origin_prototype = torch.sum(features_trans * mask[None, ...], dim=(-2, -1)) \
                           / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)

        add_mask = (pred_mask.float() + mask).long()
        mask1 = torch.zeros_like(mask)
        mask2 = torch.zeros_like(mask)
        mask1[add_mask == 2] = 1
        mask2[add_mask == 1] = 1
        mask1[mask == 0] = 0
        mask2[mask == 0] = 0

        fg_fts = self.get_fg_fts(features_trans, mask)
        fg_prototypes = self.mlp1(fg_fts.view(512, self.prototype_size ** 2)).permute(1, 0)

        if torch.sum(mask2[mask2 == 1]) > 0:
            hard_fg = self.get_random_pts(features_trans, mask2, 50)
            k = random.sample(range(len(fg_prototypes)), 50)
            fg_prototypes = torch.cat([fg_prototypes[k], hard_fg], dim=0)

        fg_prototypes = torch.cat([fg_prototypes, origin_prototype, ie_prototype], dim=0)

        return fg_prototypes

    def get_bg_pts(self, features, mask, pred_mask):
        """
        Args:
            features: (1, 512, 64, 64)
            mask: (1, 256, 256)
            pred_mask: (1, 256, 256)
        """
        target_size = (self.prototype_size, self.prototype_size)
        mask = F.interpolate(mask.unsqueeze(1).float(), size=target_size, mode='nearest').squeeze(1)
        pred_mask = F.interpolate(pred_mask.unsqueeze(1).float(), size=target_size, mode='nearest').squeeze(1)
        features_trans = F.interpolate(features, size=target_size, mode='bilinear', align_corners=True)
        bg_mask = 1 - mask

        oe_mask = torch.tensor(cv2.dilate(mask.squeeze(0).cpu().numpy(), np.ones((3, 3), dtype=np.uint8), iterations=2)).to(self.device) - mask.squeeze(0)
        oe_mask = oe_mask.unsqueeze(0)
        oe_prototype = torch.sum(features_trans * oe_mask[None, ...], dim=(-2, -1)) \
                       / (oe_mask[None, ...].sum(dim=(-2, -1)) + 1e-5)
        origin_prototype = torch.sum(features_trans * bg_mask[None, ...], dim=(-2, -1)) \
                           / (bg_mask[None, ...].sum(dim=(-2, -1)) + 1e-5)

        add_mask = (pred_mask.float() + mask).long()
        mask1 = torch.zeros_like(mask)
        mask2 = torch.zeros_like(mask)
        mask1[add_mask == 0] = 1
        mask2[add_mask == 1] = 1
        mask1[bg_mask == 0] = 0
        mask2[bg_mask == 0] = 0

        bg_fts = self.get_fg_fts(features_trans, bg_mask)
        bg_prototypes = self.mlp2(bg_fts.view(512, self.prototype_size ** 2)).permute(1, 0)

        if torch.sum(mask2[mask2 == 1]) > 0:
            hard_bg = self.get_random_pts(features_trans, mask2, 100)
            k = random.sample(range(len(bg_prototypes)), 500)
            bg_prototypes = torch.cat([bg_prototypes[k], hard_bg], dim=0)

        bg_prototypes = torch.cat([bg_prototypes, origin_prototype, oe_prototype], dim=0)

        return bg_prototypes

    def get_random_pts(self, features_trans, mask, n_prototype):
        return sample_masked_features(features_trans, mask, n_prototype)

    def get_fg_fts(self, fts, mask):
        """
        Args:
            fts: (1, 512, 256, 256)
            mask: (1, 256, 256)
        """
        _, c, h, w = fts.shape
        # select masked fg features
        fg_fts = fts * mask[None, ...]
        bg_fts = torch.ones_like(fts) * mask[None, ...]
        mask_ = mask.view(-1)
        n_pts = len(mask_) - len(mask_[mask_ == 1])
        select_pts = self.get_random_pts(fts, mask, n_pts)
        index = bg_fts == 0
        fg_fts[index] = select_pts.permute(1, 0).reshape(512*n_pts)

        return fg_fts

    def get_all_prototypes(self, fg_fts):
        """
        Args:
            fg_fts: way x shot x [all x 512]
        """
        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        prototypes = [sum([shot for shot in way]) / n_shots for way in fg_fts]

        return prototypes

    def get_fg_sim(self, fts, prototypes):
        """
        Args:
            fts: (1, 512, 64, 64)
            prototypes: (102, 512)
        """
        fts_ = fts.permute(0, 2, 3, 1)
        fts_ = F.normalize(fts_, dim=-1)
        pts_ = F.normalize(prototypes, dim=-1)
        fg_sim = torch.matmul(fts_, pts_.transpose(0, 1)).permute(0, 3, 1, 2)
        fg_sim = self.decoder1(fg_sim)

        return fg_sim

    def get_bg_sim(self, fts, prototypes):
        """
        Args:
            fts: (1, 512, 64, 64)
            prototypes: (602, 512)
        """
        fts_ = fts.permute(0, 2, 3, 1)
        fts_ = F.normalize(fts_, dim=-1)
        pts_ = F.normalize(prototypes, dim=-1)
        bg_sim = torch.matmul(fts_, pts_.transpose(0, 1)).permute(0, 3, 1, 2)
        bg_sim = self.decoder2(bg_sim)

        return bg_sim

    def get_aux_loss(self, sup_fg_pts, qry_fg_pts, sup_bg_pts, qry_bg_pts):
        """
        Args:
            sup_fg_pts: (102, 512)
            qry_fg_pts: (102, 512)
            sup_bg_pts: (602, 512)
            qry_bg_pts: (602, 512)
        """
        d1 = torch.mean(sup_fg_pts, dim=0, keepdim=True)
        d2 = torch.mean(qry_fg_pts, dim=0, keepdim=True)
        b1 = torch.mean(sup_bg_pts, dim=0, keepdim=True)
        b2 = torch.mean(qry_bg_pts, dim=0, keepdim=True)

        d1 = F.normalize(d1, dim=-1)
        d2 = F.normalize(d2, dim=-1)
        b1 = F.normalize(b1, dim=-1)
        b2 = F.normalize(b2, dim=-1)

        fg_intra = torch.matmul(d1, d2.transpose(0, 1)).squeeze(0).squeeze(0)
        bg_intra = torch.matmul(b1, b2.transpose(0, 1)).squeeze(0).squeeze(0)
        intra_loss = 2 - fg_intra - bg_intra

        zero = sup_fg_pts.new_zeros(())
        sup_inter = torch.matmul(d1, b1.transpose(0, 1))
        qry_inter = torch.matmul(d2, b2.transpose(0, 1))
        inter_loss = torch.max(zero, torch.mean(sup_inter)) + torch.max(zero, torch.mean(qry_inter))

        return intra_loss + inter_loss
