import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder import Res101Encoder
from models.sepc import SEPCPrototypeMLP, PrototypeDecoder, SEPCDecoder
import numpy as np
import random
import cv2
from boundary_loss import BoundaryLoss
from models.swpa import SWPADecoder


class BSPModel(nn.Module):
    def __init__(self, pretrained_weights="deeplabv3", ablation_variant="full"):
        super().__init__()
        from models.ablation import module_flags
        self.ablation_variant = ablation_variant
        flags = module_flags(ablation_variant)
        self.use_sepc = flags["sepc"]
        self.use_qrr = flags["qrr"]
        self.use_swpa = flags["swpa"]

        self.encoder = Res101Encoder(replace_stride_with_dilation=[True, True, False],
                                     pretrained_weights=pretrained_weights)
        self.device = torch.device('cuda')
        self.scaler = 20.0
        self.rate = 0.95
        self.kernel = (16, 8)
        self.stride = 8
        self.num = 100

        self.criterion = nn.NLLLoss()

        self.criterion_b = BoundaryLoss(theta0=3, theta=5)

        self.criterion_MSE = nn.MSELoss()

        self.fg_num = 100

        self.bg_num = 600
        self.prototype_chunk_size = 32
        self.ifr_decoder_fg = SWPADecoder(in_ch=1025)
        self.ifr_decoder_bg = SWPADecoder(in_ch=1025)
        self.mlp1 = SEPCPrototypeMLP(256, self.fg_num)
        self.mlp2 = SEPCPrototypeMLP(256, self.bg_num)
        self.fuse = nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0, bias=False)

        self.sim_decoder_fg = PrototypeDecoder(self.fg_num)

        self.sim_decoder_bg = PrototypeDecoder(self.bg_num)

        self.supp_decoder = SEPCDecoder()

    def forward(self, supp_imgs, supp_mask, qry_imgs, train=False):

        self.n_ways = len(supp_imgs)
        self.n_shots = len(supp_imgs[0])
        self.n_queries = len(qry_imgs)
        assert self.n_ways == 1
        assert self.n_queries == 1

        qry_bs = qry_imgs[0].shape[0]
        supp_bs = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]
        supp_mask = torch.stack([torch.stack(way, dim=0) for way in supp_mask],
                                dim=0).view(supp_bs, self.n_ways, self.n_shots, *img_size)

        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0), ], dim=0)
        img_fts, tao = self.encoder(imgs_concat)

        supp_fts = [img_fts[dic][:self.n_ways * self.n_shots * supp_bs].view(
            supp_bs, self.n_ways, self.n_shots, -1, *img_fts[dic].shape[-2:]) for _, dic in enumerate(img_fts)]
        supp_fts = supp_fts[0]

        qry_fts = [img_fts[dic][self.n_ways * self.n_shots * supp_bs:].view(
            qry_bs, self.n_queries, -1, *img_fts[dic].shape[-2:]) for _, dic in enumerate(img_fts)]
        qry_fts = qry_fts[0]

        self.t = tao[self.n_ways * self.n_shots * supp_bs:]
        self.thresh_pred = [self.t for _ in range(self.n_ways)]
        supp_tao = tao[:self.n_ways * self.n_shots * supp_bs].view(supp_bs, self.n_ways, self.n_shots, -1)
        qry_tao = tao[self.n_ways * self.n_shots * supp_bs:].view(qry_bs, self.n_queries, -1)

        align_loss = torch.zeros(1).to(self.device)
        b_loss = torch.zeros(1).to(self.device)
        ssp_loss = torch.zeros(1).to(self.device)
        aux_loss = torch.zeros(1).to(self.device)
        outputs = []
        for epi in range(supp_bs):

            supp_fts_ = [[F.interpolate(supp_fts[[epi], way, shot], size=img_size,
                                        mode='bilinear', align_corners=True)
                          for shot in range(self.n_shots)] for way in range(self.n_ways)]

            prototype = [[self.masked_pool(supp_fts_[way][shot], supp_mask[[epi], way, shot])
                          for shot in range(self.n_shots)] for way in range(self.n_ways)]
            prototype = self.average_prototypes(prototype)

            bg_prototype = [[self.masked_pool(supp_fts_[way][shot], 1 - supp_mask[[epi], way, shot])
                             for shot in range(self.n_shots)] for way in range(self.n_ways)]
            bg_prototype = self.average_prototypes(bg_prototype)

            if self.use_sepc:

                if supp_mask[epi][0].sum() == 0:
                    region_protos = [[self.sepc_region_contexts(supp_fts_[way][shot], supp_mask[[epi], way, shot],
                                                                self.kernel, self.stride, flag=0)
                                      for shot in range(self.n_shots)] for way in range(self.n_ways)]
                else:
                    region_protos = [[self.sepc_region_contexts(supp_fts_[way][shot], supp_mask[[epi], way, shot],
                                                                self.kernel, self.stride, flag=1)
                                      for shot in range(self.n_shots)] for way in range(self.n_ways)]

                bg_pts = [torch.cat([region_protos[way][0][0], bg_prototype[way]], dim=0) for way in range(self.n_ways)]
                fg_pts = [torch.cat([region_protos[way][0][1], prototype[way]], dim=0) for way in range(self.n_ways)]
            else:

                fg_pts, bg_pts = prototype, bg_prototype

            supp_preds = [[self.sepc_self_prediction(
                supp_fts_[way][shot],
                fg_pts[way],
                bg_pts[way]
            ) for shot in range(self.n_shots)] for way in range(self.n_ways)]

            supp_pred = supp_preds[0][0]
            supp_pred = F.interpolate(supp_pred, size=img_size, mode='bilinear', align_corners=True)
            supp_pred = torch.cat((1.0 - supp_pred, supp_pred), dim=1)
            self_pred = supp_pred
            supp_pred = torch.argmax(supp_pred, dim=1, keepdim=True).squeeze(1)

            fg_pts = [[self.sepc_foreground_prototypes(supp_fts[[epi], way, shot], supp_mask[[epi], way, shot], supp_pred)
                       for shot in range(self.n_shots)] for way in range(self.n_ways)]

            fg_pts = self.average_shot_prototypes(fg_pts)
            bg_pts = [[self.sepc_background_prototypes(supp_fts[[epi], way, shot], supp_mask[[epi], way, shot], supp_pred)
                       for shot in range(self.n_shots)] for way in range(self.n_ways)]
            bg_pts = self.average_shot_prototypes(bg_pts)

            fg_matched_fts = torch.stack([
                self.QRR_SWPA(
                    fg_pts[way],
                    qry_fts[epi],
                    supp_fts[[epi], way, 0],
                    supp_mask[[epi], way, 0],
                    qry_tao[epi, 0],
                    supp_tao[epi, way, 0]
                )
                for way in range(self.n_ways)
            ], dim=1)

            bg_matched_fts = torch.stack([
                self.QRR_SWPA(
                    bg_pts[way],
                    qry_fts[epi],
                    supp_fts[[epi], way, 0],
                    1 - supp_mask[[epi], way, 0],
                    qry_tao[epi, 0],
                    supp_tao[epi, way, 0]
                )
                for way in range(self.n_ways)
            ], dim=1)

            fg_pred = torch.cat([
                self.ifr_decoder_fg(fg_matched_fts[:, way]) for way in range(self.n_ways)
            ], dim=1)

            bg_pred = torch.cat([
                self.ifr_decoder_bg(bg_matched_fts[:, way]) for way in range(self.n_ways)
            ], dim=1)

            fg_pred = F.interpolate(fg_pred, size=img_size, mode='bilinear', align_corners=True)
            bg_pred = F.interpolate(bg_pred, size=img_size, mode='bilinear', align_corners=True)

            preds = torch.cat([bg_pred, fg_pred], dim=1)
            preds = torch.softmax(preds, dim=1)

            outputs.append(preds)
            if train:
                align_loss_epi, aux_loss_epi, b_loss_epi, ssp_loss_epi = self.auxiliary_losses(supp_fts[epi], qry_fts[epi], supp_mask[epi], preds,
                                                                                               fg_pts, bg_pts, self_pred)
                align_loss += align_loss_epi
                aux_loss += aux_loss_epi
                b_loss += b_loss_epi
                ssp_loss += ssp_loss_epi

        output = torch.stack(outputs, dim=1)
        output = output.view(-1, *output.shape[2:])

        return output, align_loss / supp_bs, aux_loss / supp_bs, b_loss / supp_bs, ssp_loss / supp_bs

    def predict_global_probability(self, fts, prototype, thresh):

        sim = -F.cosine_similarity(fts, prototype[..., None, None], dim=1) * self.scaler
        pred = 1.0 - torch.sigmoid(0.5 * (sim - thresh))

        return pred

    def sepc_self_prediction(self, supp_fts, fg_pts, bg_pts):

        if not self.use_sepc:
            fg_pts, bg_pts = fg_pts[-1:], bg_pts[-1:]
        B, C, H, W = supp_fts.shape

        fts_hw = supp_fts.permute(0, 2, 3, 1)
        fts_norm = F.normalize(fts_hw, dim=-1)

        fg_norm = F.normalize(fg_pts, dim=-1)
        fg_sim = torch.matmul(fts_norm, fg_norm.t())
        fg_attn = torch.softmax(fg_sim * self.scaler, dim=-1)
        ctx_fg = torch.matmul(fg_attn, fg_pts).permute(0, 3, 1, 2)

        bg_norm = F.normalize(bg_pts, dim=-1)
        bg_sim = torch.matmul(fts_norm, bg_norm.t())
        bg_attn = torch.softmax(bg_sim * self.scaler, dim=-1)
        ctx_bg = torch.matmul(bg_attn, bg_pts).permute(0, 3, 1, 2)

        supp_input = torch.cat([supp_fts, ctx_fg, ctx_bg], dim=1)
        supp_pred = self.supp_decoder(supp_input)

        return supp_pred

    def masked_average_features(self, fts, mask):

        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear', align_corners=True)
        masked_fts = torch.sum(fts * mask[None, ...], dim=(-2, -1)) \
            / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)

        return masked_fts

    def average_prototypes(self, fg_fts):

        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [torch.sum(torch.cat([tr for tr in way], dim=0), dim=0, keepdim=True) / n_shots
                         for way in fg_fts]

        return fg_prototypes

    def alignment_loss(self, supp_fts, qry_fts, pred, fore_mask):

        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        pred_mask = pred.argmax(dim=1, keepdim=True).squeeze(1)
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        fg_pixels = binary_masks[1].sum().item()
        is_skipped = int(binary_masks[1].sum() == 0)
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=0).float()

        loss = torch.zeros(1).to(self.device)
        b_loss = torch.zeros(1).to(self.device)
        for way in range(n_ways):
            if way in skip_ways:
                continue
            for shot in range(n_shots):
                qry_fts_ = [[self.masked_average_features(qry_fts, pred_mask[way + 1])]]
                fg_prototypes = self.average_prototypes(qry_fts_)

                supp_pred = self.predict_global_probability(supp_fts[way, [shot]], fg_prototypes[way], self.thresh_pred[way])
                supp_pred = F.interpolate(supp_pred[None, ...], size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)

                pred_ups = torch.cat((1.0 - supp_pred, supp_pred), dim=1)

                supp_label = torch.full_like(fore_mask[way, shot], 255, device=fore_mask.device)
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[fore_mask[way, shot] == 0] = 0

                eps = torch.finfo(torch.float32).eps
                log_prob = torch.log(torch.clamp(pred_ups, eps, 1 - eps))
                loss += self.criterion(log_prob, supp_label[None, ...].long()) / n_shots / n_ways

        return loss, b_loss

    def auxiliary_losses(self, supp_fts, qry_fts, fore_mask, pred, sup_fg_pts, sup_bg_pts, self_pred):

        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        pred_mask = pred.argmax(dim=1, keepdim=True).squeeze(1)

        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        fg_pixels = binary_masks[1].sum().item()
        is_skipped = int(fg_pixels == 0)
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=0).float()

        loss = torch.zeros(1).to(self.device)
        loss_aux = torch.zeros(1).to(self.device)
        b_loss = torch.zeros(1).to(self.device)
        ssp_loss = torch.zeros(1).to(self.device)
        for way in range(n_ways):
            use_support_fallback = bool((pred_mask[way + 1].sum() == 0).item())

            if use_support_fallback:
                qry_seed_prob = self.sepc_self_prediction(qry_fts, sup_fg_pts[way], sup_bg_pts[way])
                qry_seed_prob = F.interpolate(
                    qry_seed_prob,
                    size=fore_mask.shape[-2:],
                    mode='bilinear',
                    align_corners=True
                )

                qry_seed_2c = torch.cat((1.0 - qry_seed_prob, qry_seed_prob), dim=1)
                qry_seed_mask = torch.argmax(qry_seed_2c, dim=1, keepdim=True).squeeze(1)

                if qry_seed_mask.sum().item() == 0:
                    continue

                cur_qry_mask = qry_seed_mask.float()

            else:
                cur_qry_mask = pred_mask[way + 1]

            for shot in range(n_shots):
                qry_fts_ = [[self.masked_average_features(qry_fts, cur_qry_mask)]]
                fg_prototypes = self.average_prototypes(qry_fts_)

                qry_bg_fts_ = [[self.masked_average_features(qry_fts, 1.0 - cur_qry_mask)]]
                bg_prototypes = self.average_prototypes(qry_bg_fts_)

                qry_pred = self.sepc_self_prediction(qry_fts, fg_prototypes[0], bg_prototypes[0])
                qry_pred = F.interpolate(qry_pred, size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)
                qry_pred = torch.cat((1.0 - qry_pred, qry_pred), dim=1)
                qry_pred = torch.argmax(qry_pred, dim=1, keepdim=True).squeeze(1)

                fg_pts_ = [[self.sepc_foreground_prototypes(qry_fts, cur_qry_mask, qry_pred)]]
                fg_pts_ = self.average_shot_prototypes(fg_pts_)

                bg_pts_ = [[self.sepc_background_prototypes(qry_fts, cur_qry_mask, qry_pred)]]
                bg_pts_ = self.average_shot_prototypes(bg_pts_)

                loss_aux += self.prototype_discrimination_loss(sup_fg_pts[way], fg_pts_[way], sup_bg_pts[way], bg_pts_[way])

                supp_pred = self.foreground_similarity(supp_fts[way, [shot]], fg_pts_[way])
                bg_pred_ = self.background_similarity(supp_fts[way, [shot]], bg_pts_[way])
                supp_pred = F.interpolate(supp_pred, size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)
                bg_pred_ = F.interpolate(bg_pred_, size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)

                preds = torch.cat([bg_pred_, supp_pred], dim=1)
                preds = torch.softmax(preds, dim=1)

                supp_label = torch.full_like(fore_mask[way, shot], 255, device=fore_mask.device)
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[fore_mask[way, shot] == 0] = 0

                eps = torch.finfo(torch.float32).eps
                log_prob = torch.log(torch.clamp(preds, eps, 1 - eps))
                loss += self.criterion(log_prob, supp_label[None, ...].long()) / n_shots / n_ways

                ssp_log_prob = torch.log(torch.clamp(self_pred, eps, 1 - eps))
                ssp_loss += self.criterion(ssp_log_prob, supp_label[None, ...].long()) / n_shots / n_ways

        return loss, loss_aux, b_loss, ssp_loss

    def sepc_foreground_prototypes(self, features, mask, pred_mask):

        target_size = (256, 256)
        features_trans = F.interpolate(features, size=target_size, mode='bilinear', align_corners=True)
        mask = F.interpolate(mask.unsqueeze(1).float(), size=target_size, mode='nearest').squeeze(1)
        pred_mask = F.interpolate(pred_mask.unsqueeze(1).float(), size=target_size, mode='nearest').squeeze(1)

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

        fg_fts = self.fill_class_features(features_trans, mask)

        fg_prototypes = self.mlp1(fg_fts.view(512, 256 * 256)).permute(1, 0)

        if not self.use_sepc:
            mask2 = mask
        if torch.sum(mask2[mask2 == 1]) > 0:

            hard_fg = self.sample_prototype_features(features_trans, mask2, 50)
            k = random.sample(range(len(fg_prototypes)), 50)
            fg_prototypes = torch.cat([fg_prototypes[k], hard_fg], dim=0)
        fg_pts = torch.cat([
            fg_prototypes,
            ie_prototype,
            origin_prototype
        ], dim=0)

        return fg_pts

    def sepc_background_prototypes(self, features, mask, pred_mask):

        target_size = (256, 256)

        features_trans = F.interpolate(features, size=target_size, mode='bilinear', align_corners=True)
        mask = F.interpolate(mask.unsqueeze(1).float(), size=target_size, mode='nearest').squeeze(1)
        pred_mask = F.interpolate(pred_mask.unsqueeze(1).float(), size=target_size, mode='nearest').squeeze(1)

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

        bg_fts = self.fill_class_features(features_trans, bg_mask)

        bg_prototypes = self.mlp2(bg_fts.view(512, 256 * 256)).permute(1, 0)

        if not self.use_sepc:
            mask2 = bg_mask
        if torch.sum(mask2[mask2 == 1]) > 0:
            hard_bg = self.sample_prototype_features(features_trans, mask2, 100)
            k = random.sample(range(len(bg_prototypes)), 500)
            bg_prototypes = torch.cat([bg_prototypes[k], hard_bg], dim=0)

        bg_pts = torch.cat([
            bg_prototypes,
            oe_prototype,
            origin_prototype
        ], dim=0)

        return bg_pts

    def sample_prototype_features(self, features_trans, mask, n_prototype):

        features_trans = features_trans.squeeze(0)
        features_trans = features_trans.permute(1, 2, 0)
        features_trans = features_trans.view(features_trans.shape[-2] * features_trans.shape[-3],
                                             features_trans.shape[-1])
        mask = mask.squeeze(0).view(-1)
        features_trans = features_trans[mask == 1]
        if len(features_trans) >= n_prototype:
            k = random.sample(range(len(features_trans)), n_prototype)
            prototypes = features_trans[k]
        else:
            if len(features_trans) == 0:
                prototypes = torch.zeros(n_prototype, 512).to(self.device)
            else:
                r = n_prototype // len(features_trans)
                k = random.sample(range(len(features_trans)), (n_prototype - len(features_trans)) % len(features_trans))
                prototypes = torch.cat([features_trans for _ in range(r)], dim=0)
                prototypes = torch.cat([features_trans[k], prototypes], dim=0)

        return prototypes

    def QRR_SWPA(self, pts, qry_fts, sup_fts, mask, qry_thresh, sup_thresh):

        n, c, h, w = qry_fts.shape

        prototype = pts[-1].unsqueeze(0)
        pts = pts[:-1]

        pts = F.normalize(pts, dim=-1)
        prototype = F.normalize(prototype, dim=-1)

        qry_scale = 1 + self.scaler * torch.sigmoid(qry_thresh)
        sup_scale = 1 + self.scaler * torch.sigmoid(sup_thresh)

        chunk_size = self.prototype_chunk_size
        res = []

        for i in range(n):
            qry_ft = qry_fts[i].unsqueeze(0)
            fts_ = qry_ft.permute(0, 2, 3, 1)
            fts_ = F.normalize(fts_, dim=-1)

            one_sim = torch.matmul(fts_, prototype.transpose(0, 1)).permute(0, 3, 1, 2)

            K = pts.shape[0]
            if self.use_qrr:
                G = self.qrr_fusion_factor(fts_, sup_fts, mask)
                refined_chunks = []

                for start in range(0, K, chunk_size):
                    end = min(start + chunk_size, K)
                    pts_chunk = pts[start:end]

                    sim_chunk = torch.matmul(fts_, pts_chunk.transpose(0, 1)).permute(3, 0, 1, 2)
                    pseudo_mask = torch.sigmoid(sim_chunk * qry_scale)

                    qry_update = torch.sum(qry_ft * pseudo_mask, dim=(-2, -1)) / \
                        (pseudo_mask.sum(dim=(-2, -1)) + 1e-5)

                    fuse_in = torch.stack([pts_chunk * (1 - G), qry_update * G], dim=1)
                    fuse_in = fuse_in.unsqueeze(-1)
                    delta = self.fuse(fuse_in).squeeze(1).squeeze(-1)

                    pts_refined_chunk = pts_chunk + delta
                    refined_chunks.append(pts_refined_chunk)

                    del sim_chunk, pseudo_mask, qry_update, fuse_in, delta, pts_refined_chunk

                pts_refined = torch.cat(refined_chunks, dim=0)
            else:
                pts_refined = pts

            pts_norm = F.normalize(pts_refined, dim=-1)

            if not self.use_swpa:

                matched_pts = pts_refined.mean(dim=0).view(1, c, 1, 1).expand(1, c, h, w)
                mean_response = torch.matmul(fts_, pts_norm.mean(dim=0).view(c, 1))
                mean_response = mean_response.permute(0, 3, 1, 2) * sup_scale
                res.append(torch.cat([qry_ft, matched_pts, mean_response], dim=1))
                continue

            global_max = None
            for start in range(0, K, chunk_size):
                end = min(start + chunk_size, K)
                pts_chunk = pts_norm[start:end]

                logits_chunk = torch.matmul(fts_, pts_chunk.transpose(0, 1)).permute(3, 0, 1, 2)
                logits_chunk = logits_chunk * sup_scale

                chunk_max = logits_chunk.max(dim=0, keepdim=True).values
                global_max = chunk_max if global_max is None else torch.maximum(global_max, chunk_max)

                del logits_chunk, chunk_max

            denom = torch.zeros_like(global_max)
            for start in range(0, K, chunk_size):
                end = min(start + chunk_size, K)
                pts_chunk = pts_norm[start:end]

                logits_chunk = torch.matmul(fts_, pts_chunk.transpose(0, 1)).permute(3, 0, 1, 2)
                logits_chunk = logits_chunk * sup_scale

                exp_chunk = torch.exp(logits_chunk - global_max)
                denom = denom + exp_chunk.sum(dim=0, keepdim=True)

                del logits_chunk, exp_chunk

            sim_0_num = torch.zeros((1, 1, h, w), device=qry_ft.device, dtype=qry_ft.dtype)
            matched_pts_num = torch.zeros((1, c, h, w), device=qry_ft.device, dtype=qry_ft.dtype)

            for start in range(0, K, chunk_size):
                end = min(start + chunk_size, K)

                pts_chunk_norm = pts_norm[start:end]
                pts_chunk_raw = pts_refined[start:end]

                logits_chunk = torch.matmul(fts_, pts_chunk_norm.transpose(0, 1)).permute(3, 0, 1, 2)
                logits_chunk = logits_chunk * sup_scale

                exp_chunk = torch.exp(logits_chunk - global_max)
                weight_chunk = exp_chunk / (denom + 1e-12)

                sim_0_num = sim_0_num + (logits_chunk * weight_chunk).sum(dim=0, keepdim=True)

                matched_pts_chunk = pts_chunk_raw.unsqueeze(2).unsqueeze(3) * weight_chunk
                matched_pts_num = matched_pts_num + matched_pts_chunk.sum(dim=0, keepdim=True)

                del logits_chunk, exp_chunk, weight_chunk, matched_pts_chunk

            sim_0 = sim_0_num.squeeze(0)
            matched_pts = matched_pts_num

            matched_fts = torch.cat([qry_ft, matched_pts, sim_0.unsqueeze(0)], dim=1)
            res.append(matched_fts)

            del pts_refined, pts_norm, global_max, denom, sim_0_num, matched_pts_num, matched_pts, one_sim

        res = torch.cat(res, dim=0)
        return res

    def qrr_fusion_factor(self, qry_fts, sup_fts, mask):

        mask_ = F.interpolate(
            mask.unsqueeze(0).float(),
            size=sup_fts.shape[-2:],
            mode='bilinear',
            align_corners=True
        ).squeeze(0)

        if mask_.sum() == 0:
            G = torch.tensor(0.0, device=self.device)
        else:
            sup_fts_ = F.normalize(
                sup_fts.permute(0, 2, 3, 1).reshape(-1, sup_fts.shape[1]), dim=-1
            )

            qry_fts_ = qry_fts.reshape(-1, qry_fts.shape[-1])
            G = torch.matmul(sup_fts_, qry_fts_.transpose(0, 1))
            G = G[mask_.reshape(-1) > 0]
            G = torch.sigmoid(torch.mean(G))

        return G

    def fill_class_features(self, fts, mask):

        _, c, h, w = fts.shape

        fg_fts = fts * mask[None, ...]
        bg_fts = torch.ones_like(fts) * mask[None, ...]
        mask_ = mask.view(-1)
        n_pts = len(mask_) - len(mask_[mask_ == 1])

        select_pts = self.sample_prototype_features(fts, mask, n_pts)
        index = bg_fts == 0
        fg_fts[index] = select_pts.permute(1, 0).reshape(512 * n_pts)

        return fg_fts

    def average_shot_prototypes(self, fg_fts):

        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        prototypes = [sum([shot for shot in way]) / n_shots for way in fg_fts]

        return prototypes

    def foreground_similarity(self, fts, prototypes):

        fts_ = fts.permute(0, 2, 3, 1)
        fts_ = F.normalize(fts_, dim=-1)
        pts_ = F.normalize(prototypes, dim=-1)
        fg_sim = torch.matmul(fts_, pts_.transpose(0, 1)).permute(0, 3, 1, 2)

        fg_sim = self.sim_decoder_fg(fg_sim)

        return fg_sim

    def background_similarity(self, fts, prototypes):

        fts_ = fts.permute(0, 2, 3, 1)
        fts_ = F.normalize(fts_, dim=-1)
        pts_ = F.normalize(prototypes, dim=-1)
        bg_sim = torch.matmul(fts_, pts_.transpose(0, 1)).permute(0, 3, 1, 2)
        bg_sim = self.sim_decoder_bg(bg_sim)

        return bg_sim

    def masked_pool(self, fts, mask):

        masked_fts = torch.sum(fts * mask[None, ...], dim=(-2, -1)) \
            / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)

        return masked_fts

    def average_prototypes(self, fg_fts):

        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [torch.sum(torch.cat([tr for tr in way], dim=0), dim=0, keepdim=True) / n_shots for way in
                         fg_fts]

        return fg_prototypes

    def sepc_region_contexts(self, fts, mask, kernel_size=(16, 16), stride=8, flag=1, aspect_ratio_range=(0.5, 2)):

        b, c, height, width = fts.size()
        kernel_h, kernel_w = kernel_size
        n_fg = mask.sum()
        n_bg = height * width - n_fg
        fg_pts = []
        bg_pts = []
        fg_pts.append(torch.sum(fts * mask[None, ...], dim=(-2, -1)) / (n_fg + 1e-5))
        bg_pts.append(torch.sum(fts * (1 - mask[None, ...]), dim=(-2, -1)) / (n_bg + 1e-5))
        patch_fts = F.unfold(fts, kernel_size=kernel_size, stride=stride).permute(0, 2, 1).view(1, -1, c, kernel_h,
                                                                                                kernel_w)
        patch_mask = F.unfold(mask.unsqueeze(1), kernel_size=kernel_size, stride=stride).permute(0, 2, 1).view(1, -1,
                                                                                                               kernel_h,
                                                                                                               kernel_w).unsqueeze(
            2)
        bg_patch_mask = 1 - patch_mask
        random.seed(9)
        if flag == 1:
            fg_indx = torch.sum(patch_mask, dim=(-2, -1)).view(-1) >= min(kernel_h * kernel_w * self.rate, n_fg / 150)
            fg_protos = torch.sum(patch_fts[:, fg_indx, :, :, :] * patch_mask[:, fg_indx, :, :, :], dim=(-2, -1)) \
                / (patch_mask[:, fg_indx, :, :, :].sum(dim=(-2, -1)) + 1e-5)
            if fg_protos.numel() == 0:
                fg_protos = self.sample_region_features(fts, mask)
            else:
                fg_protos = fg_protos.squeeze(0)
                if len(fg_protos) > 150:
                    fg_protos = fg_protos[random.sample(range(len(fg_protos)), 150)]
            fg_pts.append(fg_protos)

        bg_indx = torch.sum(bg_patch_mask, dim=(-2, -1)).view(-1) >= min(kernel_h * kernel_w, n_bg / 250)
        bg_protos = torch.sum(patch_fts[:, bg_indx, :, :, :] * bg_patch_mask[:, bg_indx, :, :, :], dim=(-2, -1)) \
            / (bg_patch_mask[:, bg_indx, :, :, :].sum(dim=(-2, -1)) + 1e-5)
        if bg_protos.numel() == 0:
            bg_protos = self.sample_region_features(fts, 1 - mask)
        else:
            bg_protos = bg_protos.squeeze(0)
            if len(bg_protos) > 250:
                bg_protos = bg_protos[random.sample(range(len(bg_protos)), 250)]
        bg_pts.append(bg_protos)

        min_aspect_ratio, max_aspect_ratio = aspect_ratio_range
        assert min_aspect_ratio > 0
        assert max_aspect_ratio > min_aspect_ratio
        torch.manual_seed(9)
        aspect_ratio = torch.rand(self.num).to(self.device) * (max_aspect_ratio - min_aspect_ratio) + min_aspect_ratio
        aspect_ratio0 = aspect_ratio[aspect_ratio <= 1]
        bbox_height0 = (
            (torch.rand(len(aspect_ratio0)).to(self.device) * 0.9375 + 0.0625) * (height * aspect_ratio0)).int()
        bbox_width0 = (bbox_height0 / aspect_ratio0).int()

        aspect_ratio1 = aspect_ratio[aspect_ratio > 1]
        bbox_width1 = (
            (torch.rand(len(aspect_ratio1)).to(self.device) * 0.9375 + 0.0625) * (width / aspect_ratio1)).int()
        bbox_height1 = (bbox_width1 * aspect_ratio1).int()

        bbox_height = torch.cat([bbox_height0, bbox_height1], dim=0)
        bbox_width = torch.cat([bbox_width0, bbox_width1], dim=0)

        x1 = (torch.rand(self.num).to(self.device) * (height - bbox_height)).int()
        y1 = (torch.rand(self.num).to(self.device) * (width - bbox_width)).int()
        x2 = x1 + bbox_height
        y2 = y1 + bbox_width
        for i in range(self.num):
            n1 = mask[:, x1[i]:x2[i], y1[i]:y2[i]].sum()
            n0 = (1 - mask[:, x1[i]:x2[i], y1[i]:y2[i]]).sum()
            if n1 >= min(bbox_height[i] * bbox_width[i] * self.rate, n_fg / self.num) and n1 < n_fg:
                prototype = self.masked_pool(fts[:, :, x1[i]:x2[i], y1[i]:y2[i]], mask[:, x1[i]:x2[i], y1[i]:y2[i]])
                fg_pts.append(prototype)
            if n0 >= min(bbox_height[i] * bbox_width[i], n_bg / self.num) and n0 < n_bg:
                prototype = self.masked_pool(fts[:, :, x1[i]:x2[i], y1[i]:y2[i]],
                                             1 - mask[:, x1[i]:x2[i], y1[i]:y2[i]])
                bg_pts.append(prototype)

        fg_pts = torch.cat(fg_pts, dim=0)
        bg_pts = torch.cat(bg_pts, dim=0)

        return [bg_pts, fg_pts]

    def sample_region_features(self, fts, mask):
        features_trans = fts.squeeze(0)
        features_trans = features_trans.permute(1, 2, 0)
        features_trans = features_trans.view(features_trans.shape[-2] * features_trans.shape[-3],
                                             features_trans.shape[-1])
        mask = mask.squeeze(0).view(-1)
        indx = mask == 1
        features_trans = features_trans[indx]
        if len(features_trans) > 100:
            random.seed(9)
            features_trans = features_trans[random.sample(range(len(features_trans)), 100)]
        return features_trans

    def prototype_discrimination_loss(self, sup_fg_pts, qry_fg_pts, sup_bg_pts, qry_bg_pts):

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

        zero = torch.zeros(1).to(self.device).squeeze(0)
        sup_inter = torch.matmul(d1, b1.transpose(0, 1))
        qry_inter = torch.matmul(d2, b2.transpose(0, 1))
        inter_loss = torch.max(zero, torch.mean(sup_inter)) + torch.max(zero, torch.mean(qry_inter))

        return intra_loss + inter_loss
