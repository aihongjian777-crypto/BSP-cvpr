# """
# FSS via CoW
# Extended from ADNet code by Hansen et al.
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from models.encoder import Res101Encoder
# from models.modules import MLP, Decoder, Supp_Decoder
# import numpy as np
# import random
# import cv2
# from boundary_loss import BoundaryLoss
# from models.Decoders import Decoder as IFRDecoder


# class FewShotSeg(nn.Module):
#     def __init__(self, pretrained_weights="deeplabv3"):
#         super().__init__()
#         # Encoder
#         self.encoder = Res101Encoder(replace_stride_with_dilation=[True, True, False],
#                                      pretrained_weights=pretrained_weights)
#         self.device = torch.device('cuda')
#         self.scaler = 20.0
#         self.rate = 0.95
#         self.kernel = (16, 8)
#         self.stride = 8
#         self.num = 100
#         #它与 LogSoftmax（或在这个代码中的 torch.log(torch.softmax(...))）配合使用，用来衡量模型预测的像素类别与真实标签之间的差距。
#         self.criterion = nn.NLLLoss()
#         #专门用来惩罚模型在物体边缘（边界）处预测不准确的情况，能让分割结果的边缘更清晰、更贴合真实物体。
#         self.criterion_b = BoundaryLoss(theta0=3, theta=5)
#         #ean Squared Error Loss (均方误差损失)。通常用于回归任务（预测一个连续的数值）。在这个项目中，它很可能是用在 get_aux_loss（辅助损失）中，用来拉近支持集原型和查询集原型（它们都是特征向量）之间的距离。
#         self.criterion_MSE = nn.MSELoss()
#         #定义了要从“弱点”区域挖掘 100 个困难前景 (Hard Foreground) 原型。
#         self.fg_num = 100
#         #定义了要挖掘 600 个困难背景 (Hard Background) 原型。
#         self.bg_num = 600
#         self.ifr_decoder_fg = IFRDecoder(in_ch=1025)
#         self.ifr_decoder_bg = IFRDecoder(in_ch=1025)
#         self.mlp1 = MLP(256, self.fg_num)
#         self.mlp2 = MLP(256, self.bg_num)
#         self.fuse = nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0, bias=False)
#         #它的输入是查询集特征与 100 个前景原型计算出的“相似度图”，输出是最终的前景分割图。它负责**“融合”**这 100 个原型的意见。
#         self.sim_decoder_fg = Decoder(self.fg_num)
#         #同上，负责**“融合”** 600 个背景原型的意见，生成背景分割图。
#         self.sim_decoder_bg = Decoder(self.bg_num)
#         #它用于支持集自我预测 (SSP) 模块。在 getSelfPred 方法中，它接收支持集特征 F_s 和朴素原型 p 的拼接结果，并输出一个初始的、粗糙的自我预测掩码 M_sp
#         self.supp_decoder = Supp_Decoder()

#     def forward(self, supp_imgs, supp_mask, qry_imgs, train=False):
#         """
#         Args:
#             supp_imgs: support images
#                 way x shot x [B x 3 x H x W]
#             supp_mask: foreground masks for support images
#                 way x shot x [B x H x W]
#             qry_imgs: query images
#                 N x [B x 3 x H x W]
#             train: whether to train model or not
#         """
#         self.n_ways = len(supp_imgs)
#         self.n_shots = len(supp_imgs[0])
#         self.n_queries = len(qry_imgs)
#         assert self.n_ways == 1  # for now only one-way, because not every shot has multiple sub-images
#         assert self.n_queries == 1

#         qry_bs = qry_imgs[0].shape[0]
#         supp_bs = supp_imgs[0][0].shape[0]
#         img_size = supp_imgs[0][0].shape[-2:]
#         supp_mask = torch.stack([torch.stack(way, dim=0) for way in supp_mask],
#                                 dim=0).view(supp_bs, self.n_ways, self.n_shots, *img_size)

#         ###### Extract features ######
#         imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
#                                 + [torch.cat(qry_imgs, dim=0), ], dim=0)
#         img_fts, tao = self.encoder(imgs_concat)

#         supp_fts = [img_fts[dic][:self.n_ways * self.n_shots * supp_bs].view(
#             supp_bs, self.n_ways, self.n_shots, -1, *img_fts[dic].shape[-2:]) for _, dic in enumerate(img_fts)]
#         supp_fts = supp_fts[0]

#         qry_fts = [img_fts[dic][self.n_ways * self.n_shots * supp_bs:].view(
#             qry_bs, self.n_queries, -1, *img_fts[dic].shape[-2:]) for _, dic in enumerate(img_fts)]
#         qry_fts = qry_fts[0]

#         ###### Get threshold ######
#         self.t = tao[self.n_ways * self.n_shots * supp_bs:]
#         self.thresh_pred = [self.t for _ in range(self.n_ways)]
#         supp_tao = tao[:self.n_ways * self.n_shots * supp_bs].view(supp_bs, self.n_ways, self.n_shots, -1)
#         qry_tao = tao[self.n_ways * self.n_shots * supp_bs:].view(qry_bs, self.n_queries, -1)

#         ###### Compute loss ######
#         align_loss = torch.zeros(1).to(self.device)
#         b_loss = torch.zeros(1).to(self.device)
#         ssp_loss = torch.zeros(1).to(self.device)
#         aux_loss = torch.zeros(1).to(self.device)
#         outputs = []
#         for epi in range(supp_bs):
#             ###### Extract prototypes ######
#             # 1. 上采样 support feature 到 mask 尺寸
#             supp_fts_ = [[F.interpolate(supp_fts[[epi], way, shot], size=img_size,
#                                         mode='bilinear', align_corners=True)
#                           for shot in range(self.n_shots)] for way in range(self.n_ways)]

#             # 2. 提取全局前景原型 / 全局背景原型
#             prototype = [[self.get_masked_fts(supp_fts_[way][shot], supp_mask[[epi], way, shot])
#                           for shot in range(self.n_shots)] for way in range(self.n_ways)]
#             prototype = self.getPrototype(prototype)

#             bg_prototype = [[self.get_masked_fts(supp_fts_[way][shot], 1 - supp_mask[[epi], way, shot])
#                              for shot in range(self.n_shots)] for way in range(self.n_ways)]
#             bg_prototype = self.getPrototype(bg_prototype)

#             # 3. 用 CI 提取区域原型
#             if supp_mask[epi][0].sum() == 0:
#                 region_protos = [[self.CI(supp_fts_[way][shot], supp_mask[[epi], way, shot],
#                                           self.kernel, self.stride, flag=0)
#                                   for shot in range(self.n_shots)] for way in range(self.n_ways)]
#             else:
#                 region_protos = [[self.CI(supp_fts_[way][shot], supp_mask[[epi], way, shot],
#                                           self.kernel, self.stride, flag=1)
#                                   for shot in range(self.n_shots)] for way in range(self.n_ways)]

#             bg_pts = [torch.cat([region_protos[way][0][0], bg_prototype[way]], dim=0) for way in range(self.n_ways)]
#             fg_pts = [torch.cat([region_protos[way][0][1], prototype[way]], dim=0) for way in range(self.n_ways)]



#             # support self-prediction using prototype groups from CI
#             supp_preds = [[self.getSelfPred(
#                 supp_fts_[way][shot],  # [1,C,H,W]
#                 fg_pts[way],  # fg_pts
#                 bg_pts[way]  # bg_pts
#             ) for shot in range(self.n_shots)] for way in range(self.n_ways)]

#             # current 1-way 1-shot case
#             supp_pred = supp_preds[0][0]
#             supp_pred = F.interpolate(supp_pred, size=img_size, mode='bilinear', align_corners=True)
#             supp_pred = torch.cat((1.0 - supp_pred, supp_pred), dim=1)
#             self_pred = supp_pred
#             supp_pred = torch.argmax(supp_pred, dim=1, keepdim=True).squeeze(1)
#             #supp_pred[1, 512, 256, 256]

#             fg_pts = [[self.get_fg_pts(supp_fts[[epi], way, shot], supp_mask[[epi], way, shot], supp_pred)
#                        for shot in range(self.n_shots)] for way in range(self.n_ways)]
#             # fg_pts = [
#             #              < 张量
#             # shape = [102, 512] >,  # Way 0 的最终平均团队
#             # < 张量
#             # shape = [102, 512] >  # Way 1 的最终平均团队
#             # ]   变得更平均了  少了shot的一个嵌套
#             fg_pts = self.get_all_prototypes(fg_pts)
#             bg_pts = [[self.get_bg_pts(supp_fts[[epi], way, shot], supp_mask[[epi], way, shot], supp_pred)
#                        for shot in range(self.n_shots)] for way in range(self.n_ways)]
#             bg_pts = self.get_all_prototypes(bg_pts)

#             # ----- threshold split -----
#             # suppose tao shape is [num_images, 1] or compatible scalar tensor

#             # =========================
#             # Query prediction via IFR_FD
#             # =========================
#             fg_matched_fts = torch.stack([
#                 self.IFR_FD(
#                     fg_pts[way],  # (102, 512), last one is global fg prototype
#                     qry_fts[epi],  # (1, 512, H', W')
#                     supp_fts[[epi], way, 0],  # (1, 512, H', W')
#                     supp_mask[[epi], way, 0],  # (1, H, W)
#                     qry_tao[epi, 0],  # query threshold
#                     supp_tao[epi, way, 0]  # support threshold
#                 )
#                 for way in range(self.n_ways)
#             ], dim=1)  # (Nq, Wa, 1025, H', W')

#             bg_matched_fts = torch.stack([
#                 self.IFR_FD(
#                     bg_pts[way],  # (602, 512), last one is global bg prototype
#                     qry_fts[epi],  # (1, 512, H', W')
#                     supp_fts[[epi], way, 0],  # (1, 512, H', W')
#                     1 - supp_mask[[epi], way, 0],  # background mask
#                     qry_tao[epi, 0],
#                     supp_tao[epi, way, 0]
#                 )
#                 for way in range(self.n_ways)
#             ], dim=1)  # (Nq, Wa, 1025, H', W')

#             fg_pred = torch.cat([
#                 self.ifr_decoder_fg(fg_matched_fts[:, way]) for way in range(self.n_ways)
#             ], dim=1)  # (Nq, Wa, H', W') or (Nq, Wa, 1, H', W') depending on decoder

#             bg_pred = torch.cat([
#                 self.ifr_decoder_bg(bg_matched_fts[:, way]) for way in range(self.n_ways)
#             ], dim=1)

#             fg_pred = F.interpolate(fg_pred, size=img_size, mode='bilinear', align_corners=True)
#             bg_pred = F.interpolate(bg_pred, size=img_size, mode='bilinear', align_corners=True)

#             preds = torch.cat([bg_pred, fg_pred], dim=1)
#             preds = torch.softmax(preds, dim=1)

#             outputs.append(preds)
#             if train:
#                 align_loss_epi, aux_loss_epi, b_loss_epi, ssp_loss_epi = self.align_aux_Loss(supp_fts[epi], qry_fts[epi], supp_mask[epi], preds,
#                                                                                              fg_pts, bg_pts, self_pred)
#                 align_loss += align_loss_epi
#                 aux_loss += aux_loss_epi
#                 b_loss += b_loss_epi
#                 ssp_loss += ssp_loss_epi
#             #您的写法 (推荐)：
#             # Python
#             #
#             # output = torch.cat(outputs, dim=0)评价：脱裤子放屁——多此一举。
#         output = torch.stack(outputs, dim=1)
#         output = output.view(-1, *output.shape[2:])

#         return output, align_loss / supp_bs, aux_loss / supp_bs, b_loss / supp_bs, ssp_loss / supp_bs

#     def getPred(self, fts, prototype, thresh):
#         """
#         Args:
#             fts: (1, 512, 64, 64)
#             prototype: (1, 512)
#             thresh: (1, 1)
#         """
#         sim = -F.cosine_similarity(fts, prototype[..., None, None], dim=1) * self.scaler
#         pred = 1.0 - torch.sigmoid(0.5 * (sim - thresh))

#         return pred

#     def getSelfPred(self, supp_fts, fg_pts, bg_pts):
#         """
#         Args:
#             supp_fts: (1, 512, H, W)
#             fg_pts:   (N_fg, 512)
#             bg_pts:   (N_bg, 512)
#         Returns:
#             supp_pred: (1, 1, H, W)
#         """
#         B, C, H, W = supp_fts.shape

#         # (1, H, W, C)
#         fts_hw = supp_fts.permute(0, 2, 3, 1)
#         fts_norm = F.normalize(fts_hw, dim=-1)

#         # ---- foreground context ----
#         fg_norm = F.normalize(fg_pts, dim=-1)  # (N_fg, C)
#         fg_sim = torch.matmul(fts_norm, fg_norm.t())  # (1, H, W, N_fg)
#         fg_attn = torch.softmax(fg_sim * self.scaler, dim=-1)
#         ctx_fg = torch.matmul(fg_attn, fg_pts).permute(0, 3, 1, 2)  # (1, C, H, W)

#         # ---- background context ----
#         bg_norm = F.normalize(bg_pts, dim=-1)  # (N_bg, C)
#         bg_sim = torch.matmul(fts_norm, bg_norm.t())  # (1, H, W, N_bg)
#         bg_attn = torch.softmax(bg_sim * self.scaler, dim=-1)
#         ctx_bg = torch.matmul(bg_attn, bg_pts).permute(0, 3, 1, 2)  # (1, C, H, W)

#         # ---- original feature + fg context + bg context ----
#         supp_input = torch.cat([supp_fts, ctx_fg, ctx_bg], dim=1)  # (1, 1536, H, W)
#         supp_pred = self.supp_decoder(supp_input)

#         return supp_pred
#     #的作用是计算支持集图像前景区域的【平均特征】
#     def getFeatures(self, fts, mask):
#         """
#         Arg0.s:
#              fts: (1, 512, 64, 64)
#             mask: (1, 256, 256)
#         """
#         fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear', align_corners=True)
#         masked_fts = torch.sum(fts * mask[None, ...], dim=(-2, -1)) \
#                      / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)

#         return masked_fts

#     def getPrototype(self, fg_fts):
#         """
#         Args:
#             fg_fts: way x shot x [1 x 512]
#         """
#         n_ways, n_shots = len(fg_fts), len(fg_fts[0])
#         fg_prototypes = [torch.sum(torch.cat([tr for tr in way], dim=0), dim=0, keepdim=True) / n_shots
#                          for way in fg_fts]

#         return fg_prototypes

#     def alignLoss(self, supp_fts, qry_fts, pred, fore_mask):
#         """
#         Args:
#             supp_fts: (way, shot, 512, 64, 64)
#             qry_fts: (N, 512, 64, 64)
#             pred: (1, 2, H, W)
#             fore_mask: (way, shot, 256, 256)
#         """
#         n_ways, n_shots = len(fore_mask), len(fore_mask[0])

#         # Get query mask
#         pred_mask = pred.argmax(dim=1, keepdim=True).squeeze(1)
#         binary_masks = [pred_mask == i for i in range(1 + n_ways)]
#         fg_pixels = binary_masks[1].sum().item()
#         is_skipped = int(binary_masks[1].sum() == 0)
#         skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
#         pred_mask = torch.stack(binary_masks, dim=0).float()

#         # Compute the support loss
#         loss = torch.zeros(1).to(self.device)
#         b_loss = torch.zeros(1).to(self.device)
#         for way in range(n_ways):
#             if way in skip_ways:
#                 continue
#             for shot in range(n_shots):
#                 qry_fts_ = [[self.getFeatures(qry_fts, pred_mask[way + 1])]]
#                 fg_prototypes = self.getPrototype(qry_fts_)

#                 # Get predictions
#                 supp_pred = self.getPred(supp_fts[way, [shot]], fg_prototypes[way], self.thresh_pred[way])
#                 supp_pred = F.interpolate(supp_pred[None, ...], size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)

#                 pred_ups = torch.cat((1.0 - supp_pred, supp_pred), dim=1)

#                 # Construct the support Ground-Truth segmentation
#                 supp_label = torch.full_like(fore_mask[way, shot], 255, device=fore_mask.device)
#                 supp_label[fore_mask[way, shot] == 1] = 1
#                 supp_label[fore_mask[way, shot] == 0] = 0

#                 # Compute Loss
#                 eps = torch.finfo(torch.float32).eps
#                 log_prob = torch.log(torch.clamp(pred_ups, eps, 1 - eps))
#                 loss += self.criterion(log_prob, supp_label[None, ...].long()) / n_shots / n_ways

#         return loss, b_loss
#     def align_aux_Loss(self, supp_fts, qry_fts, fore_mask, pred, sup_fg_pts, sup_bg_pts, self_pred):
#         """
#         Args:
#             supp_fts: (way, shot, 512, 64, 64)
#             qry_fts: (N, 512, 64, 64)
#             fore_mask: (way, shot, 256, 256)
#             pred: (1, 2, 256, 256)
#             sup_fg_pts: way x [102 x 512]
#             sup_bg_pts: way x [602 x 512]
#             self_pred: (1, 2, 256, 256)
#         """
#         n_ways, n_shots = len(fore_mask), len(fore_mask[0])

#         # Get query mask
#         pred_mask = pred.argmax(dim=1, keepdim=True).squeeze(1)

#         binary_masks = [pred_mask == i for i in range(1 + n_ways)]
#         fg_pixels = binary_masks[1].sum().item()
#         is_skipped = int(fg_pixels == 0)
#         if not hasattr(self, "_dbg_total_epi"):
#             self._dbg_total_epi = 0
#             self._dbg_all_bg_epi = 0
#         self._dbg_total_epi += 1
#         self._dbg_all_bg_epi += is_skipped
#         if is_skipped:
#             print(
#                 f"[align_aux_Loss][ALL_BG] "
#                 f"episode={self._dbg_total_epi}, "
#                 f"fg_pixels={fg_pixels}, "
#                 f"all_bg_count={self._dbg_all_bg_epi}, "
#                 f"all_bg_rate={self._dbg_all_bg_epi / self._dbg_total_epi:.4f}",
#                 flush=True
#             )
#         skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
#         pred_mask = torch.stack(binary_masks, dim=0).float()

        

       

#         # Compute the support loss
#         loss = torch.zeros(1).to(self.device)
#         loss_aux = torch.zeros(1).to(self.device)
#         b_loss = torch.zeros(1).to(self.device)
#         ssp_loss = torch.zeros(1).to(self.device)
#         for way in range(n_ways):
#             use_support_fallback = bool((pred_mask[way + 1].sum() == 0).item())
#     # 只有全背景时才启用 fallback
#             if use_support_fallback:
#                 qry_seed_prob = self.getSelfPred(qry_fts, sup_fg_pts[way], sup_bg_pts[way])
#                 qry_seed_prob = F.interpolate(
#                     qry_seed_prob,
#                     size=fore_mask.shape[-2:],
#                     mode='bilinear',
#                     align_corners=True
#                 )

#                 qry_seed_2c = torch.cat((1.0 - qry_seed_prob, qry_seed_prob), dim=1)
#                 qry_seed_mask = torch.argmax(qry_seed_2c, dim=1, keepdim=True).squeeze(1)

#                 # 如果连 support-only 原型都分不出前景，那再跳过
#                 if qry_seed_mask.sum().item() == 0:
#                     continue

#                 cur_qry_mask = qry_seed_mask.float()

#                 print(
#                     f"[align_aux_Loss][SUPPORT_FALLBACK] "
#                     f"way={way}, fallback_fg_pixels={cur_qry_mask.sum().item()}",
#                     flush=True
#                 )
#             else:
#                 cur_qry_mask = pred_mask[way + 1]

#             # Get the query prototypes
#             for shot in range(n_shots):
#                 qry_fts_ = [[self.getFeatures(qry_fts, cur_qry_mask)]]
#                 fg_prototypes = self.getPrototype(qry_fts_)

#                 qry_bg_fts_ = [[self.getFeatures(qry_fts, 1.0 - cur_qry_mask)]]
#                 bg_prototypes = self.getPrototype(qry_bg_fts_)

#                 qry_pred = self.getSelfPred(qry_fts, fg_prototypes[0], bg_prototypes[0])
#                 qry_pred = F.interpolate(qry_pred, size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)
#                 qry_pred = torch.cat((1.0 - qry_pred, qry_pred), dim=1)
#                 qry_pred = torch.argmax(qry_pred, dim=1, keepdim=True).squeeze(1)

#                 fg_pts_ = [[self.get_fg_pts(qry_fts, cur_qry_mask, qry_pred)]]
#                 fg_pts_ = self.get_all_prototypes(fg_pts_)

#                 bg_pts_ = [[self.get_bg_pts(qry_fts, cur_qry_mask, qry_pred)]]
#                 bg_pts_ = self.get_all_prototypes(bg_pts_)

#                 loss_aux += self.get_aux_loss(sup_fg_pts[way], fg_pts_[way], sup_bg_pts[way], bg_pts_[way])

#                 # Get predictions
#                 supp_pred = self.get_fg_sim(supp_fts[way, [shot]], fg_pts_[way])
#                 bg_pred_ = self.get_bg_sim(supp_fts[way, [shot]], bg_pts_[way])
#                 supp_pred = F.interpolate(supp_pred, size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)
#                 bg_pred_ = F.interpolate(bg_pred_, size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)

#                 # Combine predictions
#                 preds = torch.cat([bg_pred_, supp_pred], dim=1)
#                 preds = torch.softmax(preds, dim=1)

#                 # Construct the support Ground-Truth segmentation
#                 supp_label = torch.full_like(fore_mask[way, shot], 255, device=fore_mask.device)
#                 supp_label[fore_mask[way, shot] == 1] = 1
#                 supp_label[fore_mask[way, shot] == 0] = 0

#                 # Compute Loss
#                 eps = torch.finfo(torch.float32).eps
#                 log_prob = torch.log(torch.clamp(preds, eps, 1 - eps))
#                 loss += self.criterion(log_prob, supp_label[None, ...].long()) / n_shots / n_ways

#                 ssp_log_prob = torch.log(torch.clamp(self_pred, eps, 1 - eps))
#                 ssp_loss += self.criterion(ssp_log_prob, supp_label[None, ...].long()) / n_shots / n_ways

#         return loss, loss_aux, b_loss, ssp_loss

#     def get_fg_pts(self, features, mask, pred_mask):
#         """
#         Args:
#         features: (1, 512, 64, 64)
#         mask: (1, 256, 256)
#         软弱原型  也就是平均原型
#         pred_mask: (1, 256, 256)
#         """
#         features_trans = F.interpolate(features, size=mask.shape[-2:], mode='bilinear', align_corners=True)
#         #边缘可以加入注意力机制
#         ie_mask = mask.squeeze(0) - torch.tensor(cv2.erode(mask.squeeze(0).cpu().numpy(), np.ones((3, 3), dtype=np.uint8), iterations=2)).to(self.device)
#         ie_mask = ie_mask.unsqueeze(0)
#         #内边缘原型  两层一层内边缘还带着一圈前景
#         ie_prototype = torch.sum(features_trans * ie_mask[None, ...], dim=(-2, -1)) \
#                        / (ie_mask[None, ...].sum(dim=(-2, -1)) + 1e-5)
#         #平均原型  也就是软弱原型
#         origin_prototype = torch.sum(features_trans * mask[None, ...], dim=(-2, -1)) \
#                            / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)

#         add_mask = (pred_mask.float() + mask).long()
#         mask1 = torch.zeros_like(mask)
#         mask2 = torch.zeros_like(mask)
#         mask1[add_mask == 2] = 1
#         mask2[add_mask == 1] = 1
#         mask1[mask == 0] = 0
#         mask2[mask == 0] = 0


#         #fg_fts #全是前景特征的图[1, 512, 256, 256]
#         fg_fts = self.get_fg_fts(features_trans, mask)
#         #可以修改更好的总结fg_prototypes = [100, 512]你可以把 mlp1 想象成一个**“会议记录员”**。

#         # 输入 (fg_fts)：是一场 65,536 人 参加的大会（全图像素）。
#         #
#         # 关键点：这 65,536 人全部都是“前景派”的人（因为我们之前填坑了）。
#         #
#         # 过程 (mlp1(...))：记录员听取了所有人的发言。
#         #
#         # 输出 (fg_prototypes)：记录员整理出了一份包含 100 条 关键信息的**“会议纪要”**。
#         #
#         # 这 100 条纪要（原型），就是对那 6 万多名前景像素特征的高度概括。
#         fg_prototypes = self.mlp1(fg_fts.view(512, 256 * 256)).permute(1, 0)

#         if torch.sum(mask2[mask2 == 1]) > 0:
#             #features_trans = features_trans[mask == 1]
#             #hard_fg  是在假阴性mask2地图中挑选的弱点
#             hard_fg = self.get_random_pts(features_trans, mask2, 50)
#             k = random.sample(range(len(fg_prototypes)), 50)
#             fg_prototypes = torch.cat([fg_prototypes[k], hard_fg], dim=0)
#         fg_pts = torch.cat([
#             fg_prototypes,  # local / hard prototypes
#             ie_prototype,  # edge prototype also interacts with query
#             origin_prototype  # global prototype MUST be last
#         ], dim=0)  # (102, 512)

#         #[102, 512]  fg_prototypes
#         return fg_pts

#     def get_bg_pts(self, features, mask, pred_mask):
#         """
#         Args:
#             features: (1, 512, 64, 64)
#             mask: (1, 256, 256)
#             pred_mask: (1, 256, 256)
#         """
#         bg_mask = 1 - mask
#         features_trans = F.interpolate(features, size=bg_mask.shape[-2:], mode='bilinear', align_corners=True)

#         oe_mask = torch.tensor(cv2.dilate(mask.squeeze(0).cpu().numpy(), np.ones((3, 3), dtype=np.uint8), iterations=2)).to(self.device) - mask.squeeze(0)
#         oe_mask = oe_mask.unsqueeze(0)
#         #外边缘原型
#         oe_prototype = torch.sum(features_trans * oe_mask[None, ...], dim=(-2, -1)) \
#                        / (oe_mask[None, ...].sum(dim=(-2, -1)) + 1e-5)
#         #背景平均原型
#         origin_prototype = torch.sum(features_trans * bg_mask[None, ...], dim=(-2, -1)) \
#                            / (bg_mask[None, ...].sum(dim=(-2, -1)) + 1e-5)

#         add_mask = (pred_mask.float() + mask).long()
#         mask1 = torch.zeros_like(mask)#高度置信背景
#         mask2 = torch.zeros_like(mask)#假阳性
#         mask1[add_mask == 0] = 1
#         mask2[add_mask == 1] = 1
#         mask1[bg_mask == 0] = 0
#         mask2[bg_mask == 0] = 0

#         bg_fts = self.get_fg_fts(features_trans, bg_mask)
#         bg_prototypes = self.mlp2(bg_fts.view(512, 256 * 256)).permute(1, 0)

#         if torch.sum(mask2[mask2 == 1]) > 0:
#             hard_bg = self.get_random_pts(features_trans, mask2, 100)
#             k = random.sample(range(len(bg_prototypes)), 500)
#             bg_prototypes = torch.cat([bg_prototypes[k], hard_bg], dim=0)
#         # global prototype MUST be last
#         bg_pts = torch.cat([
#             bg_prototypes,  # local / hard bg prototypes
#             oe_prototype,  # edge prototype
#             origin_prototype  # global bg prototype MUST be last
#         ], dim=0)  # (602, 512)

#         return bg_pts

#     def get_random_pts(self, features_trans, mask, n_prototype):
#         """
#         Args:
#             features_trans: (1, 512, 256, 256)
#             mask: (1, 256, 256)
#             n_prototype: int
#         """
#         features_trans = features_trans.squeeze(0)
#         features_trans = features_trans.permute(1, 2, 0)
#         features_trans = features_trans.view(features_trans.shape[-2] * features_trans.shape[-3],
#                                              features_trans.shape[-1])
#         mask = mask.squeeze(0).view(-1)
#         features_trans = features_trans[mask == 1]
#         if len(features_trans) >= n_prototype:
#             k = random.sample(range(len(features_trans)), n_prototype)
#             prototypes = features_trans[k]
#         else:
#             if len(features_trans) == 0:
#                 prototypes = torch.zeros(n_prototype, 512).to(self.device)
#             else:
#                 r = n_prototype // len(features_trans)
#                 k = random.sample(range(len(features_trans)), (n_prototype - len(features_trans)) % len(features_trans))
#                 prototypes = torch.cat([features_trans for _ in range(r)], dim=0)
#                 prototypes = torch.cat([features_trans[k], prototypes], dim=0)
#         ##select_pts[55536, 512]  全是前景特征的地图
#         return prototypes

#     def IFR_FD(self, pts, qry_fts, sup_fts, mask, qry_thresh, sup_thresh):
#         """
#         Args:
#             pts:        (K+1, C)
#                         pts[:-1] = local prototypes (hard + edge)
#                         pts[-1]  = global prototype
#             qry_fts:    (Nq, C, H, W)
#             sup_fts:    (1, C, H, W)
#             mask:       (1, H_img, W_img)
#             qry_thresh: scalar tensor
#             sup_thresh: scalar tensor

#         Returns:
#             matched_fts: (Nq, 2C+1, H, W)
#         """
#         n, c, h, w = qry_fts.shape

#         # split local/global prototypes
#         prototype = pts[-1].unsqueeze(0)  # (1, C), global prototype
#         pts = pts[:-1]  # (K, C), local prototypes

#         pts = F.normalize(pts, dim=-1)
#         prototype = F.normalize(prototype, dim=-1)

#         qry_scale = 1 + self.scaler * torch.sigmoid(qry_thresh)
#         sup_scale = 1 + self.scaler * torch.sigmoid(sup_thresh)

#         res = []
#         for i in range(n):
#             qry_ft = qry_fts[i].unsqueeze(0)  # (1, C, H, W)
#             fts_ = qry_ft.permute(0, 2, 3, 1)  # (1, H, W, C)
#             fts_ = F.normalize(fts_, dim=-1)

#             # ---------------- global similarity ----------------
#             one_sim = torch.matmul(fts_, prototype.transpose(0, 1)).permute(0, 3, 1, 2)  # (1,1,H,W)

#             # ---------------- IFR ----------------
#             sim = torch.matmul(fts_, pts.transpose(0, 1)).permute(3, 0, 1, 2)  # (K,1,H,W)
#             pseudo_mask = torch.sigmoid(sim * qry_scale)  # (K,1,H,W)

#             # query-guided prototype update
#             qry_ft_expand = qry_ft.expand(sim.shape[0], -1, -1, -1)  # (K,C,H,W)
#             qry_update = torch.sum(qry_ft_expand * pseudo_mask, dim=(-2, -1)) \
#                          / (pseudo_mask.sum(dim=(-2, -1)) + 1e-5)  # (K,C)

#             G = self.get_fuse_factor(fts_, sup_fts, mask)  # scalar

#             fuse_in = torch.stack([pts * (1 - G), qry_update * G], dim=1)  # (K,2,C)
#             fuse_in = fuse_in.permute(0, 1, 2).unsqueeze(-1)  # (K,2,C,1)
#             delta = self.fuse(fuse_in).squeeze(1).squeeze(-1)  # (K,C)

#             pts_refined = pts + delta  # (K,C # (K,C)

#             # ---------------- FD ----------------
#             pts_norm = F.normalize(pts_refined, dim=-1)
#             sim = torch.matmul(fts_, pts_norm.transpose(0, 1)).permute(3, 0, 1, 2)  # (K,1,H,W)

#             sim_map = torch.softmax(sim * sup_scale, dim=0)  # (K,1,H,W)
#             sim_0 = torch.sum(sim * sim_map, dim=0)  # (1,H,W)

#             matched_pts = pts_refined.unsqueeze(2).unsqueeze(3) * sim_map  # (K,C,H,W)
#             matched_pts = torch.sum(matched_pts, dim=0).unsqueeze(0)  # (1,C,H,W)

#             matched_fts = torch.cat([qry_ft, matched_pts, sim_0.unsqueeze(0)], dim=1)  # (1,2C+1,H,W)
#             res.append(matched_fts)

#         res = torch.cat(res, dim=0)  # (Nq, 2C+1, H, W)
#         return res

#     def get_fuse_factor(self, qry_fts, sup_fts, mask):
#         """
#         Args:
#             qry_fts: (1, H, W, 512)
#             sup_fts: (1, 512, Hs, Ws)
#             mask:    (1, H_img, W_img)
#         Returns:
#             G: scalar tensor
#         """
#         mask_ = F.interpolate(
#             mask.unsqueeze(0).float(),
#             size=sup_fts.shape[-2:],
#             mode='bilinear',
#             align_corners=True
#         ).squeeze(0)

#         if mask_.sum() == 0:
#             G = torch.tensor(0.0, device=self.device)
#         else:
#             sup_fts_ = F.normalize(
#                 sup_fts.permute(0, 2, 3, 1).reshape(-1, sup_fts.shape[1]), dim=-1
#             )  # (Hs*Ws, C)

#             qry_fts_ = qry_fts.reshape(-1, qry_fts.shape[-1])  # (H*W, C)
#             G = torch.matmul(sup_fts_, qry_fts_.transpose(0, 1))  # (Hs*Ws, H*W)
#             G = G[mask_.reshape(-1) > 0]
#             G = torch.sigmoid(torch.mean(G))

#         return G

#     def get_fg_fts(self, fts, mask):
#         """
#         Args:
#             fts: (1, 512, 256, 256)
#             mask: (1, 256, 256)
#             前景背景公用这个函数
#         """
#         _, c, h, w = fts.shape
#         # select masked fg features
#         fg_fts = fts * mask[None, ...]
#         bg_fts = torch.ones_like(fts) * mask[None, ...]
#         mask_ = mask.view(-1)
#         n_pts = len(mask_) - len(mask_[mask_ == 1])
#         #select_pts[55536, 512]  全是前景特征的地图  背景亦如此
#         select_pts = self.get_random_pts(fts, mask, n_pts)
#         index = bg_fts == 0
#         fg_fts[index] = select_pts.permute(1, 0).reshape(512*n_pts)
#         #全是前景特征的图[1, 512, 256, 256]
#         return fg_fts

#     def get_all_prototypes(self, fg_fts):
#         """
#         Args:
#             fg_fts: way x shot x [all x 512]
#         """
#         n_ways, n_shots = len(fg_fts), len(fg_fts[0])
#         prototypes = [sum([shot for shot in way]) / n_shots for way in fg_fts]

#         return prototypes

#     def get_fg_sim(self, fts, prototypes):
#         """
#         Args:
#             fts: (1, 512, 64, 64)
#             prototypes: (102, 512)
#         """
#         fts_ = fts.permute(0, 2, 3, 1)
#         fts_ = F.normalize(fts_, dim=-1)
#         pts_ = F.normalize(prototypes, dim=-1)
#         fg_sim = torch.matmul(fts_, pts_.transpose(0, 1)).permute(0, 3, 1, 2)
#         #得到 102 张“原始评分表”。

#         #Decoder (“阅卷老师”) 把这 102 张表融合成 1 张 最终的“肿瘤相似度图”。形状: [1, 1, 64, 64]。
#         fg_sim = self.sim_decoder_fg(fg_sim)

#         return fg_sim

#     def get_bg_sim(self, fts, prototypes):
#         """
#         Args:
#             fts: (1, 512, 64, 64)
#             prototypes: (602, 512)
#         """
#         fts_ = fts.permute(0, 2, 3, 1)
#         fts_ = F.normalize(fts_, dim=-1)
#         pts_ = F.normalize(prototypes, dim=-1)
#         bg_sim = torch.matmul(fts_, pts_.transpose(0, 1)).permute(0, 3, 1, 2)
#         bg_sim = self.sim_decoder_bg(bg_sim)

#         return bg_sim
#     def get_masked_fts(self, fts, mask):

#         masked_fts = torch.sum(fts * mask[None, ...], dim=(-2, -1)) \
#                      / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)  # 1 x C

#         return masked_fts

#     def getPrototype(self, fg_fts):
#         """
#         Average the features to obtain the prototype

#         Args:
#             fg_fts: lists of list of foreground features for each way/shot
#                 expect shape: Wa x Sh x [1 x C] (1, 1, (1, 512))
#             bg_fts: lists of list of background features for each way/shot
#                 expect shape: Wa x Sh x [1 x C]
#         """
#         n_ways, n_shots = len(fg_fts), len(fg_fts[0])
#         fg_prototypes = [torch.sum(torch.cat([tr for tr in way], dim=0), dim=0, keepdim=True) / n_shots for way in
#                          fg_fts]  ## concat all fg_fts   (n_way, (1, 512))

#         return fg_prototypes

#     def CI(self, fts, mask, kernel_size=(16, 16), stride=8, flag=1, aspect_ratio_range=(0.5, 2)):
#         # SI
#         b, c, height, width = fts.size()
#         kernel_h, kernel_w = kernel_size
#         n_fg = mask.sum()
#         n_bg = height * width - n_fg
#         fg_pts = []
#         bg_pts = []
#         fg_pts.append(torch.sum(fts * mask[None, ...], dim=(-2, -1)) / (n_fg + 1e-5))  # 1 x C
#         bg_pts.append(torch.sum(fts * (1 - mask[None, ...]), dim=(-2, -1)) / (n_bg + 1e-5))  # 1 x C
#         patch_fts = F.unfold(fts, kernel_size=kernel_size, stride=stride).permute(0, 2, 1).view(1, -1, c, kernel_h,
#                                                                                                 kernel_w)
#         patch_mask = F.unfold(mask.unsqueeze(1), kernel_size=kernel_size, stride=stride).permute(0, 2, 1).view(1, -1,
#                                                                                                                kernel_h,
#                                                                                                                kernel_w).unsqueeze(
#             2)
#         bg_patch_mask = 1 - patch_mask
#         random.seed(9)
#         if flag == 1:
#             fg_indx = torch.sum(patch_mask, dim=(-2, -1)).view(-1) >= min(kernel_h * kernel_w * self.rate, n_fg / 150)
#             fg_protos = torch.sum(patch_fts[:, fg_indx, :, :, :] * patch_mask[:, fg_indx, :, :, :], dim=(-2, -1)) \
#                         / (patch_mask[:, fg_indx, :, :, :].sum(dim=(-2, -1)) + 1e-5)  # (1, n, 512)
#             if fg_protos.numel() == 0:
#                 fg_protos = self.get_pts(fts, mask)
#             else:
#                 fg_protos = fg_protos.squeeze(0)
#                 if len(fg_protos) > 150:
#                     fg_protos = fg_protos[random.sample(range(len(fg_protos)), 150)]
#             fg_pts.append(fg_protos)

#         bg_indx = torch.sum(bg_patch_mask, dim=(-2, -1)).view(-1) >= min(kernel_h * kernel_w, n_bg / 250)
#         bg_protos = torch.sum(patch_fts[:, bg_indx, :, :, :] * bg_patch_mask[:, bg_indx, :, :, :], dim=(-2, -1)) \
#                     / (bg_patch_mask[:, bg_indx, :, :, :].sum(dim=(-2, -1)) + 1e-5)
#         if bg_protos.numel() == 0:
#             bg_protos = self.get_pts(fts, 1 - mask)
#         else:
#             bg_protos = bg_protos.squeeze(0)
#             if len(bg_protos) > 250:
#                 bg_protos = bg_protos[random.sample(range(len(bg_protos)), 250)]
#         bg_pts.append(bg_protos)

#         # RI
#         min_aspect_ratio, max_aspect_ratio = aspect_ratio_range
#         assert min_aspect_ratio > 0
#         assert max_aspect_ratio > min_aspect_ratio
#         torch.manual_seed(9)
#         aspect_ratio = torch.rand(self.num).to(self.device) * (max_aspect_ratio - min_aspect_ratio) + min_aspect_ratio
#         aspect_ratio0 = aspect_ratio[aspect_ratio <= 1]
#         bbox_height0 = (
#                     (torch.rand(len(aspect_ratio0)).to(self.device) * 0.9375 + 0.0625) * (height * aspect_ratio0)).int()
#         bbox_width0 = (bbox_height0 / aspect_ratio0).int()

#         aspect_ratio1 = aspect_ratio[aspect_ratio > 1]
#         bbox_width1 = (
#                     (torch.rand(len(aspect_ratio1)).to(self.device) * 0.9375 + 0.0625) * (width / aspect_ratio1)).int()
#         bbox_height1 = (bbox_width1 * aspect_ratio1).int()

#         bbox_height = torch.cat([bbox_height0, bbox_height1], dim=0)
#         bbox_width = torch.cat([bbox_width0, bbox_width1], dim=0)

#         x1 = (torch.rand(self.num).to(self.device) * (height - bbox_height)).int()
#         y1 = (torch.rand(self.num).to(self.device) * (width - bbox_width)).int()
#         x2 = x1 + bbox_height
#         y2 = y1 + bbox_width
#         for i in range(self.num):
#             n1 = mask[:, x1[i]:x2[i], y1[i]:y2[i]].sum()
#             n0 = (1 - mask[:, x1[i]:x2[i], y1[i]:y2[i]]).sum()
#             if n1 >= min(bbox_height[i] * bbox_width[i] * self.rate, n_fg / self.num) and n1 < n_fg:
#                 prototype = self.get_masked_fts(fts[:, :, x1[i]:x2[i], y1[i]:y2[i]], mask[:, x1[i]:x2[i], y1[i]:y2[i]])
#                 fg_pts.append(prototype)
#             if n0 >= min(bbox_height[i] * bbox_width[i], n_bg / self.num) and n0 < n_bg:
#                 prototype = self.get_masked_fts(fts[:, :, x1[i]:x2[i], y1[i]:y2[i]],
#                                                 1 - mask[:, x1[i]:x2[i], y1[i]:y2[i]])
#                 bg_pts.append(prototype)

#         fg_pts = torch.cat(fg_pts, dim=0)
#         bg_pts = torch.cat(bg_pts, dim=0)

#         return [bg_pts, fg_pts]
#     def get_pts(self, fts, mask):
#         features_trans = fts.squeeze(0)
#         features_trans = features_trans.permute(1, 2, 0)
#         features_trans = features_trans.view(features_trans.shape[-2] * features_trans.shape[-3],
#                                              features_trans.shape[-1])
#         mask = mask.squeeze(0).view(-1)
#         indx = mask == 1
#         features_trans = features_trans[indx]  # (n_fg x 512)
#         if len(features_trans) > 100:
#             random.seed(9)
#             features_trans = features_trans[random.sample(range(len(features_trans)), 100)]
#         return features_trans

#     def get_aux_loss(self, sup_fg_pts, qry_fg_pts, sup_bg_pts, qry_bg_pts):
#         """
#         Args:
#             sup_fg_pts: (102, 512)
#             qry_fg_pts: (102, 512)
#             sup_bg_pts: (602, 512)
#             qry_bg_pts: (602, 512)
#         """
#         d1 = torch.mean(sup_fg_pts, dim=0, keepdim=True)
#         d2 = torch.mean(qry_fg_pts, dim=0, keepdim=True)
#         b1 = torch.mean(sup_bg_pts, dim=0, keepdim=True)
#         b2 = torch.mean(qry_bg_pts, dim=0, keepdim=True)

#         d1 = F.normalize(d1, dim=-1)
#         d2 = F.normalize(d2, dim=-1)
#         b1 = F.normalize(b1, dim=-1)
#         b2 = F.normalize(b2, dim=-1)

#         fg_intra = torch.matmul(d1, d2.transpose(0, 1)).squeeze(0).squeeze(0)
#         bg_intra = torch.matmul(b1, b2.transpose(0, 1)).squeeze(0).squeeze(0)
#         intra_loss = 2 - fg_intra - bg_intra

#         zero = torch.zeros(1).to(self.device).squeeze(0)
#         sup_inter = torch.matmul(d1, b1.transpose(0, 1))
#         qry_inter = torch.matmul(d2, b2.transpose(0, 1))
#         inter_loss = torch.max(zero, torch.mean(sup_inter)) + torch.max(zero, torch.mean(qry_inter))

#         return intra_loss + inter_loss
# """
# FSS via CoW
# Extended from ADNet code by Hansen et al.
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from models.encoder import Res101Encoder
# from models.modules import MLP, Decoder, Supp_Decoder
# import numpy as np
# import random
# import cv2
# from boundary_loss import BoundaryLoss
# from models.Decoders import Decoder as IFRDecoder


# class FewShotSeg(nn.Module):
#     def __init__(self, pretrained_weights="deeplabv3"):
#         super().__init__()
#         # Encoder
#         self.encoder = Res101Encoder(replace_stride_with_dilation=[True, True, False],
#                                      pretrained_weights=pretrained_weights)
#         self.device = torch.device('cuda')
#         self.scaler = 20.0
#         self.rate = 0.95
#         self.kernel = (16, 8)
#         self.stride = 8
#         self.num = 100
#         #它与 LogSoftmax（或在这个代码中的 torch.log(torch.softmax(...))）配合使用，用来衡量模型预测的像素类别与真实标签之间的差距。
#         self.criterion = nn.NLLLoss()
#         #专门用来惩罚模型在物体边缘（边界）处预测不准确的情况，能让分割结果的边缘更清晰、更贴合真实物体。
#         self.criterion_b = BoundaryLoss(theta0=3, theta=5)
#         #ean Squared Error Loss (均方误差损失)。通常用于回归任务（预测一个连续的数值）。在这个项目中，它很可能是用在 get_aux_loss（辅助损失）中，用来拉近支持集原型和查询集原型（它们都是特征向量）之间的距离。
#         self.criterion_MSE = nn.MSELoss()
#         #定义了要从“弱点”区域挖掘 100 个困难前景 (Hard Foreground) 原型。
#         self.fg_num = 100
#         #定义了要挖掘 600 个困难背景 (Hard Background) 原型。
#         self.bg_num = 600
#         self.ifr_decoder_fg = IFRDecoder(in_ch=1025)
#         self.ifr_decoder_bg = IFRDecoder(in_ch=1025)
#         self.mlp1 = MLP(256, self.fg_num)
#         self.mlp2 = MLP(256, self.bg_num)
#         self.fuse = nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0, bias=False)
#         #它的输入是查询集特征与 100 个前景原型计算出的“相似度图”，输出是最终的前景分割图。它负责**“融合”**这 100 个原型的意见。
#         self.sim_decoder_fg = Decoder(self.fg_num)
#         #同上，负责**“融合”** 600 个背景原型的意见，生成背景分割图。
#         self.sim_decoder_bg = Decoder(self.bg_num)
#         #它用于支持集自我预测 (SSP) 模块。在 getSelfPred 方法中，它接收支持集特征 F_s 和朴素原型 p 的拼接结果，并输出一个初始的、粗糙的自我预测掩码 M_sp
#         self.supp_decoder = Supp_Decoder()

#     def forward(self, supp_imgs, supp_mask, qry_imgs, train=False):
#         """
#         Args:
#             supp_imgs: support images
#                 way x shot x [B x 3 x H x W]
#             supp_mask: foreground masks for support images
#                 way x shot x [B x H x W]
#             qry_imgs: query images
#                 N x [B x 3 x H x W]
#             train: whether to train model or not
#         """
#         self.n_ways = len(supp_imgs)
#         self.n_shots = len(supp_imgs[0])
#         self.n_queries = len(qry_imgs)
#         assert self.n_ways == 1  # for now only one-way, because not every shot has multiple sub-images
#         assert self.n_queries == 1

#         qry_bs = qry_imgs[0].shape[0]
#         supp_bs = supp_imgs[0][0].shape[0]
#         img_size = supp_imgs[0][0].shape[-2:]
#         supp_mask = torch.stack([torch.stack(way, dim=0) for way in supp_mask],
#                                 dim=0).view(supp_bs, self.n_ways, self.n_shots, *img_size)

#         ###### Extract features ######
#         imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
#                                 + [torch.cat(qry_imgs, dim=0), ], dim=0)
#         img_fts, tao = self.encoder(imgs_concat)

#         supp_fts = [img_fts[dic][:self.n_ways * self.n_shots * supp_bs].view(
#             supp_bs, self.n_ways, self.n_shots, -1, *img_fts[dic].shape[-2:]) for _, dic in enumerate(img_fts)]
#         supp_fts = supp_fts[0]

#         qry_fts = [img_fts[dic][self.n_ways * self.n_shots * supp_bs:].view(
#             qry_bs, self.n_queries, -1, *img_fts[dic].shape[-2:]) for _, dic in enumerate(img_fts)]
#         qry_fts = qry_fts[0]

#         ###### Get threshold ######
#         self.t = tao[self.n_ways * self.n_shots * supp_bs:]
#         self.thresh_pred = [self.t for _ in range(self.n_ways)]
#         supp_tao = tao[:self.n_ways * self.n_shots * supp_bs].view(supp_bs, self.n_ways, self.n_shots, -1)
#         qry_tao = tao[self.n_ways * self.n_shots * supp_bs:].view(qry_bs, self.n_queries, -1)

#         ###### Compute loss ######
#         align_loss = torch.zeros(1).to(self.device)
#         b_loss = torch.zeros(1).to(self.device)
#         ssp_loss = torch.zeros(1).to(self.device)
#         aux_loss = torch.zeros(1).to(self.device)
#         outputs = []
#         for epi in range(supp_bs):
#             ###### Extract prototypes ######
#             # 1. 上采样 support feature 到 mask 尺寸
#             supp_fts_ = [[F.interpolate(supp_fts[[epi], way, shot], size=img_size,
#                                         mode='bilinear', align_corners=True)
#                           for shot in range(self.n_shots)] for way in range(self.n_ways)]

#             # 2. 提取全局前景原型 / 全局背景原型
#             prototype = [[self.get_masked_fts(supp_fts_[way][shot], supp_mask[[epi], way, shot])
#                           for shot in range(self.n_shots)] for way in range(self.n_ways)]
#             prototype = self.getPrototype(prototype)

#             bg_prototype = [[self.get_masked_fts(supp_fts_[way][shot], 1 - supp_mask[[epi], way, shot])
#                              for shot in range(self.n_shots)] for way in range(self.n_ways)]
#             bg_prototype = self.getPrototype(bg_prototype)

#             # 3. 用 CI 提取区域原型
#             if supp_mask[epi][0].sum() == 0:
#                 region_protos = [[self.CI(supp_fts_[way][shot], supp_mask[[epi], way, shot],
#                                           self.kernel, self.stride, flag=0)
#                                   for shot in range(self.n_shots)] for way in range(self.n_ways)]
#             else:
#                 region_protos = [[self.CI(supp_fts_[way][shot], supp_mask[[epi], way, shot],
#                                           self.kernel, self.stride, flag=1)
#                                   for shot in range(self.n_shots)] for way in range(self.n_ways)]

#             bg_pts = [torch.cat([region_protos[way][0][0], bg_prototype[way]], dim=0) for way in range(self.n_ways)]
#             fg_pts = [torch.cat([region_protos[way][0][1], prototype[way]], dim=0) for way in range(self.n_ways)]



#             # support self-prediction using prototype groups from CI
#             supp_preds = [[self.getSelfPred(
#                 supp_fts_[way][shot],  # [1,C,H,W]
#                 fg_pts[way],  # fg_pts
#                 bg_pts[way]  # bg_pts
#             ) for shot in range(self.n_shots)] for way in range(self.n_ways)]

#             # current 1-way 1-shot case
#             supp_pred = supp_preds[0][0]
#             supp_pred = F.interpolate(supp_pred, size=img_size, mode='bilinear', align_corners=True)
#             supp_pred = torch.cat((1.0 - supp_pred, supp_pred), dim=1)
#             self_pred = supp_pred
#             supp_pred = torch.argmax(supp_pred, dim=1, keepdim=True).squeeze(1)
#             #supp_pred[1, 512, 256, 256]

#             fg_pts = [[self.get_fg_pts(supp_fts[[epi], way, shot], supp_mask[[epi], way, shot], supp_pred)
#                        for shot in range(self.n_shots)] for way in range(self.n_ways)]
#             # fg_pts = [
#             #              < 张量
#             # shape = [102, 512] >,  # Way 0 的最终平均团队
#             # < 张量
#             # shape = [102, 512] >  # Way 1 的最终平均团队
#             # ]   变得更平均了  少了shot的一个嵌套
#             fg_pts = self.get_all_prototypes(fg_pts)
#             bg_pts = [[self.get_bg_pts(supp_fts[[epi], way, shot], supp_mask[[epi], way, shot], supp_pred)
#                        for shot in range(self.n_shots)] for way in range(self.n_ways)]
#             bg_pts = self.get_all_prototypes(bg_pts)

#             # ----- threshold split -----
#             # suppose tao shape is [num_images, 1] or compatible scalar tensor

#             # =========================
#             # Query prediction via IFR_FD
#             # =========================
#             fg_matched_fts = torch.stack([
#                 self.IFR_FD(
#                     fg_pts[way],  # (102, 512), last one is global fg prototype
#                     qry_fts[epi],  # (1, 512, H', W')
#                     supp_fts[[epi], way, 0],  # (1, 512, H', W')
#                     supp_mask[[epi], way, 0],  # (1, H, W)
#                     qry_tao[epi, 0],  # query threshold
#                     supp_tao[epi, way, 0]  # support threshold
#                 )
#                 for way in range(self.n_ways)
#             ], dim=1)  # (Nq, Wa, 1025, H', W')

#             bg_matched_fts = torch.stack([
#                 self.IFR_FD(
#                     bg_pts[way],  # (602, 512), last one is global bg prototype
#                     qry_fts[epi],  # (1, 512, H', W')
#                     supp_fts[[epi], way, 0],  # (1, 512, H', W')
#                     1 - supp_mask[[epi], way, 0],  # background mask
#                     qry_tao[epi, 0],
#                     supp_tao[epi, way, 0]
#                 )
#                 for way in range(self.n_ways)
#             ], dim=1)  # (Nq, Wa, 1025, H', W')

#             fg_pred = torch.cat([
#                 self.ifr_decoder_fg(fg_matched_fts[:, way]) for way in range(self.n_ways)
#             ], dim=1)  # (Nq, Wa, H', W') or (Nq, Wa, 1, H', W') depending on decoder

#             bg_pred = torch.cat([
#                 self.ifr_decoder_bg(bg_matched_fts[:, way]) for way in range(self.n_ways)
#             ], dim=1)

#             fg_pred = F.interpolate(fg_pred, size=img_size, mode='bilinear', align_corners=True)
#             bg_pred = F.interpolate(bg_pred, size=img_size, mode='bilinear', align_corners=True)

#             preds = torch.cat([bg_pred, fg_pred], dim=1)
#             preds = torch.softmax(preds, dim=1)

#             outputs.append(preds)
#             if train:
#                 align_loss_epi, aux_loss_epi, b_loss_epi, ssp_loss_epi = self.align_aux_Loss(supp_fts[epi], qry_fts[epi], supp_mask[epi], preds,
#                                                                                              fg_pts, bg_pts, self_pred)
#                 align_loss += align_loss_epi
#                 aux_loss += aux_loss_epi
#                 b_loss += b_loss_epi
#                 ssp_loss += ssp_loss_epi
#             #您的写法 (推荐)：
#             # Python
#             #
#             # output = torch.cat(outputs, dim=0)评价：脱裤子放屁——多此一举。
#         output = torch.stack(outputs, dim=1)
#         output = output.view(-1, *output.shape[2:])

#         return output, align_loss / supp_bs, aux_loss / supp_bs, b_loss / supp_bs, ssp_loss / supp_bs

#     def getPred(self, fts, prototype, thresh):
#         """
#         Args:
#             fts: (1, 512, 64, 64)
#             prototype: (1, 512)
#             thresh: (1, 1)
#         """
#         sim = -F.cosine_similarity(fts, prototype[..., None, None], dim=1) * self.scaler
#         pred = 1.0 - torch.sigmoid(0.5 * (sim - thresh))

#         return pred

#     def getSelfPred(self, supp_fts, fg_pts, bg_pts):
#         """
#         Args:
#             supp_fts: (1, 512, H, W)
#             fg_pts:   (N_fg, 512)
#             bg_pts:   (N_bg, 512)
#         Returns:
#             supp_pred: (1, 1, H, W)
#         """
#         B, C, H, W = supp_fts.shape

#         # (1, H, W, C)
#         fts_hw = supp_fts.permute(0, 2, 3, 1)
#         fts_norm = F.normalize(fts_hw, dim=-1)

#         # ---- foreground context ----
#         fg_norm = F.normalize(fg_pts, dim=-1)  # (N_fg, C)
#         fg_sim = torch.matmul(fts_norm, fg_norm.t())  # (1, H, W, N_fg)
#         fg_attn = torch.softmax(fg_sim * self.scaler, dim=-1)
#         ctx_fg = torch.matmul(fg_attn, fg_pts).permute(0, 3, 1, 2)  # (1, C, H, W)

#         # ---- background context ----
#         bg_norm = F.normalize(bg_pts, dim=-1)  # (N_bg, C)
#         bg_sim = torch.matmul(fts_norm, bg_norm.t())  # (1, H, W, N_bg)
#         bg_attn = torch.softmax(bg_sim * self.scaler, dim=-1)
#         ctx_bg = torch.matmul(bg_attn, bg_pts).permute(0, 3, 1, 2)  # (1, C, H, W)

#         # ---- original feature + fg context + bg context ----
#         supp_input = torch.cat([supp_fts, ctx_fg, ctx_bg], dim=1)  # (1, 1536, H, W)
#         supp_pred = self.supp_decoder(supp_input)

#         return supp_pred
#     #的作用是计算支持集图像前景区域的【平均特征】
#     def getFeatures(self, fts, mask):
#         """
#         Arg0.s:
#              fts: (1, 512, 64, 64)
#             mask: (1, 256, 256)
#         """
#         fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear', align_corners=True)
#         masked_fts = torch.sum(fts * mask[None, ...], dim=(-2, -1)) \
#                      / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)

#         return masked_fts

#     def getPrototype(self, fg_fts):
#         """
#         Args:
#             fg_fts: way x shot x [1 x 512]
#         """
#         n_ways, n_shots = len(fg_fts), len(fg_fts[0])
#         fg_prototypes = [torch.sum(torch.cat([tr for tr in way], dim=0), dim=0, keepdim=True) / n_shots
#                          for way in fg_fts]

#         return fg_prototypes

#     def alignLoss(self, supp_fts, qry_fts, pred, fore_mask):
#         """
#         Args:
#             supp_fts: (way, shot, 512, 64, 64)
#             qry_fts: (N, 512, 64, 64)
#             pred: (1, 2, H, W)
#             fore_mask: (way, shot, 256, 256)
#         """
#         n_ways, n_shots = len(fore_mask), len(fore_mask[0])

#         # Get query mask
#         pred_mask = pred.argmax(dim=1, keepdim=True).squeeze(1)
#         binary_masks = [pred_mask == i for i in range(1 + n_ways)]
#         fg_pixels = binary_masks[1].sum().item()
#         is_skipped = int(binary_masks[1].sum() == 0)
#         skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
#         pred_mask = torch.stack(binary_masks, dim=0).float()

#         # Compute the support loss
#         loss = torch.zeros(1).to(self.device)
#         b_loss = torch.zeros(1).to(self.device)
#         for way in range(n_ways):
#             if way in skip_ways:
#                 continue
#             for shot in range(n_shots):
#                 qry_fts_ = [[self.getFeatures(qry_fts, pred_mask[way + 1])]]
#                 fg_prototypes = self.getPrototype(qry_fts_)

#                 # Get predictions
#                 supp_pred = self.getPred(supp_fts[way, [shot]], fg_prototypes[way], self.thresh_pred[way])
#                 supp_pred = F.interpolate(supp_pred[None, ...], size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)

#                 pred_ups = torch.cat((1.0 - supp_pred, supp_pred), dim=1)

#                 # Construct the support Ground-Truth segmentation
#                 supp_label = torch.full_like(fore_mask[way, shot], 255, device=fore_mask.device)
#                 supp_label[fore_mask[way, shot] == 1] = 1
#                 supp_label[fore_mask[way, shot] == 0] = 0

#                 # Compute Loss
#                 eps = torch.finfo(torch.float32).eps
#                 log_prob = torch.log(torch.clamp(pred_ups, eps, 1 - eps))
#                 loss += self.criterion(log_prob, supp_label[None, ...].long()) / n_shots / n_ways

#         return loss, b_loss
#     def align_aux_Loss(self, supp_fts, qry_fts, fore_mask, pred, sup_fg_pts, sup_bg_pts, self_pred):
#         """
#         Args:
#             supp_fts: (way, shot, 512, 64, 64)
#             qry_fts: (N, 512, 64, 64)
#             fore_mask: (way, shot, 256, 256)
#             pred: (1, 2, 256, 256)
#             sup_fg_pts: way x [102 x 512]
#             sup_bg_pts: way x [602 x 512]
#             self_pred: (1, 2, 256, 256)
#         """
#         n_ways, n_shots = len(fore_mask), len(fore_mask[0])

#         # Get query mask
#         pred_mask = pred.argmax(dim=1, keepdim=True).squeeze(1)

#         binary_masks = [pred_mask == i for i in range(1 + n_ways)]
#         fg_pixels = binary_masks[1].sum().item()
#         is_skipped = int(fg_pixels == 0)
#         if not hasattr(self, "_dbg_total_epi"):
#             self._dbg_total_epi = 0
#             self._dbg_all_bg_epi = 0
#         self._dbg_total_epi += 1
#         self._dbg_all_bg_epi += is_skipped
#         if is_skipped:
#             print(
#                 f"[align_aux_Loss][ALL_BG] "
#                 f"episode={self._dbg_total_epi}, "
#                 f"fg_pixels={fg_pixels}, "
#                 f"all_bg_count={self._dbg_all_bg_epi}, "
#                 f"all_bg_rate={self._dbg_all_bg_epi / self._dbg_total_epi:.4f}",
#                 flush=True
#             )
#         skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
#         pred_mask = torch.stack(binary_masks, dim=0).float()

        

       

#         # Compute the support loss
#         loss = torch.zeros(1).to(self.device)
#         loss_aux = torch.zeros(1).to(self.device)
#         b_loss = torch.zeros(1).to(self.device)
#         ssp_loss = torch.zeros(1).to(self.device)
#         for way in range(n_ways):
#             use_support_fallback = bool((pred_mask[way + 1].sum() == 0).item())
#     # 只有全背景时才启用 fallback
#             if use_support_fallback:
#                 qry_seed_prob = self.getSelfPred(qry_fts, sup_fg_pts[way], sup_bg_pts[way])
#                 qry_seed_prob = F.interpolate(
#                     qry_seed_prob,
#                     size=fore_mask.shape[-2:],
#                     mode='bilinear',
#                     align_corners=True
#                 )

#                 qry_seed_2c = torch.cat((1.0 - qry_seed_prob, qry_seed_prob), dim=1)
#                 qry_seed_mask = torch.argmax(qry_seed_2c, dim=1, keepdim=True).squeeze(1)

#                 # 如果连 support-only 原型都分不出前景，那再跳过
#                 if qry_seed_mask.sum().item() == 0:
#                     continue

#                 cur_qry_mask = qry_seed_mask.float()

#                 print(
#                     f"[align_aux_Loss][SUPPORT_FALLBACK] "
#                     f"way={way}, fallback_fg_pixels={cur_qry_mask.sum().item()}",
#                     flush=True
#                 )
#             else:
#                 cur_qry_mask = pred_mask[way + 1]

#             # Get the query prototypes
#             for shot in range(n_shots):
#                 qry_fts_ = [[self.getFeatures(qry_fts, cur_qry_mask)]]
#                 fg_prototypes = self.getPrototype(qry_fts_)

#                 qry_bg_fts_ = [[self.getFeatures(qry_fts, 1.0 - cur_qry_mask)]]
#                 bg_prototypes = self.getPrototype(qry_bg_fts_)

#                 qry_pred = self.getSelfPred(qry_fts, fg_prototypes[0], bg_prototypes[0])
#                 qry_pred = F.interpolate(qry_pred, size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)
#                 qry_pred = torch.cat((1.0 - qry_pred, qry_pred), dim=1)
#                 qry_pred = torch.argmax(qry_pred, dim=1, keepdim=True).squeeze(1)

#                 fg_pts_ = [[self.get_fg_pts(qry_fts, cur_qry_mask, qry_pred)]]
#                 fg_pts_ = self.get_all_prototypes(fg_pts_)

#                 bg_pts_ = [[self.get_bg_pts(qry_fts, cur_qry_mask, qry_pred)]]
#                 bg_pts_ = self.get_all_prototypes(bg_pts_)

#                 loss_aux += self.get_aux_loss(sup_fg_pts[way], fg_pts_[way], sup_bg_pts[way], bg_pts_[way])

#                 # Get predictions
#                 supp_pred = self.get_fg_sim(supp_fts[way, [shot]], fg_pts_[way])
#                 bg_pred_ = self.get_bg_sim(supp_fts[way, [shot]], bg_pts_[way])
#                 supp_pred = F.interpolate(supp_pred, size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)
#                 bg_pred_ = F.interpolate(bg_pred_, size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)

#                 # Combine predictions
#                 preds = torch.cat([bg_pred_, supp_pred], dim=1)
#                 preds = torch.softmax(preds, dim=1)

#                 # Construct the support Ground-Truth segmentation
#                 supp_label = torch.full_like(fore_mask[way, shot], 255, device=fore_mask.device)
#                 supp_label[fore_mask[way, shot] == 1] = 1
#                 supp_label[fore_mask[way, shot] == 0] = 0

#                 # Compute Loss
#                 eps = torch.finfo(torch.float32).eps
#                 log_prob = torch.log(torch.clamp(preds, eps, 1 - eps))
#                 loss += self.criterion(log_prob, supp_label[None, ...].long()) / n_shots / n_ways

#                 ssp_log_prob = torch.log(torch.clamp(self_pred, eps, 1 - eps))
#                 ssp_loss += self.criterion(ssp_log_prob, supp_label[None, ...].long()) / n_shots / n_ways

#         return loss, loss_aux, b_loss, ssp_loss

#     def get_fg_pts(self, features, mask, pred_mask):
#         """
#         Args:
#         features: (1, 512, 64, 64)
#         mask: (1, 256, 256)
#         软弱原型  也就是平均原型
#         pred_mask: (1, 256, 256)
#         """
#         features_trans = F.interpolate(features, size=mask.shape[-2:], mode='bilinear', align_corners=True)
#         #边缘可以加入注意力机制
#         ie_mask = mask.squeeze(0) - torch.tensor(cv2.erode(mask.squeeze(0).cpu().numpy(), np.ones((3, 3), dtype=np.uint8), iterations=2)).to(self.device)
#         ie_mask = ie_mask.unsqueeze(0)
#         #内边缘原型  两层一层内边缘还带着一圈前景
#         ie_prototype = torch.sum(features_trans * ie_mask[None, ...], dim=(-2, -1)) \
#                        / (ie_mask[None, ...].sum(dim=(-2, -1)) + 1e-5)
#         #平均原型  也就是软弱原型
#         origin_prototype = torch.sum(features_trans * mask[None, ...], dim=(-2, -1)) \
#                            / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)

#         add_mask = (pred_mask.float() + mask).long()
#         mask1 = torch.zeros_like(mask)
#         mask2 = torch.zeros_like(mask)
#         mask1[add_mask == 2] = 1
#         mask2[add_mask == 1] = 1
#         mask1[mask == 0] = 0
#         mask2[mask == 0] = 0


#         #fg_fts #全是前景特征的图[1, 512, 256, 256]
#         fg_fts = self.get_fg_fts(features_trans, mask)
#         #可以修改更好的总结fg_prototypes = [100, 512]你可以把 mlp1 想象成一个**“会议记录员”**。

#         # 输入 (fg_fts)：是一场 65,536 人 参加的大会（全图像素）。
#         #
#         # 关键点：这 65,536 人全部都是“前景派”的人（因为我们之前填坑了）。
#         #
#         # 过程 (mlp1(...))：记录员听取了所有人的发言。
#         #
#         # 输出 (fg_prototypes)：记录员整理出了一份包含 100 条 关键信息的**“会议纪要”**。
#         #
#         # 这 100 条纪要（原型），就是对那 6 万多名前景像素特征的高度概括。
#         fg_prototypes = self.mlp1(fg_fts.view(512, 256 * 256)).permute(1, 0)

#         if torch.sum(mask2[mask2 == 1]) > 0:
#             #features_trans = features_trans[mask == 1]
#             #hard_fg  是在假阴性mask2地图中挑选的弱点
#             hard_fg = self.get_random_pts(features_trans, mask2, 50)
#             k = random.sample(range(len(fg_prototypes)), 50)
#             fg_prototypes = torch.cat([fg_prototypes[k], hard_fg], dim=0)
#         fg_pts = torch.cat([
#             fg_prototypes,  # local / hard prototypes
#             ie_prototype,  # edge prototype also interacts with query
#             origin_prototype  # global prototype MUST be last
#         ], dim=0)  # (102, 512)

#         #[102, 512]  fg_prototypes
#         return fg_pts

#     def get_bg_pts(self, features, mask, pred_mask):
#         """
#         Args:
#             features: (1, 512, 64, 64)
#             mask: (1, 256, 256)
#             pred_mask: (1, 256, 256)
#         """
#         bg_mask = 1 - mask
#         features_trans = F.interpolate(features, size=bg_mask.shape[-2:], mode='bilinear', align_corners=True)

#         oe_mask = torch.tensor(cv2.dilate(mask.squeeze(0).cpu().numpy(), np.ones((3, 3), dtype=np.uint8), iterations=2)).to(self.device) - mask.squeeze(0)
#         oe_mask = oe_mask.unsqueeze(0)
#         #外边缘原型
#         oe_prototype = torch.sum(features_trans * oe_mask[None, ...], dim=(-2, -1)) \
#                        / (oe_mask[None, ...].sum(dim=(-2, -1)) + 1e-5)
#         #背景平均原型
#         origin_prototype = torch.sum(features_trans * bg_mask[None, ...], dim=(-2, -1)) \
#                            / (bg_mask[None, ...].sum(dim=(-2, -1)) + 1e-5)

#         add_mask = (pred_mask.float() + mask).long()
#         mask1 = torch.zeros_like(mask)#高度置信背景
#         mask2 = torch.zeros_like(mask)#假阳性
#         mask1[add_mask == 0] = 1
#         mask2[add_mask == 1] = 1
#         mask1[bg_mask == 0] = 0
#         mask2[bg_mask == 0] = 0

#         bg_fts = self.get_fg_fts(features_trans, bg_mask)
#         bg_prototypes = self.mlp2(bg_fts.view(512, 256 * 256)).permute(1, 0)

#         if torch.sum(mask2[mask2 == 1]) > 0:
#             hard_bg = self.get_random_pts(features_trans, mask2, 100)
#             k = random.sample(range(len(bg_prototypes)), 500)
#             bg_prototypes = torch.cat([bg_prototypes[k], hard_bg], dim=0)
#         # global prototype MUST be last
#         bg_pts = torch.cat([
#             bg_prototypes,  # local / hard bg prototypes
#             oe_prototype,  # edge prototype
#             origin_prototype  # global bg prototype MUST be last
#         ], dim=0)  # (602, 512)

#         return bg_pts

#     def get_random_pts(self, features_trans, mask, n_prototype):
#         """
#         Args:
#             features_trans: (1, 512, 256, 256)
#             mask: (1, 256, 256)
#             n_prototype: int
#         """
#         features_trans = features_trans.squeeze(0)
#         features_trans = features_trans.permute(1, 2, 0)
#         features_trans = features_trans.view(features_trans.shape[-2] * features_trans.shape[-3],
#                                              features_trans.shape[-1])
#         mask = mask.squeeze(0).view(-1)
#         features_trans = features_trans[mask == 1]
#         if len(features_trans) >= n_prototype:
#             k = random.sample(range(len(features_trans)), n_prototype)
#             prototypes = features_trans[k]
#         else:
#             if len(features_trans) == 0:
#                 prototypes = torch.zeros(n_prototype, 512).to(self.device)
#             else:
#                 r = n_prototype // len(features_trans)
#                 k = random.sample(range(len(features_trans)), (n_prototype - len(features_trans)) % len(features_trans))
#                 prototypes = torch.cat([features_trans for _ in range(r)], dim=0)
#                 prototypes = torch.cat([features_trans[k], prototypes], dim=0)
#         ##select_pts[55536, 512]  全是前景特征的地图
#         return prototypes

#     def IFR_FD(self, pts, qry_fts, sup_fts, mask, qry_thresh, sup_thresh):
#         """
#         Args:
#             pts:        (K+1, C)
#                         pts[:-1] = local prototypes (hard + edge)
#                         pts[-1]  = global prototype
#             qry_fts:    (Nq, C, H, W)
#             sup_fts:    (1, C, H, W)
#             mask:       (1, H_img, W_img)
#             qry_thresh: scalar tensor
#             sup_thresh: scalar tensor

#         Returns:
#             matched_fts: (Nq, 2C+1, H, W)
#         """
#         n, c, h, w = qry_fts.shape

#         # split local/global prototypes
#         prototype = pts[-1].unsqueeze(0)  # (1, C), global prototype
#         pts = pts[:-1]  # (K, C), local prototypes

#         pts = F.normalize(pts, dim=-1)
#         prototype = F.normalize(prototype, dim=-1)

#         qry_scale = 1 + self.scaler * torch.sigmoid(qry_thresh)
#         sup_scale = 1 + self.scaler * torch.sigmoid(sup_thresh)

#         res = []
#         for i in range(n):
#             qry_ft = qry_fts[i].unsqueeze(0)  # (1, C, H, W)
#             fts_ = qry_ft.permute(0, 2, 3, 1)  # (1, H, W, C)
#             fts_ = F.normalize(fts_, dim=-1)

#             # ---------------- global similarity ----------------
#             one_sim = torch.matmul(fts_, prototype.transpose(0, 1)).permute(0, 3, 1, 2)  # (1,1,H,W)

#             # ---------------- IFR ----------------
#             sim = torch.matmul(fts_, pts.transpose(0, 1)).permute(3, 0, 1, 2)  # (K,1,H,W)
#             pseudo_mask = torch.sigmoid(sim * qry_scale)  # (K,1,H,W)

#             # query-guided prototype update
#             qry_ft_expand = qry_ft.expand(sim.shape[0], -1, -1, -1)  # (K,C,H,W)
#             qry_update = torch.sum(qry_ft_expand * pseudo_mask, dim=(-2, -1)) \
#                          / (pseudo_mask.sum(dim=(-2, -1)) + 1e-5)  # (K,C)

#             G = self.get_fuse_factor(fts_, sup_fts, mask)  # scalar

#             fuse_in = torch.stack([pts * (1 - G), qry_update * G], dim=1)  # (K,2,C)
#             fuse_in = fuse_in.permute(0, 1, 2).unsqueeze(-1)  # (K,2,C,1)
#             delta = self.fuse(fuse_in).squeeze(1).squeeze(-1)  # (K,C)

#             pts_refined = pts + delta  # (K,C # (K,C)

#             # ---------------- FD ----------------
#             pts_norm = F.normalize(pts_refined, dim=-1)
#             sim = torch.matmul(fts_, pts_norm.transpose(0, 1)).permute(3, 0, 1, 2)  # (K,1,H,W)

#             sim_map = torch.softmax(sim * sup_scale, dim=0)  # (K,1,H,W)
#             sim_0 = torch.sum(sim * sim_map, dim=0)  # (1,H,W)

#             matched_pts = pts_refined.unsqueeze(2).unsqueeze(3) * sim_map  # (K,C,H,W)
#             matched_pts = torch.sum(matched_pts, dim=0).unsqueeze(0)  # (1,C,H,W)

#             matched_fts = torch.cat([qry_ft, matched_pts, sim_0.unsqueeze(0)], dim=1)  # (1,2C+1,H,W)
#             res.append(matched_fts)

#         res = torch.cat(res, dim=0)  # (Nq, 2C+1, H, W)
#         return res

#     def get_fuse_factor(self, qry_fts, sup_fts, mask):
#         """
#         Args:
#             qry_fts: (1, H, W, 512)
#             sup_fts: (1, 512, Hs, Ws)
#             mask:    (1, H_img, W_img)
#         Returns:
#             G: scalar tensor
#         """
#         mask_ = F.interpolate(
#             mask.unsqueeze(0).float(),
#             size=sup_fts.shape[-2:],
#             mode='bilinear',
#             align_corners=True
#         ).squeeze(0)

#         if mask_.sum() == 0:
#             G = torch.tensor(0.0, device=self.device)
#         else:
#             sup_fts_ = F.normalize(
#                 sup_fts.permute(0, 2, 3, 1).reshape(-1, sup_fts.shape[1]), dim=-1
#             )  # (Hs*Ws, C)

#             qry_fts_ = qry_fts.reshape(-1, qry_fts.shape[-1])  # (H*W, C)
#             G = torch.matmul(sup_fts_, qry_fts_.transpose(0, 1))  # (Hs*Ws, H*W)
#             G = G[mask_.reshape(-1) > 0]
#             G = torch.sigmoid(torch.mean(G))

#         return G

#     def get_fg_fts(self, fts, mask):
#         """
#         Args:
#             fts: (1, 512, 256, 256)
#             mask: (1, 256, 256)
#             前景背景公用这个函数
#         """
#         _, c, h, w = fts.shape
#         # select masked fg features
#         fg_fts = fts * mask[None, ...]
#         bg_fts = torch.ones_like(fts) * mask[None, ...]
#         mask_ = mask.view(-1)
#         n_pts = len(mask_) - len(mask_[mask_ == 1])
#         #select_pts[55536, 512]  全是前景特征的地图  背景亦如此
#         select_pts = self.get_random_pts(fts, mask, n_pts)
#         index = bg_fts == 0
#         fg_fts[index] = select_pts.permute(1, 0).reshape(512*n_pts)
#         #全是前景特征的图[1, 512, 256, 256]
#         return fg_fts

#     def get_all_prototypes(self, fg_fts):
#         """
#         Args:
#             fg_fts: way x shot x [all x 512]
#         """
#         n_ways, n_shots = len(fg_fts), len(fg_fts[0])
#         prototypes = [sum([shot for shot in way]) / n_shots for way in fg_fts]

#         return prototypes

#     def get_fg_sim(self, fts, prototypes):
#         """
#         Args:
#             fts: (1, 512, 64, 64)
#             prototypes: (102, 512)
#         """
#         fts_ = fts.permute(0, 2, 3, 1)
#         fts_ = F.normalize(fts_, dim=-1)
#         pts_ = F.normalize(prototypes, dim=-1)
#         fg_sim = torch.matmul(fts_, pts_.transpose(0, 1)).permute(0, 3, 1, 2)
#         #得到 102 张“原始评分表”。

#         #Decoder (“阅卷老师”) 把这 102 张表融合成 1 张 最终的“肿瘤相似度图”。形状: [1, 1, 64, 64]。
#         fg_sim = self.sim_decoder_fg(fg_sim)

#         return fg_sim

#     def get_bg_sim(self, fts, prototypes):
#         """
#         Args:
#             fts: (1, 512, 64, 64)
#             prototypes: (602, 512)
#         """
#         fts_ = fts.permute(0, 2, 3, 1)
#         fts_ = F.normalize(fts_, dim=-1)
#         pts_ = F.normalize(prototypes, dim=-1)
#         bg_sim = torch.matmul(fts_, pts_.transpose(0, 1)).permute(0, 3, 1, 2)
#         bg_sim = self.sim_decoder_bg(bg_sim)

#         return bg_sim
#     def get_masked_fts(self, fts, mask):

#         masked_fts = torch.sum(fts * mask[None, ...], dim=(-2, -1)) \
#                      / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)  # 1 x C

#         return masked_fts

#     def getPrototype(self, fg_fts):
#         """
#         Average the features to obtain the prototype

#         Args:
#             fg_fts: lists of list of foreground features for each way/shot
#                 expect shape: Wa x Sh x [1 x C] (1, 1, (1, 512))
#             bg_fts: lists of list of background features for each way/shot
#                 expect shape: Wa x Sh x [1 x C]
#         """
#         n_ways, n_shots = len(fg_fts), len(fg_fts[0])
#         fg_prototypes = [torch.sum(torch.cat([tr for tr in way], dim=0), dim=0, keepdim=True) / n_shots for way in
#                          fg_fts]  ## concat all fg_fts   (n_way, (1, 512))

#         return fg_prototypes

#     def CI(self, fts, mask, kernel_size=(16, 16), stride=8, flag=1, aspect_ratio_range=(0.5, 2)):
#         # SI
#         b, c, height, width = fts.size()
#         kernel_h, kernel_w = kernel_size
#         n_fg = mask.sum()
#         n_bg = height * width - n_fg
#         fg_pts = []
#         bg_pts = []
#         fg_pts.append(torch.sum(fts * mask[None, ...], dim=(-2, -1)) / (n_fg + 1e-5))  # 1 x C
#         bg_pts.append(torch.sum(fts * (1 - mask[None, ...]), dim=(-2, -1)) / (n_bg + 1e-5))  # 1 x C
#         patch_fts = F.unfold(fts, kernel_size=kernel_size, stride=stride).permute(0, 2, 1).view(1, -1, c, kernel_h,
#                                                                                                 kernel_w)
#         patch_mask = F.unfold(mask.unsqueeze(1), kernel_size=kernel_size, stride=stride).permute(0, 2, 1).view(1, -1,
#                                                                                                                kernel_h,
#                                                                                                                kernel_w).unsqueeze(
#             2)
#         bg_patch_mask = 1 - patch_mask
#         random.seed(9)
#         if flag == 1:
#             fg_indx = torch.sum(patch_mask, dim=(-2, -1)).view(-1) >= min(kernel_h * kernel_w * self.rate, n_fg / 150)
#             fg_protos = torch.sum(patch_fts[:, fg_indx, :, :, :] * patch_mask[:, fg_indx, :, :, :], dim=(-2, -1)) \
#                         / (patch_mask[:, fg_indx, :, :, :].sum(dim=(-2, -1)) + 1e-5)  # (1, n, 512)
#             if fg_protos.numel() == 0:
#                 fg_protos = self.get_pts(fts, mask)
#             else:
#                 fg_protos = fg_protos.squeeze(0)
#                 if len(fg_protos) > 150:
#                     fg_protos = fg_protos[random.sample(range(len(fg_protos)), 150)]
#             fg_pts.append(fg_protos)

#         bg_indx = torch.sum(bg_patch_mask, dim=(-2, -1)).view(-1) >= min(kernel_h * kernel_w, n_bg / 250)
#         bg_protos = torch.sum(patch_fts[:, bg_indx, :, :, :] * bg_patch_mask[:, bg_indx, :, :, :], dim=(-2, -1)) \
#                     / (bg_patch_mask[:, bg_indx, :, :, :].sum(dim=(-2, -1)) + 1e-5)
#         if bg_protos.numel() == 0:
#             bg_protos = self.get_pts(fts, 1 - mask)
#         else:
#             bg_protos = bg_protos.squeeze(0)
#             if len(bg_protos) > 250:
#                 bg_protos = bg_protos[random.sample(range(len(bg_protos)), 250)]
#         bg_pts.append(bg_protos)

#         # RI
#         min_aspect_ratio, max_aspect_ratio = aspect_ratio_range
#         assert min_aspect_ratio > 0
#         assert max_aspect_ratio > min_aspect_ratio
#         torch.manual_seed(9)
#         aspect_ratio = torch.rand(self.num).to(self.device) * (max_aspect_ratio - min_aspect_ratio) + min_aspect_ratio
#         aspect_ratio0 = aspect_ratio[aspect_ratio <= 1]
#         bbox_height0 = (
#                     (torch.rand(len(aspect_ratio0)).to(self.device) * 0.9375 + 0.0625) * (height * aspect_ratio0)).int()
#         bbox_width0 = (bbox_height0 / aspect_ratio0).int()

#         aspect_ratio1 = aspect_ratio[aspect_ratio > 1]
#         bbox_width1 = (
#                     (torch.rand(len(aspect_ratio1)).to(self.device) * 0.9375 + 0.0625) * (width / aspect_ratio1)).int()
#         bbox_height1 = (bbox_width1 * aspect_ratio1).int()

#         bbox_height = torch.cat([bbox_height0, bbox_height1], dim=0)
#         bbox_width = torch.cat([bbox_width0, bbox_width1], dim=0)

#         x1 = (torch.rand(self.num).to(self.device) * (height - bbox_height)).int()
#         y1 = (torch.rand(self.num).to(self.device) * (width - bbox_width)).int()
#         x2 = x1 + bbox_height
#         y2 = y1 + bbox_width
#         for i in range(self.num):
#             n1 = mask[:, x1[i]:x2[i], y1[i]:y2[i]].sum()
#             n0 = (1 - mask[:, x1[i]:x2[i], y1[i]:y2[i]]).sum()
#             if n1 >= min(bbox_height[i] * bbox_width[i] * self.rate, n_fg / self.num) and n1 < n_fg:
#                 prototype = self.get_masked_fts(fts[:, :, x1[i]:x2[i], y1[i]:y2[i]], mask[:, x1[i]:x2[i], y1[i]:y2[i]])
#                 fg_pts.append(prototype)
#             if n0 >= min(bbox_height[i] * bbox_width[i], n_bg / self.num) and n0 < n_bg:
#                 prototype = self.get_masked_fts(fts[:, :, x1[i]:x2[i], y1[i]:y2[i]],
#                                                 1 - mask[:, x1[i]:x2[i], y1[i]:y2[i]])
#                 bg_pts.append(prototype)

#         fg_pts = torch.cat(fg_pts, dim=0)
#         bg_pts = torch.cat(bg_pts, dim=0)

#         return [bg_pts, fg_pts]
#     def get_pts(self, fts, mask):
#         features_trans = fts.squeeze(0)
#         features_trans = features_trans.permute(1, 2, 0)
#         features_trans = features_trans.view(features_trans.shape[-2] * features_trans.shape[-3],
#                                              features_trans.shape[-1])
#         mask = mask.squeeze(0).view(-1)
#         indx = mask == 1
#         features_trans = features_trans[indx]  # (n_fg x 512)
#         if len(features_trans) > 100:
#             random.seed(9)
#             features_trans = features_trans[random.sample(range(len(features_trans)), 100)]
#         return features_trans

#     def get_aux_loss(self, sup_fg_pts, qry_fg_pts, sup_bg_pts, qry_bg_pts):
#         """
#         Args:
#             sup_fg_pts: (102, 512)
#             qry_fg_pts: (102, 512)
#             sup_bg_pts: (602, 512)
#             qry_bg_pts: (602, 512)
#         """
#         d1 = torch.mean(sup_fg_pts, dim=0, keepdim=True)
#         d2 = torch.mean(qry_fg_pts, dim=0, keepdim=True)
#         b1 = torch.mean(sup_bg_pts, dim=0, keepdim=True)
#         b2 = torch.mean(qry_bg_pts, dim=0, keepdim=True)

#         d1 = F.normalize(d1, dim=-1)
#         d2 = F.normalize(d2, dim=-1)
#         b1 = F.normalize(b1, dim=-1)
#         b2 = F.normalize(b2, dim=-1)

#         fg_intra = torch.matmul(d1, d2.transpose(0, 1)).squeeze(0).squeeze(0)
#         bg_intra = torch.matmul(b1, b2.transpose(0, 1)).squeeze(0).squeeze(0)
#         intra_loss = 2 - fg_intra - bg_intra

#         zero = torch.zeros(1).to(self.device).squeeze(0)
#         sup_inter = torch.matmul(d1, b1.transpose(0, 1))
#         qry_inter = torch.matmul(d2, b2.transpose(0, 1))
#         inter_loss = torch.max(zero, torch.mean(sup_inter)) + torch.max(zero, torch.mean(qry_inter))

#         return intra_loss + inter_loss
"""
FSS via CoW
Extended from ADNet code by Hansen et al.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder import Res101Encoder
from models.modules import MLP, Decoder, Supp_Decoder
import numpy as np
import random
import cv2
from boundary_loss import BoundaryLoss
from models.Decoders import Decoder as IFRDecoder


class FewShotSeg(nn.Module):
    def __init__(self, pretrained_weights="deeplabv3", ablation_variant="full"):
        super().__init__()
        from models.ablation import module_flags
        self.ablation_variant = ablation_variant
        flags = module_flags(ablation_variant)
        self.use_sepc = flags["sepc"]
        self.use_qrr = flags["qrr"]
        self.use_swpa = flags["swpa"]
        # Encoder
        self.encoder = Res101Encoder(replace_stride_with_dilation=[True, True, False],
                                     pretrained_weights=pretrained_weights)
        self.device = torch.device('cuda')
        self.scaler = 20.0
        self.rate = 0.95
        self.kernel = (16, 8)
        self.stride = 8
        self.num = 100
        #它与 LogSoftmax（或在这个代码中的 torch.log(torch.softmax(...))）配合使用，用来衡量模型预测的像素类别与真实标签之间的差距。
        self.criterion = nn.NLLLoss()
        #专门用来惩罚模型在物体边缘（边界）处预测不准确的情况，能让分割结果的边缘更清晰、更贴合真实物体。
        self.criterion_b = BoundaryLoss(theta0=3, theta=5)
        #ean Squared Error Loss (均方误差损失)。通常用于回归任务（预测一个连续的数值）。在这个项目中，它很可能是用在 get_aux_loss（辅助损失）中，用来拉近支持集原型和查询集原型（它们都是特征向量）之间的距离。
        self.criterion_MSE = nn.MSELoss()
        #定义了要从“弱点”区域挖掘 100 个困难前景 (Hard Foreground) 原型。
        self.fg_num = 100
        #定义了要挖掘 600 个困难背景 (Hard Background) 原型。
        self.bg_num = 600
        self.ifr_chunk_size = 32
        self.ifr_decoder_fg = IFRDecoder(in_ch=1025)
        self.ifr_decoder_bg = IFRDecoder(in_ch=1025)
        self.mlp1 = MLP(256, self.fg_num)
        self.mlp2 = MLP(256, self.bg_num)
        self.fuse = nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0, bias=False)
        #它的输入是查询集特征与 100 个前景原型计算出的“相似度图”，输出是最终的前景分割图。它负责**“融合”**这 100 个原型的意见。
        self.sim_decoder_fg = Decoder(self.fg_num)
        #同上，负责**“融合”** 600 个背景原型的意见，生成背景分割图。
        self.sim_decoder_bg = Decoder(self.bg_num)
        #它用于支持集自我预测 (SSP) 模块。在 getSelfPred 方法中，它接收支持集特征 F_s 和朴素原型 p 的拼接结果，并输出一个初始的、粗糙的自我预测掩码 M_sp
        self.supp_decoder = Supp_Decoder()

    def forward(self, supp_imgs, supp_mask, qry_imgs, train=False):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W]
            supp_mask: foreground masks for support images
                way x shot x [B x H x W]
            qry_imgs: query images
                N x [B x 3 x H x W]
            train: whether to train model or not
        """
        self.n_ways = len(supp_imgs)
        self.n_shots = len(supp_imgs[0])
        self.n_queries = len(qry_imgs)
        assert self.n_ways == 1  # for now only one-way, because not every shot has multiple sub-images
        assert self.n_queries == 1

        qry_bs = qry_imgs[0].shape[0]
        supp_bs = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]
        supp_mask = torch.stack([torch.stack(way, dim=0) for way in supp_mask],
                                dim=0).view(supp_bs, self.n_ways, self.n_shots, *img_size)

        ###### Extract features ######
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0), ], dim=0)
        img_fts, tao = self.encoder(imgs_concat)

        supp_fts = [img_fts[dic][:self.n_ways * self.n_shots * supp_bs].view(
            supp_bs, self.n_ways, self.n_shots, -1, *img_fts[dic].shape[-2:]) for _, dic in enumerate(img_fts)]
        supp_fts = supp_fts[0]

        qry_fts = [img_fts[dic][self.n_ways * self.n_shots * supp_bs:].view(
            qry_bs, self.n_queries, -1, *img_fts[dic].shape[-2:]) for _, dic in enumerate(img_fts)]
        qry_fts = qry_fts[0]

        ###### Get threshold ######
        self.t = tao[self.n_ways * self.n_shots * supp_bs:]
        self.thresh_pred = [self.t for _ in range(self.n_ways)]
        supp_tao = tao[:self.n_ways * self.n_shots * supp_bs].view(supp_bs, self.n_ways, self.n_shots, -1)
        qry_tao = tao[self.n_ways * self.n_shots * supp_bs:].view(qry_bs, self.n_queries, -1)

        ###### Compute loss ######
        align_loss = torch.zeros(1).to(self.device)
        b_loss = torch.zeros(1).to(self.device)
        ssp_loss = torch.zeros(1).to(self.device)
        aux_loss = torch.zeros(1).to(self.device)
        outputs = []
        for epi in range(supp_bs):
            ###### Extract prototypes ######
            # 1. 上采样 support feature 到 mask 尺寸
            supp_fts_ = [[F.interpolate(supp_fts[[epi], way, shot], size=img_size,
                                        mode='bilinear', align_corners=True)
                          for shot in range(self.n_shots)] for way in range(self.n_ways)]

            # 2. 提取全局前景原型 / 全局背景原型
            prototype = [[self.get_masked_fts(supp_fts_[way][shot], supp_mask[[epi], way, shot])
                          for shot in range(self.n_shots)] for way in range(self.n_ways)]
            prototype = self.getPrototype(prototype)

            bg_prototype = [[self.get_masked_fts(supp_fts_[way][shot], 1 - supp_mask[[epi], way, shot])
                             for shot in range(self.n_shots)] for way in range(self.n_ways)]
            bg_prototype = self.getPrototype(bg_prototype)

            if self.use_sepc:
                # 3. 用 CI 提取区域原型
                if supp_mask[epi][0].sum() == 0:
                    region_protos = [[self.CI(supp_fts_[way][shot], supp_mask[[epi], way, shot],
                                              self.kernel, self.stride, flag=0)
                                      for shot in range(self.n_shots)] for way in range(self.n_ways)]
                else:
                    region_protos = [[self.CI(supp_fts_[way][shot], supp_mask[[epi], way, shot],
                                              self.kernel, self.stride, flag=1)
                                      for shot in range(self.n_shots)] for way in range(self.n_ways)]

                bg_pts = [torch.cat([region_protos[way][0][0], bg_prototype[way]], dim=0) for way in range(self.n_ways)]
                fg_pts = [torch.cat([region_protos[way][0][1], prototype[way]], dim=0) for way in range(self.n_ways)]
            else:
                # Parameter-matched global context for the support decoder.
                fg_pts, bg_pts = prototype, bg_prototype

            # support self-prediction using prototype groups from CI
            supp_preds = [[self.getSelfPred(
                supp_fts_[way][shot],  # [1,C,H,W]
                fg_pts[way],  # fg_pts
                bg_pts[way]  # bg_pts
            ) for shot in range(self.n_shots)] for way in range(self.n_ways)]

            # current 1-way 1-shot case
            supp_pred = supp_preds[0][0]
            supp_pred = F.interpolate(supp_pred, size=img_size, mode='bilinear', align_corners=True)
            supp_pred = torch.cat((1.0 - supp_pred, supp_pred), dim=1)
            self_pred = supp_pred
            supp_pred = torch.argmax(supp_pred, dim=1, keepdim=True).squeeze(1)
            #supp_pred[1, 512, 256, 256]

            fg_pts = [[self.get_fg_pts(supp_fts[[epi], way, shot], supp_mask[[epi], way, shot], supp_pred)
                       for shot in range(self.n_shots)] for way in range(self.n_ways)]
            # fg_pts = [
            #              < 张量
            # shape = [102, 512] >,  # Way 0 的最终平均团队
            # < 张量
            # shape = [102, 512] >  # Way 1 的最终平均团队
            # ]   变得更平均了  少了shot的一个嵌套
            fg_pts = self.get_all_prototypes(fg_pts)
            bg_pts = [[self.get_bg_pts(supp_fts[[epi], way, shot], supp_mask[[epi], way, shot], supp_pred)
                       for shot in range(self.n_shots)] for way in range(self.n_ways)]
            bg_pts = self.get_all_prototypes(bg_pts)

            # ----- threshold split -----
            # suppose tao shape is [num_images, 1] or compatible scalar tensor

            # =========================
            # Query prediction via IFR_FD
            # =========================
            fg_matched_fts = torch.stack([
                self.IFR_FD(
                    fg_pts[way],  # (102, 512), last one is global fg prototype
                    qry_fts[epi],  # (1, 512, H', W')
                    supp_fts[[epi], way, 0],  # (1, 512, H', W')
                    supp_mask[[epi], way, 0],  # (1, H, W)
                    qry_tao[epi, 0],  # query threshold
                    supp_tao[epi, way, 0]  # support threshold
                )
                for way in range(self.n_ways)
            ], dim=1)  # (Nq, Wa, 1025, H', W')

            bg_matched_fts = torch.stack([
                self.IFR_FD(
                    bg_pts[way],  # (602, 512), last one is global bg prototype
                    qry_fts[epi],  # (1, 512, H', W')
                    supp_fts[[epi], way, 0],  # (1, 512, H', W')
                    1 - supp_mask[[epi], way, 0],  # background mask
                    qry_tao[epi, 0],
                    supp_tao[epi, way, 0]
                )
                for way in range(self.n_ways)
            ], dim=1)  # (Nq, Wa, 1025, H', W')

            fg_pred = torch.cat([
                self.ifr_decoder_fg(fg_matched_fts[:, way]) for way in range(self.n_ways)
            ], dim=1)  # (Nq, Wa, H', W') or (Nq, Wa, 1, H', W') depending on decoder

            bg_pred = torch.cat([
                self.ifr_decoder_bg(bg_matched_fts[:, way]) for way in range(self.n_ways)
            ], dim=1)

            fg_pred = F.interpolate(fg_pred, size=img_size, mode='bilinear', align_corners=True)
            bg_pred = F.interpolate(bg_pred, size=img_size, mode='bilinear', align_corners=True)

            preds = torch.cat([bg_pred, fg_pred], dim=1)
            preds = torch.softmax(preds, dim=1)

            outputs.append(preds)
            if train:
                align_loss_epi, aux_loss_epi, b_loss_epi, ssp_loss_epi = self.align_aux_Loss(supp_fts[epi], qry_fts[epi], supp_mask[epi], preds,
                                                                                             fg_pts, bg_pts, self_pred)
                align_loss += align_loss_epi
                aux_loss += aux_loss_epi
                b_loss += b_loss_epi
                ssp_loss += ssp_loss_epi
            #您的写法 (推荐)：
            # Python
            #
            # output = torch.cat(outputs, dim=0)评价：脱裤子放屁——多此一举。
        output = torch.stack(outputs, dim=1)
        output = output.view(-1, *output.shape[2:])

        return output, align_loss / supp_bs, aux_loss / supp_bs, b_loss / supp_bs, ssp_loss / supp_bs

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

    def getSelfPred(self, supp_fts, fg_pts, bg_pts):
        """
        Args:
            supp_fts: (1, 512, H, W)
            fg_pts:   (N_fg, 512)
            bg_pts:   (N_bg, 512)
        Returns:
            supp_pred: (1, 1, H, W)
        """
        if not self.use_sepc:
            fg_pts, bg_pts = fg_pts[-1:], bg_pts[-1:]
        B, C, H, W = supp_fts.shape

        # (1, H, W, C)
        fts_hw = supp_fts.permute(0, 2, 3, 1)
        fts_norm = F.normalize(fts_hw, dim=-1)

        # ---- foreground context ----
        fg_norm = F.normalize(fg_pts, dim=-1)  # (N_fg, C)
        fg_sim = torch.matmul(fts_norm, fg_norm.t())  # (1, H, W, N_fg)
        fg_attn = torch.softmax(fg_sim * self.scaler, dim=-1)
        ctx_fg = torch.matmul(fg_attn, fg_pts).permute(0, 3, 1, 2)  # (1, C, H, W)

        # ---- background context ----
        bg_norm = F.normalize(bg_pts, dim=-1)  # (N_bg, C)
        bg_sim = torch.matmul(fts_norm, bg_norm.t())  # (1, H, W, N_bg)
        bg_attn = torch.softmax(bg_sim * self.scaler, dim=-1)
        ctx_bg = torch.matmul(bg_attn, bg_pts).permute(0, 3, 1, 2)  # (1, C, H, W)

        # ---- original feature + fg context + bg context ----
        supp_input = torch.cat([supp_fts, ctx_fg, ctx_bg], dim=1)  # (1, 1536, H, W)
        supp_pred = self.supp_decoder(supp_input)

        return supp_pred
    #的作用是计算支持集图像前景区域的【平均特征】
    def getFeatures(self, fts, mask):
        """
        Arg0.s:
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
        fg_pixels = binary_masks[1].sum().item()
        is_skipped = int(binary_masks[1].sum() == 0)
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
        """
        Args:
            supp_fts: (way, shot, 512, 64, 64)
            qry_fts: (N, 512, 64, 64)
            fore_mask: (way, shot, 256, 256)
            pred: (1, 2, 256, 256)
            sup_fg_pts: way x [102 x 512]
            sup_bg_pts: way x [602 x 512]
            self_pred: (1, 2, 256, 256)
        """
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Get query mask
        pred_mask = pred.argmax(dim=1, keepdim=True).squeeze(1)

        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        fg_pixels = binary_masks[1].sum().item()
        is_skipped = int(fg_pixels == 0)
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=0).float()
        # Compute the support loss
        loss = torch.zeros(1).to(self.device)
        loss_aux = torch.zeros(1).to(self.device)
        b_loss = torch.zeros(1).to(self.device)
        ssp_loss = torch.zeros(1).to(self.device)
        for way in range(n_ways):
            use_support_fallback = bool((pred_mask[way + 1].sum() == 0).item())
    # 只有全背景时才启用 fallback
            if use_support_fallback:
                qry_seed_prob = self.getSelfPred(qry_fts, sup_fg_pts[way], sup_bg_pts[way])
                qry_seed_prob = F.interpolate(
                    qry_seed_prob,
                    size=fore_mask.shape[-2:],
                    mode='bilinear',
                    align_corners=True
                )

                qry_seed_2c = torch.cat((1.0 - qry_seed_prob, qry_seed_prob), dim=1)
                qry_seed_mask = torch.argmax(qry_seed_2c, dim=1, keepdim=True).squeeze(1)

                # 如果连 support-only 原型都分不出前景，那再跳过
                if qry_seed_mask.sum().item() == 0:
                    continue

                cur_qry_mask = qry_seed_mask.float()

            else:
                cur_qry_mask = pred_mask[way + 1]

            # Get the query prototypes
            for shot in range(n_shots):
                qry_fts_ = [[self.getFeatures(qry_fts, cur_qry_mask)]]
                fg_prototypes = self.getPrototype(qry_fts_)

                qry_bg_fts_ = [[self.getFeatures(qry_fts, 1.0 - cur_qry_mask)]]
                bg_prototypes = self.getPrototype(qry_bg_fts_)

                qry_pred = self.getSelfPred(qry_fts, fg_prototypes[0], bg_prototypes[0])
                qry_pred = F.interpolate(qry_pred, size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)
                qry_pred = torch.cat((1.0 - qry_pred, qry_pred), dim=1)
                qry_pred = torch.argmax(qry_pred, dim=1, keepdim=True).squeeze(1)

                fg_pts_ = [[self.get_fg_pts(qry_fts, cur_qry_mask, qry_pred)]]
                fg_pts_ = self.get_all_prototypes(fg_pts_)

                bg_pts_ = [[self.get_bg_pts(qry_fts, cur_qry_mask, qry_pred)]]
                bg_pts_ = self.get_all_prototypes(bg_pts_)

                loss_aux += self.get_aux_loss(sup_fg_pts[way], fg_pts_[way], sup_bg_pts[way], bg_pts_[way])

                # Get predictions
                supp_pred = self.get_fg_sim(supp_fts[way, [shot]], fg_pts_[way])
                bg_pred_ = self.get_bg_sim(supp_fts[way, [shot]], bg_pts_[way])
                supp_pred = F.interpolate(supp_pred, size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)
                bg_pred_ = F.interpolate(bg_pred_, size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)

                # Combine predictions
                preds = torch.cat([bg_pred_, supp_pred], dim=1)
                preds = torch.softmax(preds, dim=1)

                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255, device=fore_mask.device)
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[fore_mask[way, shot] == 0] = 0

                # Compute Loss
                eps = torch.finfo(torch.float32).eps
                log_prob = torch.log(torch.clamp(preds, eps, 1 - eps))
                loss += self.criterion(log_prob, supp_label[None, ...].long()) / n_shots / n_ways

                ssp_log_prob = torch.log(torch.clamp(self_pred, eps, 1 - eps))
                ssp_loss += self.criterion(ssp_log_prob, supp_label[None, ...].long()) / n_shots / n_ways

        return loss, loss_aux, b_loss, ssp_loss

    def get_fg_pts(self, features, mask, pred_mask):
        """
        Args:
        features: (1, 512, 64, 64)
        mask: (1, 256, 256)
        软弱原型  也就是平均原型
        pred_mask: (1, 256, 256)
        """
        target_size = (256, 256)
        features_trans = F.interpolate(features, size=target_size, mode='bilinear', align_corners=True)
        mask = F.interpolate(mask.unsqueeze(1).float(), size=target_size, mode='nearest').squeeze(1)
        pred_mask = F.interpolate(pred_mask.unsqueeze(1).float(), size=target_size, mode='nearest').squeeze(1)
    
        
        #边缘可以加入注意力机制
        ie_mask = mask.squeeze(0) - torch.tensor(cv2.erode(mask.squeeze(0).cpu().numpy(), np.ones((3, 3), dtype=np.uint8), iterations=2)).to(self.device)
        ie_mask = ie_mask.unsqueeze(0)
        #内边缘原型  两层一层内边缘还带着一圈前景
        ie_prototype = torch.sum(features_trans * ie_mask[None, ...], dim=(-2, -1)) \
                       / (ie_mask[None, ...].sum(dim=(-2, -1)) + 1e-5)
        #平均原型  也就是软弱原型
        origin_prototype = torch.sum(features_trans * mask[None, ...], dim=(-2, -1)) \
                           / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)

        add_mask = (pred_mask.float() + mask).long()
        mask1 = torch.zeros_like(mask)
        mask2 = torch.zeros_like(mask)
        mask1[add_mask == 2] = 1
        mask2[add_mask == 1] = 1
        mask1[mask == 0] = 0
        mask2[mask == 0] = 0


        #fg_fts #全是前景特征的图[1, 512, 256, 256]
        fg_fts = self.get_fg_fts(features_trans, mask)
       
        #可以修改更好的总结fg_prototypes = [100, 512]你可以把 mlp1 想象成一个**“会议记录员”**。

        # 输入 (fg_fts)：是一场 65,536 人 参加的大会（全图像素）。
        #
        # 关键点：这 65,536 人全部都是“前景派”的人（因为我们之前填坑了）。
        #
        # 过程 (mlp1(...))：记录员听取了所有人的发言。
        #
        # 输出 (fg_prototypes)：记录员整理出了一份包含 100 条 关键信息的**“会议纪要”**。
        #
        # 这 100 条纪要（原型），就是对那 6 万多名前景像素特征的高度概括。
        fg_prototypes = self.mlp1(fg_fts.view(512, 256 * 256)).permute(1, 0)
        

        if not self.use_sepc:
            mask2 = mask  # Same-class sampling, no FN-based selection.
        if torch.sum(mask2[mask2 == 1]) > 0:
            #features_trans = features_trans[mask == 1]
            #hard_fg  是在假阴性mask2地图中挑选的弱点
            hard_fg = self.get_random_pts(features_trans, mask2, 50)
            k = random.sample(range(len(fg_prototypes)), 50)
            fg_prototypes = torch.cat([fg_prototypes[k], hard_fg], dim=0)
        fg_pts = torch.cat([
            fg_prototypes,  # local / hard prototypes
            ie_prototype,  # edge prototype also interacts with query
            origin_prototype  # global prototype MUST be last
        ], dim=0)  # (102, 512)
        

        #[102, 512]  fg_prototypes
        return fg_pts

    def get_bg_pts(self, features, mask, pred_mask):
        """
        Args:
            features: (1, 512, 64, 64)
            mask: (1, 256, 256)
            pred_mask: (1, 256, 256)
        """
        target_size = (256, 256)

        features_trans = F.interpolate(features, size=target_size, mode='bilinear', align_corners=True)
        mask = F.interpolate(mask.unsqueeze(1).float(), size=target_size, mode='nearest').squeeze(1)
        pred_mask = F.interpolate(pred_mask.unsqueeze(1).float(), size=target_size, mode='nearest').squeeze(1)
    

        bg_mask = 1 - mask
        #features_trans = F.interpolate(features, size=bg_mask.shape[-2:], mode='bilinear', align_corners=True)

        oe_mask = torch.tensor(cv2.dilate(mask.squeeze(0).cpu().numpy(), np.ones((3, 3), dtype=np.uint8), iterations=2)).to(self.device) - mask.squeeze(0)
        oe_mask = oe_mask.unsqueeze(0)
        #外边缘原型
        oe_prototype = torch.sum(features_trans * oe_mask[None, ...], dim=(-2, -1)) \
                       / (oe_mask[None, ...].sum(dim=(-2, -1)) + 1e-5)
        #背景平均原型
        origin_prototype = torch.sum(features_trans * bg_mask[None, ...], dim=(-2, -1)) \
                           / (bg_mask[None, ...].sum(dim=(-2, -1)) + 1e-5)

        add_mask = (pred_mask.float() + mask).long()
        mask1 = torch.zeros_like(mask)#高度置信背景
        mask2 = torch.zeros_like(mask)#假阳性
        mask1[add_mask == 0] = 1
        mask2[add_mask == 1] = 1
        mask1[bg_mask == 0] = 0
        mask2[bg_mask == 0] = 0

        bg_fts = self.get_fg_fts(features_trans, bg_mask)
        
        bg_prototypes = self.mlp2(bg_fts.view(512, 256 * 256)).permute(1, 0)
        
        if not self.use_sepc:
            mask2 = bg_mask  # Same-class sampling, no FP-based selection.
        if torch.sum(mask2[mask2 == 1]) > 0:
            hard_bg = self.get_random_pts(features_trans, mask2, 100)
            k = random.sample(range(len(bg_prototypes)), 500)
            bg_prototypes = torch.cat([bg_prototypes[k], hard_bg], dim=0)
        # global prototype MUST be last
        bg_pts = torch.cat([
            bg_prototypes,  # local / hard bg prototypes
            oe_prototype,  # edge prototype
            origin_prototype  # global bg prototype MUST be last
        ], dim=0)  # (602, 512)
        

        return bg_pts

    def get_random_pts(self, features_trans, mask, n_prototype):
        """
        Args:
            features_trans: (1, 512, 256, 256)
            mask: (1, 256, 256)
            n_prototype: int
        """
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
        ##select_pts[55536, 512]  全是前景特征的地图
        return prototypes

    """ def IFR_FD(self, pts, qry_fts, sup_fts, mask, qry_thresh, sup_thresh):
        
        Args:
            pts:        (K+1, C)
                        pts[:-1] = local prototypes (hard + edge)
                        pts[-1]  = global prototype
            qry_fts:    (Nq, C, H, W)
            sup_fts:    (1, C, H, W)
            mask:       (1, H_img, W_img)
            qry_thresh: scalar tensor
            sup_thresh: scalar tensor

        Returns:
            matched_fts: (Nq, 2C+1, H, W)
        
        n, c, h, w = qry_fts.shape

        # split local/global prototypes
        prototype = pts[-1].unsqueeze(0)  # (1, C), global prototype
        pts = pts[:-1]  # (K, C), local prototypes

        pts = F.normalize(pts, dim=-1)
        prototype = F.normalize(prototype, dim=-1)

        qry_scale = 1 + self.scaler * torch.sigmoid(qry_thresh)
        sup_scale = 1 + self.scaler * torch.sigmoid(sup_thresh)

        res = []
        for i in range(n):
            qry_ft = qry_fts[i].unsqueeze(0)  # (1, C, H, W)
            fts_ = qry_ft.permute(0, 2, 3, 1)  # (1, H, W, C)
            fts_ = F.normalize(fts_, dim=-1)

            # ---------------- global similarity ----------------
            one_sim = torch.matmul(fts_, prototype.transpose(0, 1)).permute(0, 3, 1, 2)  # (1,1,H,W)

            # ---------------- IFR ----------------
            sim = torch.matmul(fts_, pts.transpose(0, 1)).permute(3, 0, 1, 2)  # (K,1,H,W)
            pseudo_mask = torch.sigmoid(sim * qry_scale)  # (K,1,H,W)

            # query-guided prototype update
            qry_ft_expand = qry_ft.expand(sim.shape[0], -1, -1, -1)  # (K,C,H,W)
            qry_update = torch.sum(qry_ft_expand * pseudo_mask, dim=(-2, -1)) \
                         / (pseudo_mask.sum(dim=(-2, -1)) + 1e-5)  # (K,C)

            G = self.get_fuse_factor(fts_, sup_fts, mask)  # scalar

            fuse_in = torch.stack([pts * (1 - G), qry_update * G], dim=1)  # (K,2,C)
            fuse_in = fuse_in.permute(0, 1, 2).unsqueeze(-1)  # (K,2,C,1)
            delta = self.fuse(fuse_in).squeeze(1).squeeze(-1)  # (K,C)

            pts_refined = pts + delta  # (K,C # (K,C)

            # ---------------- FD ----------------
            pts_norm = F.normalize(pts_refined, dim=-1)
            sim = torch.matmul(fts_, pts_norm.transpose(0, 1)).permute(3, 0, 1, 2)  # (K,1,H,W)

            sim_map = torch.softmax(sim * sup_scale, dim=0)  # (K,1,H,W)
            sim_0 = torch.sum(sim * sim_map, dim=0)  # (1,H,W)

            matched_pts = pts_refined.unsqueeze(2).unsqueeze(3) * sim_map  # (K,C,H,W)
            matched_pts = torch.sum(matched_pts, dim=0).unsqueeze(0)  # (1,C,H,W)

            matched_fts = torch.cat([qry_ft, matched_pts, sim_0.unsqueeze(0)], dim=1)  # (1,2C+1,H,W)
            res.append(matched_fts)

        res = torch.cat(res, dim=0)  # (Nq, 2C+1, H, W)
        return res """
    def IFR_FD(self, pts, qry_fts, sup_fts, mask, qry_thresh, sup_thresh):
        """
        Chunked low-memory version of IFR_FD.
        与原版保持同样的输入输出接口，只是把 prototype 维 K 分块计算，降低峰值显存。

        Args:
            pts:        (K+1, C)
                        pts[:-1] = local prototypes (hard + edge)
                        pts[-1]  = global prototype
            qry_fts:    (Nq, C, H, W)
            sup_fts:    (1, C, H, W)
            mask:       (1, H_img, W_img)
            qry_thresh: scalar tensor
            sup_thresh: scalar tensor

        Returns:
            matched_fts: (Nq, 2C+1, H, W)
        """
        n, c, h, w = qry_fts.shape

        # split local/global prototypes
        prototype = pts[-1].unsqueeze(0)   # (1, C), global prototype
        pts = pts[:-1]                     # (K, C), local prototypes

        pts = F.normalize(pts, dim=-1)
        prototype = F.normalize(prototype, dim=-1)

        qry_scale = 1 + self.scaler * torch.sigmoid(qry_thresh)
        sup_scale = 1 + self.scaler * torch.sigmoid(sup_thresh)

        chunk_size = self.ifr_chunk_size
        res = []

        for i in range(n):
            qry_ft = qry_fts[i].unsqueeze(0)                 # (1, C, H, W)
            fts_ = qry_ft.permute(0, 2, 3, 1)               # (1, H, W, C)
            fts_ = F.normalize(fts_, dim=-1)

            # ---------------- global similarity ----------------
            one_sim = torch.matmul(fts_, prototype.transpose(0, 1)).permute(0, 3, 1, 2)  # (1,1,H,W)

            # ============================================================
            # Stage 1: IFR  (chunk over K)
            # ============================================================
            K = pts.shape[0]
            if self.use_qrr:
                G = self.get_fuse_factor(fts_, sup_fts, mask)   # scalar
                refined_chunks = []

                for start in range(0, K, chunk_size):
                    end = min(start + chunk_size, K)
                    pts_chunk = pts[start:end]                                  # (kc, C)

                    # sim_chunk: (kc, 1, H, W)
                    sim_chunk = torch.matmul(fts_, pts_chunk.transpose(0, 1)).permute(3, 0, 1, 2)
                    pseudo_mask = torch.sigmoid(sim_chunk * qry_scale)          # (kc,1,H,W)

                    # 不再显式 expand query feature，利用广播减少峰值显存
                    qry_update = torch.sum(qry_ft * pseudo_mask, dim=(-2, -1)) / \
                                (pseudo_mask.sum(dim=(-2, -1)) + 1e-5)         # (kc, C)

                    fuse_in = torch.stack([pts_chunk * (1 - G), qry_update * G], dim=1)  # (kc,2,C)
                    fuse_in = fuse_in.unsqueeze(-1)                                        # (kc,2,C,1)
                    delta = self.fuse(fuse_in).squeeze(1).squeeze(-1)                      # (kc,C)

                    pts_refined_chunk = pts_chunk + delta                                   # (kc,C)
                    refined_chunks.append(pts_refined_chunk)

                    # 手动释放块内中间变量，帮助降低峰值
                    del sim_chunk, pseudo_mask, qry_update, fuse_in, delta, pts_refined_chunk

                pts_refined = torch.cat(refined_chunks, dim=0)                              # (K,C)
            else:
                pts_refined = pts

            # ============================================================
            # Stage 2: FD  (strict/global softmax over K with 2-pass chunk)
            # ============================================================
            pts_norm = F.normalize(pts_refined, dim=-1)

            if not self.use_swpa:
                # Same 2C+1 decoder interface, uniform weights over all prototypes.
                matched_pts = pts_refined.mean(dim=0).view(1, c, 1, 1).expand(1, c, h, w)
                mean_response = torch.matmul(fts_, pts_norm.mean(dim=0).view(c, 1))
                mean_response = mean_response.permute(0, 3, 1, 2) * sup_scale
                res.append(torch.cat([qry_ft, matched_pts, mean_response], dim=1))
                continue

            # ---------- pass 1: find global max over K ----------
            global_max = None
            for start in range(0, K, chunk_size):
                end = min(start + chunk_size, K)
                pts_chunk = pts_norm[start:end]                                         # (kc,C)

                # logits_chunk: (kc,1,H,W)
                logits_chunk = torch.matmul(fts_, pts_chunk.transpose(0, 1)).permute(3, 0, 1, 2)
                logits_chunk = logits_chunk * sup_scale

                chunk_max = logits_chunk.max(dim=0, keepdim=True).values                # (1,1,H,W)
                global_max = chunk_max if global_max is None else torch.maximum(global_max, chunk_max)

                del logits_chunk, chunk_max

            # ---------- pass 2: compute global denominator ----------
            denom = torch.zeros_like(global_max)                                        # (1,1,H,W)
            for start in range(0, K, chunk_size):
                end = min(start + chunk_size, K)
                pts_chunk = pts_norm[start:end]

                logits_chunk = torch.matmul(fts_, pts_chunk.transpose(0, 1)).permute(3, 0, 1, 2)
                logits_chunk = logits_chunk * sup_scale

                exp_chunk = torch.exp(logits_chunk - global_max)                        # (kc,1,H,W)
                denom = denom + exp_chunk.sum(dim=0, keepdim=True)                      # (1,1,H,W)

                del logits_chunk, exp_chunk

            # ---------- pass 3: weighted sum ----------
                        # ---------- pass 3: weighted sum ----------
            sim_0_num = torch.zeros((1, 1, h, w), device=qry_ft.device, dtype=qry_ft.dtype)
            matched_pts_num = torch.zeros((1, c, h, w), device=qry_ft.device, dtype=qry_ft.dtype)

            for start in range(0, K, chunk_size):
                end = min(start + chunk_size, K)

                pts_chunk_norm = pts_norm[start:end]          # (kc, C)
                pts_chunk_raw = pts_refined[start:end]        # (kc, C)

                logits_chunk = torch.matmul(fts_, pts_chunk_norm.transpose(0, 1)).permute(3, 0, 1, 2)
                logits_chunk = logits_chunk * sup_scale       # (kc, 1, H, W)

                exp_chunk = torch.exp(logits_chunk - global_max)      # (kc, 1, H, W)
                weight_chunk = exp_chunk / (denom + 1e-12)            # (kc, 1, H, W)

                # 关键修复：先在 chunk 维求和，再加到 (1,1,H,W)
                sim_0_num = sim_0_num + (logits_chunk * weight_chunk).sum(dim=0, keepdim=True)

                # matched_pts = sum_k pts_refined_k * weight_k
                matched_pts_chunk = pts_chunk_raw.unsqueeze(2).unsqueeze(3) * weight_chunk   # (kc,C,H,W)
                matched_pts_num = matched_pts_num + matched_pts_chunk.sum(dim=0, keepdim=True)

                del logits_chunk, exp_chunk, weight_chunk, matched_pts_chunk

            sim_0 = sim_0_num.squeeze(0)     # (1, H, W)
            matched_pts = matched_pts_num    # (1, C, H, W)

            # 关键修复：这里要拼 sim_0，不是 one_sim
            matched_fts = torch.cat([qry_ft, matched_pts, sim_0.unsqueeze(0)], dim=1)           # (1,2C+1,H,W)
            res.append(matched_fts)

            del pts_refined, pts_norm, global_max, denom, sim_0_num, matched_pts_num, matched_pts, one_sim

        res = torch.cat(res, dim=0)                                                     # (Nq,2C+1,H,W)
        return res

    def get_fuse_factor(self, qry_fts, sup_fts, mask):
        """
        Args:
            qry_fts: (1, H, W, 512)
            sup_fts: (1, 512, Hs, Ws)
            mask:    (1, H_img, W_img)
        Returns:
            G: scalar tensor
        """
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
            )  # (Hs*Ws, C)

            qry_fts_ = qry_fts.reshape(-1, qry_fts.shape[-1])  # (H*W, C)
            G = torch.matmul(sup_fts_, qry_fts_.transpose(0, 1))  # (Hs*Ws, H*W)
            G = G[mask_.reshape(-1) > 0]
            G = torch.sigmoid(torch.mean(G))

        return G

    def get_fg_fts(self, fts, mask):
        """
        Args:
            fts: (1, 512, 256, 256)
            mask: (1, 256, 256)
            前景背景公用这个函数
        """
        _, c, h, w = fts.shape
        # select masked fg features
        fg_fts = fts * mask[None, ...]
        bg_fts = torch.ones_like(fts) * mask[None, ...]
        mask_ = mask.view(-1)
        n_pts = len(mask_) - len(mask_[mask_ == 1])
        #select_pts[55536, 512]  全是前景特征的地图  背景亦如此
        select_pts = self.get_random_pts(fts, mask, n_pts)
        index = bg_fts == 0
        fg_fts[index] = select_pts.permute(1, 0).reshape(512*n_pts)
        #全是前景特征的图[1, 512, 256, 256]
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
        #得到 102 张“原始评分表”。

        #Decoder (“阅卷老师”) 把这 102 张表融合成 1 张 最终的“肿瘤相似度图”。形状: [1, 1, 64, 64]。
        fg_sim = self.sim_decoder_fg(fg_sim)

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
        bg_sim = self.sim_decoder_bg(bg_sim)

        return bg_sim
    def get_masked_fts(self, fts, mask):

        masked_fts = torch.sum(fts * mask[None, ...], dim=(-2, -1)) \
                     / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)  # 1 x C

        return masked_fts

    def getPrototype(self, fg_fts):
        """
        Average the features to obtain the prototype

        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C] (1, 1, (1, 512))
            bg_fts: lists of list of background features for each way/shot
                expect shape: Wa x Sh x [1 x C]
        """
        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [torch.sum(torch.cat([tr for tr in way], dim=0), dim=0, keepdim=True) / n_shots for way in
                         fg_fts]  ## concat all fg_fts   (n_way, (1, 512))

        return fg_prototypes

    def CI(self, fts, mask, kernel_size=(16, 16), stride=8, flag=1, aspect_ratio_range=(0.5, 2)):
        # SI
        b, c, height, width = fts.size()
        kernel_h, kernel_w = kernel_size
        n_fg = mask.sum()
        n_bg = height * width - n_fg
        fg_pts = []
        bg_pts = []
        fg_pts.append(torch.sum(fts * mask[None, ...], dim=(-2, -1)) / (n_fg + 1e-5))  # 1 x C
        bg_pts.append(torch.sum(fts * (1 - mask[None, ...]), dim=(-2, -1)) / (n_bg + 1e-5))  # 1 x C
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
                        / (patch_mask[:, fg_indx, :, :, :].sum(dim=(-2, -1)) + 1e-5)  # (1, n, 512)
            if fg_protos.numel() == 0:
                fg_protos = self.get_pts(fts, mask)
            else:
                fg_protos = fg_protos.squeeze(0)
                if len(fg_protos) > 150:
                    fg_protos = fg_protos[random.sample(range(len(fg_protos)), 150)]
            fg_pts.append(fg_protos)

        bg_indx = torch.sum(bg_patch_mask, dim=(-2, -1)).view(-1) >= min(kernel_h * kernel_w, n_bg / 250)
        bg_protos = torch.sum(patch_fts[:, bg_indx, :, :, :] * bg_patch_mask[:, bg_indx, :, :, :], dim=(-2, -1)) \
                    / (bg_patch_mask[:, bg_indx, :, :, :].sum(dim=(-2, -1)) + 1e-5)
        if bg_protos.numel() == 0:
            bg_protos = self.get_pts(fts, 1 - mask)
        else:
            bg_protos = bg_protos.squeeze(0)
            if len(bg_protos) > 250:
                bg_protos = bg_protos[random.sample(range(len(bg_protos)), 250)]
        bg_pts.append(bg_protos)

        # RI
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
                prototype = self.get_masked_fts(fts[:, :, x1[i]:x2[i], y1[i]:y2[i]], mask[:, x1[i]:x2[i], y1[i]:y2[i]])
                fg_pts.append(prototype)
            if n0 >= min(bbox_height[i] * bbox_width[i], n_bg / self.num) and n0 < n_bg:
                prototype = self.get_masked_fts(fts[:, :, x1[i]:x2[i], y1[i]:y2[i]],
                                                1 - mask[:, x1[i]:x2[i], y1[i]:y2[i]])
                bg_pts.append(prototype)

        fg_pts = torch.cat(fg_pts, dim=0)
        bg_pts = torch.cat(bg_pts, dim=0)

        return [bg_pts, fg_pts]
    def get_pts(self, fts, mask):
        features_trans = fts.squeeze(0)
        features_trans = features_trans.permute(1, 2, 0)
        features_trans = features_trans.view(features_trans.shape[-2] * features_trans.shape[-3],
                                             features_trans.shape[-1])
        mask = mask.squeeze(0).view(-1)
        indx = mask == 1
        features_trans = features_trans[indx]  # (n_fg x 512)
        if len(features_trans) > 100:
            random.seed(9)
            features_trans = features_trans[random.sample(range(len(features_trans)), 100)]
        return features_trans

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

        zero = torch.zeros(1).to(self.device).squeeze(0)
        sup_inter = torch.matmul(d1, b1.transpose(0, 1))
        qry_inter = torch.matmul(d2, b2.transpose(0, 1))
        inter_loss = torch.max(zero, torch.mean(sup_inter)) + torch.max(zero, torch.mean(qry_inter))

        return intra_loss + inter_loss
