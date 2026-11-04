"""One-time source migration; original bytes are stored in output/revision_20261019/original."""
import ast
from pathlib import Path
import textwrap

ROOT=Path(__file__).resolve().parents[1]
COMMON_FORWARD='''
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
                                         flag=int(bool(sm.any().item())))
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
        elif not bool(sm.any().item()):
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
'''
SHARED_AUX='''
def align_aux_Loss(self, supp_fts, qry_fts, fore_mask, pred, sup_fg_pts, sup_bg_pts, self_pred):
    return self.compute_auxiliary_losses(supp_fts, qry_fts, fore_mask, pred,
                                         sup_fg_pts, sup_bg_pts, self_pred)
'''
SHARED_SAMPLE='''
def get_random_pts(self, features_trans, mask, n_prototype):
    return sample_masked_features(features_trans, mask, n_prototype)
'''
for name,size in [('cow.py',256),('cow_ct.py',257)]:
    path=ROOT/'models'/name
    source=path.read_text(encoding='utf-8')
    if name=='cow.py':
        source='"""BSP MRI model. Original historical/commented source is archived in the revision backup."""\n'+source[source.index('\nimport torch\n')+1:]
    tree=ast.parse(source); cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FewShotSeg')
    lines=source.splitlines(keepends=True); edits=[]
    prototypes=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='getPrototype']
    for node in prototypes[:-1]: edits.append((node.lineno-1,node.end_lineno,''))
    for node in cls.body:
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str) and 'def IFR_FD' in node.value.value:
            edits.append((node.lineno-1,node.end_lineno,''))
        if not isinstance(node,ast.FunctionDef): continue
        replacement={'forward':COMMON_FORWARD,'align_aux_Loss':SHARED_AUX,'get_random_pts':SHARED_SAMPLE}.get(node.name)
        if replacement: edits.append((node.lineno-1,node.end_lineno,textwrap.indent(textwrap.dedent(replacement).strip()+'\n','    ')))
        if node.name=='getSelfPred':
            chunk=''.join(lines[node.lineno-1:node.end_lineno])
            if 'torch.sigmoid(supp_logit)' not in chunk:
                chunk=chunk.replace('supp_pred = self.supp_decoder(supp_input)','supp_logit = self.supp_decoder(supp_input)').replace('supp_pred = self.supp_decoder(supp_pred)','supp_logit = self.supp_decoder(supp_pred)').replace('return supp_pred','return torch.sigmoid(supp_logit)')
            edits.append((node.lineno-1,node.end_lineno,chunk))
        if node.name in ['get_fg_pts','get_bg_pts']:
            chunk=''.join(lines[node.lineno-1:node.end_lineno])
            chunk=chunk.replace('target_size = (256, 256)','target_size = (self.prototype_size, self.prototype_size)')
            chunk=chunk.replace('256 * 256','self.prototype_size ** 2').replace('257 * 257','self.prototype_size ** 2')
            if name=='cow_ct.py':
                first=chunk.index('        features_trans = F.interpolate')
                if node.name=='get_bg_pts':
                    # Move bg_mask creation after resizing; all pools share the CT 257 grid.
                    chunk=chunk.replace('        bg_mask = 1 - mask\n','')
                    first=chunk.index('        features_trans = F.interpolate')
                end=chunk.index('\n',first)
                prefix='''        target_size = (self.prototype_size, self.prototype_size)
        mask = F.interpolate(mask.unsqueeze(1).float(), size=target_size, mode='nearest').squeeze(1)
        pred_mask = F.interpolate(pred_mask.unsqueeze(1).float(), size=target_size, mode='nearest').squeeze(1)
        features_trans = F.interpolate(features, size=target_size, mode='bilinear', align_corners=True)
'''
                if node.name=='get_bg_pts': prefix+='        bg_mask = 1 - mask\n'
                chunk=chunk[:first]+prefix+chunk[end+1:]
            edits.append((node.lineno-1,node.end_lineno,chunk))
    for start,end,chunk in sorted(edits,reverse=True): lines[start:end]=[chunk]
    source=''.join(lines).replace('class FewShotSeg(nn.Module):','class FewShotSeg(EpisodicSupportMixin, nn.Module):')
    source=source.replace('from boundary_loss import BoundaryLoss','from boundary_loss import BoundaryLoss\nfrom models.episodic import EpisodicSupportMixin, binary_probabilities, sample_masked_features')
    source=source.replace('def __init__(self, pretrained_weights="deeplabv3"):',f'def __init__(self, pretrained_weights="deeplabv3", prototype_size={size}):')
    source=source.replace("        self.device = torch.device('cuda')",f'        self.prototype_size = prototype_size\n        self.context_self_prediction = {name=="cow.py"}')
    source=source.replace('MLP(256, self.','MLP(prototype_size, self.').replace('MLP(257, self.','MLP(prototype_size, self.')
    source=source.replace('self.criterion = nn.NLLLoss()','self.criterion = nn.NLLLoss(ignore_index=255)')
    source=source.replace('zero = torch.zeros(1).squeeze(0)','zero = sup_fg_pts.new_zeros(())')
    ast.parse(source)
    path.write_text(source,encoding='utf-8')
    print('Revised',path.relative_to(ROOT))
