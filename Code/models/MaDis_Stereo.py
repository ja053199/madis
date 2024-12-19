# Modified for MaDis-Stereo
# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import torch
from torch import nn
from .croco import CroCoNet

def croco_args_from_ckpt(ckpt):
    if 'croco_kwargs' in ckpt:  # CroCo v2 released models
        return ckpt['croco_kwargs']
    elif 'args' in ckpt and hasattr(ckpt['args'], 'model'):  # pretrained using the official code release
        s = ckpt['args'].model  # eg "CroCoNet(enc_embed_dim=1024, enc_num_heads=16, enc_depth=24)"
        assert s.startswith('CroCoNet(')
        return eval('dict' + s[len('CroCoNet'):])  # transform it into the string of a dictionary and evaluate it
    else:  # CroCo v1 released models
        return dict()

class MaDis_finetuning(CroCoNet):

    def __init__(self, head, **kwargs):
        super(MaDis_finetuning, self).__init__(**kwargs)
        head.setup(self)
        self.head = head

    def masking_feat(self, feat, mask):
        B, Nenc, C = feat.size()
        if mask is None:
            feat_ = feat
        else:
            Ntotal = mask.size(1)
            func = nn.Parameter(torch.zeros(1, 1, 768))
            feat_ = func.repeat(B, Ntotal, 1).to(dtype=feat.dtype).to('cuda')
            feat_[~mask] = feat.view(B * Nenc, C)
        return feat_

    def forward(self, img1, img2, visfeatL=None, visposL=None, visfeatR=None, visposR=None, mode=None):
        B, C, H, W = img1.size()
        img_info = {'height': H, 'width': W}
        return_all_blocks = hasattr(self.head, 'return_all_blocks') and self.head.return_all_blocks
        if mode == 1:
            out1, pos1, maskL = self._encode_image(img1, do_mask=True, return_all_blocks=return_all_blocks)
            out1_mask = self.masking_feat(out1[out1.__len__() - 1], maskL)
            out2, pos2, maskR = self._encode_image(img2, do_mask=True, return_all_blocks=False)
            out2_mask = self.masking_feat(out2, maskR)

            if return_all_blocks:
                # for reconstructing mask 1
                decout = self._decoder(out1[-1], pos1, maskL, out2_mask, pos2, return_all_blocks=return_all_blocks)
                reconL = self.prediction_head(decout[decout.__len__() - 1])
                # for reconstructing mask 2
                decout2 = self._decoder(out2, pos2, maskR, out1_mask, pos1, return_all_blocks=return_all_blocks)
                reconR = self.prediction_head(decout2[decout2.__len__() - 1])

                decout = out1 + decout

            else:
                decout = self._decoder(out1, pos1, maskL, out2, pos2, return_all_blocks=return_all_blocks)
                reconL = self.prediction_head(decout)

            targetL = self.patchify(img1)
            targetR = self.patchify(img2)

            return self.head(decout, img_info), reconL, maskL, targetL, reconR, maskR, targetR

        if mode == 2:
            out, pos = visfeatL, visposL
            out2, pos2 = visfeatR, visposR
            if return_all_blocks:
                decout = self._decoder(out[-1], pos, None, out2, pos2, return_all_blocks=return_all_blocks)
                decout = out + decout
            else:
                decout = self._decoder(out, pos, None, out2, pos2, return_all_blocks=return_all_blocks)
            return self.head(decout, img_info)

        else:
            out, pos, _  = self._encode_image(img1, do_mask=False, return_all_blocks=return_all_blocks)
            out2, pos2, _ = self._encode_image(img2, do_mask=False, return_all_blocks=False)
            if return_all_blocks:
                decout = self._decoder(out[-1], pos, None, out2, pos2, return_all_blocks=return_all_blocks)
                decout = out + decout
            else:
                decout = self._decoder(out, pos, None, out2, pos2, return_all_blocks=return_all_blocks)

            return self.head(decout, img_info)