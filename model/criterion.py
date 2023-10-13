import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.measure import label

from utils import calc_topk_accuracy


def norm(t):
    return F.normalize(t, dim=-1, eps=1e-10)


def query_ce_loss(gc_query, lc_query, num_queries, temperature=1):
    B = gc_query.size(0)
    N = 2 * num_queries
    criterion = nn.CrossEntropyLoss()
    mask = mask_correlated_samples(num_queries, gc_query.device)

    # calculate ce loss for each query set in B
    loss = top1_avg = 0
    labels = torch.zeros(N, device=gc_query.device).long()
    for i in range(B):
        z = torch.cat((gc_query[i], lc_query[i]), dim=0)

        sim = torch.matmul(z, z.T) / temperature
        sim_gc_lc = torch.diag(sim, num_queries)
        sim_lc_gc = torch.diag(sim, -num_queries)

        positive_samples = torch.cat((sim_gc_lc, sim_lc_gc), dim=0).reshape(N, 1)
        negative_samples = sim[mask].reshape(N, -1)

        logits = torch.cat((positive_samples, negative_samples), dim=1)
        ce_loss = criterion(logits, labels)

        top1 = calc_topk_accuracy(logits, labels, (1,))

        loss += ce_loss
        top1_avg += top1[0]

    return loss / B, top1_avg / B


def mask_correlated_samples(num_seq, device):
    N = 2 * num_seq
    mask = torch.ones((N, N), device=device)
    mask = mask.fill_diagonal_(0)
    for i in range(num_seq):
        mask[i, num_seq + i] = 0
        mask[num_seq + i, i] = 0
    mask = mask.bool()
    return mask


class AlignCriterion(nn.Module):
    def __init__(self, patch_size=16,
                 num_queries=5,
                 nmb_crops=(1, 1),
                 roi_align_kernel_size=7,
                 ce_temperature=1,
                 negative_pressure=0.1,
                 last_self_attention=True):
        super(AlignCriterion, self).__init__()
        self.patch_size = patch_size
        self.num_queries = num_queries
        self.nmb_crops = nmb_crops
        self.roi_align_kernel_size = roi_align_kernel_size
        self.ce_temperature = ce_temperature
        self.negative_pressure = negative_pressure
        self.last_self_attention = last_self_attention

    def forward(self, results, bboxes):
        all_queries, gc_output, lc_output, attn_hard, gc_spatial_res, lc_spatial_res = results
        B = gc_output.size(0)

        # prepare foreground mask
        mask = attn_hard.reshape(B*sum(self.nmb_crops), -1)
        mask = mask.int()
        mask_gc, masks_lc = mask[:B * self.nmb_crops[0]], mask[B * self.nmb_crops[0]:]

        loss = 0
        '''
        1. Compute patch correlation to assignment similarity alignment loss
            -- Compute similarity between Queries and spatial_tokens, and align to patch correlation
            -- use attention map as foreground hint to mask correlation matrix
            -- assuming there is ONLY 1 global crop
        '''
        # compute patch correlation between gc and lc, use as assignment target later
        with torch.no_grad():
            gclc_correlations = []
            masks_gc_lc = []
            mask_gc = mask_gc.repeat(1, lc_spatial_res**2).reshape(B, lc_spatial_res**2, -1)
            mask_gc = mask_gc.transpose(1, 2)      # (B,n,m)
            for i in range(self.nmb_crops[-1]):
                # compute cosine similarity
                correlation = torch.einsum("bnc,bmc->bnm", norm(gc_output), norm(lc_output[:, i]))
                # spatial centering for better recognizing small objects
                old_mean = correlation.mean()
                correlation -= correlation.mean(dim=-1, keepdim=True)
                correlation = correlation - correlation.mean() + old_mean
                gclc_correlations.append(correlation)

                # compute gc-lc foreground intersection mask
                mask_lc_ = masks_lc[i*B:(i+1)*B]    # (B, m)
                mask_lc_ = mask_lc_.repeat(1, gc_spatial_res**2).reshape(B, gc_spatial_res**2, -1)    # (B,n,m)
                mask_gc_lc_ = mask_gc * mask_lc_
                masks_gc_lc.append(mask_gc_lc_.bool())

        # compute gc and lc token assignment
        gc_token_assign = torch.einsum("bnc,bqc->bnq", norm(gc_output), norm(all_queries[0]))

        gclc_cor_loss = 0
        lc_assigns_detached = []
        for i in range(self.nmb_crops[-1]):
            lc_token_assign = torch.einsum("bmc,bqc->bmq", norm(lc_output[:, i]), norm(all_queries[i+1]))
            # store lc intersection assignment
            lc_tmp = torch.clone(lc_token_assign.detach())
            lc_tmp = lc_tmp.reshape(B, lc_spatial_res, lc_spatial_res, -1).permute(0, 3, 1, 2)  # (B, num_queries, 6, 6)
            lc_assigns_detached.append(lc_tmp)

            # note here correlation value is not cosine similarity
            gc_token_assign_ = gc_token_assign.clamp(min=0.0)
            lc_token_assign_ = lc_token_assign.clamp(min=0.0)
            gclc_assign_cor = torch.einsum("bnq,bmq->bnm", gc_token_assign_.softmax(dim=-1), lc_token_assign_.softmax(dim=-1))
            # align patch assignment similarity to feature correlation
            cor_align_loss = (- gclc_assign_cor * (gclc_correlations[i] - self.negative_pressure))[masks_gc_lc[i]]
            gclc_cor_loss += 0.15*cor_align_loss.sum()

        loss += gclc_cor_loss / self.nmb_crops[-1]

        '''
        2. Compute Global-Local Query Alignment loss
            -- use cross-entropy loss to align queries, and make each query different
        '''
        query_align_loss = 0
        for i in range(self.nmb_crops[-1]):
            tmp_loss, top1 = query_ce_loss(norm(all_queries[0]), norm(all_queries[i + 1]), self.num_queries,
                                           self.ce_temperature)
            query_align_loss += tmp_loss

        loss += query_align_loss / self.nmb_crops[-1]

        return loss
