import torch
import torch.nn as nn
import torch.nn.functional as F

from model.vit import vit_small, vit_base, vit_large, trunc_normal_
from utils import process_attentions


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                pos=None,
                query_pos=None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos), value=memory)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, query_pos=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=1024, dropout=0.0):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.activation = F.relu

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.gelu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class AlignSegmentor(nn.Module):

    def __init__(self, arch='vit_small',
                 patch_size=16,
                 embed_dim=384,
                 hidden_dim=384,
                 num_heads=4,
                 num_queries=21,
                 nmb_crops=(1, 0),
                 num_decode_layers=1,
                 last_self_attention=True):
        super(AlignSegmentor, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.nmb_crops = nmb_crops
        self.num_decode_layers = num_decode_layers
        self.last_self_attention = last_self_attention

        # Initialize model
        if arch == 'vit_small':
            self.backbone = vit_small(patch_size=patch_size)
        elif arch == 'vit_base':
            self.backbone = vit_base(patch_size=patch_size)
        elif arch == 'vit_large':
            self.backbone = vit_large(patch_size=patch_size)
        else:
            raise ValueError(f"{self.arch} is not supported")

        # learnable CLS queries and/or positional queries
        self.clsQueries = nn.Embedding(num_queries, embed_dim)

        # simple Transformer Decoder with num_decoder_layers
        self.decoder_cross_attention_layers = nn.ModuleList()
        self.decoder_self_attention_layers = nn.ModuleList()
        self.decoder_ffn_layers = nn.ModuleList()
        for _ in range(self.num_decode_layers):
            self.decoder_cross_attention_layers.append(
                CrossAttentionLayer(d_model=embed_dim, nhead=num_heads)
            )
            self.decoder_self_attention_layers.append(
                SelfAttentionLayer(d_model=embed_dim, nhead=num_heads)
            )
            self.decoder_ffn_layers.append(
                FFNLayer(d_model=embed_dim, dim_feedforward=hidden_dim)
            )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def set_clsQuery(self, prototypes):
        # initialize clsQueries with generated prototypes of [num_queries, embed_dim]
        self.clsQueries.weight = nn.Parameter(prototypes)

    def forward(self, inputs, threshold=0.6):
        # inputs is a list of crop images
        B = inputs[0].size(0)

        # repeat query for batch use, (B, num_queries, embed_dim)
        outQueries = self.clsQueries.weight.unsqueeze(0).repeat(B, 1, 1)
        posQueries = pos = None

        # Extract feature
        outputs = self.backbone(inputs, self.nmb_crops, self.last_self_attention)
        if self.last_self_attention:
            outputs, attentions = outputs  # outputs=[B*N(196+36), embed_dim], attentions(only global)=[B, heads, 196]

        # calculate gc and lc resolutions. Split output in gc and lc embeddings
        gc_res_w = inputs[0].size(2) / self.patch_size
        gc_res_h = inputs[0].size(3) / self.patch_size
        assert gc_res_w.is_integer() and gc_res_w.is_integer(), "Image dims need to be divisible by patch size"
        assert gc_res_w == gc_res_h, f"Only supporting square images not {inputs[0].size(2)}x{inputs[0].size(3)}"
        gc_spatial_res = int(gc_res_w)
        lc_res_w = inputs[-1].size(2) / self.patch_size
        assert lc_res_w.is_integer(), "Image dims need to be divisible by patch size"
        lc_spatial_res = int(lc_res_w)
        gc_spatial_output, lc_spatial_output = outputs[:B * self.nmb_crops[0] * gc_spatial_res ** 2], \
            outputs[B * self.nmb_crops[0] * gc_spatial_res ** 2:]
        # (B*N, C) -> (B, N, C)
        gc_spatial_output = gc_spatial_output.reshape(B, -1, self.embed_dim)
        if self.nmb_crops[-1] != 0:
            lc_spatial_output = lc_spatial_output.reshape(B, self.nmb_crops[-1], lc_spatial_res**2, self.embed_dim)

        # merge attention heads and threshold attentions
        attn_hard = None
        if self.last_self_attention:
            attn_smooth = sum(attentions[:, i] * 1 / attentions.size(1) for i in range(attentions.size(1)))
            attn_smooth = attn_smooth.reshape(B * sum(self.nmb_crops), 1, gc_spatial_res, gc_spatial_res)
            # attn_hard is later served as 'foreground' hint, use attn_hard.bool()
            attn_hard = process_attentions(attn_smooth, gc_spatial_res, threshold=threshold, blur_sigma=0.6)
            attn_hard = attn_hard.squeeze(1)

        # Align Queries to each image crop's features with decoder, assuming only 1 global crop
        all_queries = []
        for i in range(sum(self.nmb_crops)):
            if i == 0:
                features = gc_spatial_output
            else:
                features = lc_spatial_output[:, i-1]
            for j in range(self.num_decode_layers):
                # attention: cross-attention first
                queries = self.decoder_cross_attention_layers[j](
                    outQueries, features, pos=pos, query_pos=posQueries)
                # self-attention
                queries = self.decoder_self_attention_layers[j](
                    queries, query_pos=posQueries)
                # FFN
                queries = self.decoder_ffn_layers[j](queries)

            all_queries.append(queries)

        return all_queries, gc_spatial_output, lc_spatial_output, attn_hard, gc_spatial_res, lc_spatial_res


if __name__ == '__main__':
    model = AlignSegmentor()