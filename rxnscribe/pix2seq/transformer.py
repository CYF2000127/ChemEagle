# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Pix2Seq Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .attention_layer import Attention

from transformers import EncoderDecoderConfig, EncoderDecoderModel, AutoConfig, BertConfig


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", normalize_before=False, num_vocal=2094,
                 pred_eos=False, tokenizer=None):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        self._reset_parameters()

        self.num_vocal = num_vocal
        self.vocal_classifier = nn.Linear(d_model, num_vocal)
        self.det_embed = nn.Embedding(1, d_model)
        self.vocal_embed = nn.Embedding(self.num_vocal - 2, d_model)
        self.pred_eos = pred_eos

        self.d_model = d_model
        self.nhead = nhead
        self.num_decoder_layers = num_decoder_layers
        self.tokenizer = tokenizer

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, input_seq, mask, pos_embed, max_len=500):
        """
        Args:
            src: shape[B, C, H, W]
            input_seq: shape[B, 501, C] for training and shape[B, 1, C] for inference
            mask: shape[B, H, W]
            pos_embed: shape[B, C, H, W]
        """
        # flatten NxCxHxW to HWxNxC
        bs = src.shape[0]
        src = src.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        pre_kv = [torch.as_tensor([[], []], device=memory.device)
                  for _ in range(self.num_decoder_layers)]

        if self.training:
            input_seq = input_seq.clamp(max=self.num_vocal - 3)
            input_embed = torch.cat(
                [self.det_embed.weight.unsqueeze(0).repeat(bs, 1, 1),
                 self.vocal_embed(input_seq)], dim=1)
            input_embed = input_embed.transpose(0, 1)
            num_seq = input_embed.shape[0]
            self_attn_mask = torch.triu(torch.ones((num_seq, num_seq)), diagonal=1).bool().to(input_embed.device)
            hs, pre_kv = self.decoder(
                input_embed,
                memory,
                memory_key_padding_mask=mask,
                pos=pos_embed,
                pre_kv_list=pre_kv,
                self_attn_mask=self_attn_mask)
            # hs: N x B x D
            pred_seq_logits = self.vocal_classifier(hs.transpose(0, 1))
            return pred_seq_logits
        else:
            end = torch.zeros(bs).bool().to(memory.device)
            end_lens = torch.zeros(bs).long().to(memory.device)
            input_embed = self.det_embed.weight.unsqueeze(0).repeat(bs, 1, 1).transpose(0, 1)
            states, pred_token = [None] * bs, [None] * bs
            pred_seq, pred_scores = [], []
            for seq_i in range(max_len):
                hs, pre_kv = self.decoder(
                    input_embed,
                    memory,
                    memory_key_padding_mask=mask,
                    pos=pos_embed,
                    pre_kv_list=pre_kv)
                # hs: N x B x D
                logits = self.vocal_classifier(hs.transpose(0, 1))
                log_probs = F.log_softmax(logits, dim=-1)
                if self.tokenizer.output_constraint:
                    states, output_masks = self.tokenizer.update_states_and_masks(states, pred_token)
                    output_masks = torch.tensor(output_masks, device=logits.device).unsqueeze(1)
                    log_probs.masked_fill_(output_masks, -10000)
                if not self.pred_eos:
                    log_probs[:, :, self.tokenizer.EOS_ID] = -10000

                score, pred_token = log_probs.max(dim=-1)
                pred_seq.append(pred_token)

                pred_scores.append(score)

                if self.pred_eos:
                    stop_state = pred_token.squeeze(1).eq(self.tokenizer.EOS_ID)
                    end_lens += seq_i * (~end * stop_state)
                    end = (stop_state + end).bool()
                    if end.all() and seq_i > 4:
                        break

                token = log_probs[:, :, :self.num_vocal - 2].argmax(dim=-1)
                input_embed = self.vocal_embed(token.transpose(0, 1))

            if not self.pred_eos:
                end_lens = end_lens.fill_(max_len)
            pred_seq = torch.cat(pred_seq, dim=1)
            pred_seq = [seq[:end_idx] for end_idx, seq in zip(end_lens, pred_seq)]
            pred_scores = torch.cat(pred_scores, dim=1)
            pred_scores = [scores[:end_idx] for end_idx, scores in zip(end_lens, pred_scores)]
            return pred_seq, pred_scores


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, memory_key_padding_mask, pos, pre_kv_list=None, self_attn_mask=None):
        output = tgt
        cur_kv_list = []
        for layer, pre_kv in zip(self.layers, pre_kv_list):
            output, cur_kv = layer(
                output,
                memory,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                self_attn_mask=self_attn_mask,
                pre_kv=pre_kv)
            cur_kv_list.append(cur_kv)

        if self.norm is not None:
            output = self.norm(output)

        return output, cur_kv_list


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_key_padding_mask, pos)
        return self.forward_post(src, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = Attention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
            self,
            tgt,
            memory,
            memory_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
            self_attn_mask: Optional[Tensor] = None,
            pre_kv=None,
    ):
        tgt2, pre_kv = self.self_attn(tgt, pre_kv=pre_kv, attn_mask=self_attn_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(
            query=tgt,
            key=self.with_pos_embed(memory, pos),
            value=memory,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, pre_kv

    def forward_pre(
            self,
            tgt,
            memory,
            memory_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
            self_attn_mask: Optional[Tensor] = None,
            pre_kv=None,
    ):
        tgt2 = self.norm1(tgt)
        tgt2, pre_kv = self.self_attn(tgt2, pre_kv=pre_kv, attn_mask=self_attn_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=tgt2,
            key=self.with_pos_embed(memory, pos),
            value=memory,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt, pre_kv

    def forward(
            self,
            tgt,
            memory,
            memory_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
            self_attn_mask: Optional[Tensor] = None,
            pre_kv=None,
    ):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_key_padding_mask, pos, self_attn_mask, pre_kv)
        return self.forward_post(tgt, memory, memory_key_padding_mask, pos, self_attn_mask, pre_kv)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args, tokenizer):
    if args.use_hf_transformer:
        num_vocal = len(tokenizer)
        encoder_config = BertConfig(max_position_embeddings = 1764, hidden_size = 256, num_attention_heads = 4, vocab_size = num_vocal, num_hidden_layers = 4, intermediate_size = 1024)
        decoder_config = BertConfig(max_position_embeddings = 1764, hidden_size = 256, num_attention_heads = 4, vocab_size = num_vocal, is_decoder = True, num_hidden_layers = 4, intermediate_size = 1024)
        config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config, add_pooling_layer = False, decoder_add_pooling_layer = False)

        model = EncoderDecoderModel(config=config)
        model.config.vocab_size = num_vocal
        model.config.decoder_start_token_id = tokenizer.SOS_ID
        model.config.pad_token_id = tokenizer.PAD_ID
        model.config.eos_token_id = tokenizer.EOS_ID
        model.encoder.embeddings.word_embeddings = None
        model.encoder.pooler = None
        return model
    else:
        num_vocal = len(tokenizer)
        return Transformer(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.enc_layers,
            num_decoder_layers=args.dec_layers,
            normalize_before=args.pre_norm,
            num_vocal=num_vocal,
            pred_eos=args.pred_eos,
            tokenizer=tokenizer
        )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
