# Copyright 2022 Digital Brain Laboratory
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of DB1-transformerXL based model
Adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/transfo_xl/modeling_transfo_xl.py"""

from typing import List, Optional
import torch
import torch.nn as nn
from src.data.input_specs import (
    GatoInputBase,
    NLPTaskInput,
    RLTaskInput,
    ICTaskInput,
    VQATaskInput,
)

from src.model.utils import ACT2FN
from src.tokenizer.vision_embedding import VisionEmbedding
import torch.nn.functional as F
from src.mpu import print_with_rank


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super().__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[None, :, :].expand(bsz, -1, -1)
        else:
            return pos_emb[None, :, :]


class RelPartialLearnableMultiHeadAttn(nn.Module):
    def __init__(
        self,
        n_head,
        d_model,
        d_head,
        dropout,
        dropatt=0,
        pre_lnorm=False,
        r_r_bias=None,
        r_w_bias=None,
        layer_norm_epsilon=1e-5,
    ):
        super().__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        assert self.d_head * self.n_head == self.d_model, (
            self.d_head,
            self.n_head,
            self.d_model,
        )

        self.qkv_net = nn.Linear(
            self.d_model, 3 * self.n_head * self.d_head, bias=False
        )

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(self.n_head * self.d_head, self.d_model, bias=False)

        self.scale = 1 / (self.d_head ** 0.5)

        if r_r_bias is None or r_w_bias is None:  # Biases are not shared
            self.r_r_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
            self.r_w_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
        else:
            self.r_r_bias = r_r_bias
            self.r_w_bias = r_w_bias

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)
        self.layer_norm = nn.LayerNorm(self.d_model, eps=layer_norm_epsilon)
        self.pre_lnorm = pre_lnorm

    def _rel_shift(self, x):  # qlen x klen x bsz x n_head
        # bsz x qlen x klen x n_head
        bsz, qlen, klen, n_head = x.size()
        zero_pad_shape = (bsz, qlen, 1, n_head)  # bsz x qlen x 1 x n_head
        zero_pad = torch.zeros(zero_pad_shape, device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=2)  # bsz x qlen x klen + 1 x n_head

        x_padded_shape = (bsz, klen + 1, qlen, n_head)
        x_padded = x_padded.view(*x_padded_shape)  # bsz x klen+1 x qlen x n_head

        x = x_padded[:, 1:].view_as(x)  # bsz x qlen x klen x n_head

        return x

    def forward(
        self,
        w,
        r,
        mem=None,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        deepnorm_alpha: Optional[float] = None,
    ):
        qlen, rlen, bsz = w.size(1), r.size(1), w.size(0)

        if mem is not None:
            cat = torch.cat([mem, w], 1)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[:, -qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(1)

        w_head_q = w_head_q.view(
            bsz, qlen, self.n_head, self.d_head
        )  # bsz x qlen x n_head x d_head
        w_head_k = w_head_k.view(
            bsz, klen, self.n_head, self.d_head
        )  # bsz x klen x n_head x d_head
        w_head_v = w_head_v.view(
            bsz, klen, self.n_head, self.d_head
        )  # bsz x klen x n_head x d_head

        # original shape:  1 x rlen x d_model
        r_head_k = r_head_k.view(
            rlen, self.n_head, self.d_head
        )  # rlen x n_head x d_head

        # compute attention score
        rw_head_q = w_head_q + self.r_w_bias  # bsz x qlen x n_head x d_head

        AC = torch.einsum(
            "bind,bjnd->bijn", (rw_head_q.float(), w_head_k.float())
        )  # bsz x qlen x klen x n_head

        rr_head_q = w_head_q + self.r_r_bias
        BD = torch.einsum(
            "bind,jnd->bijn", (rr_head_q.float(), r_head_k.float())
        )  # bsz x qlen x klen x n_head
        BD = self._rel_shift(BD)
        # [bsz x qlen x klen x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        # compute attention probability
        if attention_mask is not None and torch.sum(attention_mask).item():
            attention_mask = attention_mask == 1  # Switch to bool
            if attention_mask.dim() == 2:
                if next(self.parameters()).dtype == torch.float16:
                    attn_score = (
                        attn_score.float()
                        .masked_fill(attention_mask[None, :, :, None], -1e30)
                        .type_as(attn_score)
                    )
                else:
                    attn_score = (
                        attn_score.float()
                        .masked_fill(attention_mask[None, :, :, None], -1e30)
                        .type_as(attn_score)
                    )
            elif attention_mask.dim() == 3:
                if next(self.parameters()).dtype == torch.float16:
                    attn_score = (
                        attn_score.float()
                        .masked_fill(attention_mask[:, :, :, None], -1e30)
                        .type_as(attn_score)
                    )
                else:
                    attn_score = (
                        attn_score.float()
                        .masked_fill(attention_mask[:, :, :, None], -1e30)
                        .type_as(attn_score)
                    )
        else:
            raise ValueError

        # [bsz x qlen x klen x n_head]
        attn_prob = nn.functional.softmax(attn_score, dim=2)
        attn_prob = self.dropatt(attn_prob)

        # Mask heads if we want to
        if head_mask is not None:
            attn_prob = attn_prob * head_mask

        if next(self.parameters()).dtype == torch.float16:
            attn_prob = attn_prob.half()

        # compute attention vector
        attn_vec = torch.einsum("bijn,bjnd->bind", (attn_prob, w_head_v))

        # [bsz x qlen x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head
        )

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            outputs = (w + attn_out,)
        else:
            if deepnorm_alpha is None:
                deepnorm_alpha = 1.
            # residual connection + layer normalization
            outputs = (self.layer_norm(w * deepnorm_alpha + attn_out),)

        if output_attentions:
            outputs = outputs + (attn_prob,)

        return outputs


class PositionwiseFF(nn.Module):
    def __init__(
        self,
        d_model,
        d_inner,
        dropout,
        activation,
        pre_lnorm=False,
        layer_norm_epsilon=1e-5,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout
        if activation == "geglu":
            assert d_inner % 2 == 0

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner),
            ACT2FN[activation](),
            # nn.Dropout(dropout), # XXX: in next version we may remove the extra Dropout Module but now current released DB1 have this regularization.
            nn.Linear(d_inner if activation != "geglu" else d_inner // 2, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_epsilon)

        self.pre_lnorm = pre_lnorm

    def forward(self, inp, deepnorm_alpha: Optional[float] = None,):
        if self.pre_lnorm:
            # layer normalization + positionwise feed-forward
            core_out = self.CoreNet(self.layer_norm(inp))

            # residual connection
            output = core_out + inp
        else:
            # positionwise feed-forward
            core_out = self.CoreNet(inp)

            # residual connection + layer normalization
            if deepnorm_alpha is None:
                deepnorm_alpha = 1.
            output = self.layer_norm(inp * deepnorm_alpha + core_out)

        return output


class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(
        self,
        n_head,
        d_model,
        d_head,
        d_inner,
        dropout,
        activation,
        layer_norm_epsilon=1e-5,
        **kwargs
    ):
        super().__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(
            n_head,
            d_model,
            d_head,
            dropout,
            layer_norm_epsilon=layer_norm_epsilon,
            **kwargs
        )
        self.pos_ff = PositionwiseFF(
            d_model,
            d_inner,
            dropout,
            activation=activation,
            pre_lnorm=kwargs.get("pre_lnorm"),
            layer_norm_epsilon=layer_norm_epsilon,
        )

    def forward(
        self,
        dec_inp,
        r,
        attention_mask=None,
        mems=None,
        head_mask=None,
        output_attentions=False,
        deepnorm_alpha: Optional[float]=None
    ):

        attn_outputs = self.dec_attn(
            dec_inp,
            r,
            attention_mask=attention_mask,
            mem=mems,
            head_mask=head_mask,
            deepnorm_alpha=deepnorm_alpha,
            output_attentions=output_attentions,
        )
        ff_output = self.pos_ff(
            attn_outputs[0], 
            deepnorm_alpha=deepnorm_alpha
        )

        outputs = (ff_output,) + attn_outputs[1:]

        return outputs


class TransformerXL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embed = config.n_embed
        self.n_position = config.n_position
        self.n_layer = config.n_layer
        self.n_head = config.n_head
        self.d_head = self.n_embed // self.n_head
        self.d_model = self.n_embed
        if config.n_inner is None:
            self.d_inner = 4 * self.d_model
        else:
            self.d_inner = config.n_inner

        self.pre_lnorm = config.pre_lnorm

        self.mem_len = config.mem_len if config.mem_len is not None else 0
        self.same_length = config.same_length
        self.clamp_len = self.n_position
        self.untie_r = config.untie_r

        ##### Build embedding ######
        self.text_vocab_size = config.text_vocab_size
        self.discrete_vocab_size = config.num_discrete_values
        self.continuous_vocab_size = config.num_continuous_bin
        self.discrete_overlap_with_text = config.overlap_with_text
        total_vocab_size = (
            self.text_vocab_size
            + self.continuous_vocab_size
            + (0 if self.discrete_overlap_with_text else self.discrete_vocab_size)
        )

        # include a additional token as rl seperator '|' in Gato.
        self.total_vocab_size = total_vocab_size + 1
        self.rl_separator_token_id = total_vocab_size
        del total_vocab_size

        
        self.word_embedding = nn.Embedding(self.total_vocab_size, self.n_embed)
        # self.word_positional_embedding = nn.Embedding(self.n_position, self.n_embed)
        self.pos_emb = PositionalEmbedding(
            self.n_embed
        )  # use relative positional encoding, fixed
        if not self.untie_r:
            self.r_w_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
            self.r_r_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))

        self.vision_encoder = VisionEmbedding(config)
        self.ic_encoder = self.vision_encoder

        # positional encoding for RL timesteps, 0 is used for unique action embedding
        self.rl_local_timestep_embedding = nn.Embedding(512 + 1, self.n_embed)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList(
            [
                RelPartialLearnableDecoderLayer(
                    self.n_head,
                    self.d_model,
                    self.d_head,
                    self.d_inner,
                    config.drop,
                    dropatt=config.dropattn,
                    activation=config.activation_fn,
                    pre_lnorm=self.pre_lnorm,
                    r_w_bias=None if self.untie_r else self.r_w_bias,
                    r_r_bias=None if self.untie_r else self.r_r_bias,
                    layer_norm_epsilon=config.layer_norm_epsilon,
                )
                for i in range(config.n_layer)
            ]
        )
        # self.ln_f = nn.LayerNorm(self.n_embed, eps=config.layer_norm_epsilon)
        self.share_input_output_embedding = config.share_input_output_embedding
        if config.share_input_output_embedding:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.n_embed, self.total_vocab_size, bias=False)

        self.apply(self._init_weights)

        self.use_deepnorm = config.use_deepnorm
        self.deepnorm_alpha = (2 * self.n_layer) ** 0.25 if self.use_deepnorm else None
        self.deepnorm_beta = (8 * self.n_layer) ** -0.25 if self.use_deepnorm else None
        # reinit weights required by deepnorm
        if self.use_deepnorm:
            self._deepnorm_init()
        
    def _deepnorm_init(self):
        if self.use_deepnorm:
            for name, module in self.named_modules():
                if "pos_ff" in name:
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight, gain=self.deepnorm_beta)
                elif "o_net" in name:
                    nn.init.xavier_uniform_(module.weight, gain=self.deepnorm_beta)
                elif "qkv_net" in name:
                    nn.init.xavier_uniform_(module.weight, gain=1)
                    nn.init.xavier_uniform_(module.weight[2 * self.d_model:, :], gain=self.deepnorm_beta)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        else:
            if hasattr(module, "r_r_bias"):
                module.r_r_bias.data.normal_(mean=0.0, std=0.02)
            if hasattr(module, "r_w_bias"):
                module.r_w_bias.data.normal_(mean=0.0, std=0.02)

    def init_mem(self, batch_size):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for i in range(self.n_layer):
                empty = torch.zeros(
                    batch_size,
                    self.mem_len,
                    self.n_embed,
                    dtype=param.dtype,
                    device=param.device,
                )
                mems.append(empty)
            return mems
        else:
            return None

    def _update_mem(self, hiddens, mems, mlen, qlen):
        # does not deal with None
        if mems is None:
            return None

        # mems is not None
        assert len(hiddens) == len(mems), "len(hids) != len(mems)"

        # There are `mlen + qlen` steps that can be cached into mems
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hiddens)):
                cat = torch.cat([mems[i], hiddens[i]], dim=1)
                new_mems.append(cat[:, beg_idx:end_idx])

        return new_mems

    def forward(
        self, tasks_input: List[GatoInputBase], compute_loss: bool = True, mems=None
    ):
        batch_embeddings, batch_loss_maks, batch_attention_masks, batch_labels = (
            [],
            [],
            [],
            [],
        )
        assert not (
            compute_loss and mems is not None
        ), "During training, Gato does not use memory mechanism."

        for single_task in tasks_input:
            if isinstance(single_task, RLTaskInput):
                forward_fn = self._forward_rl
            elif isinstance(single_task, NLPTaskInput):
                forward_fn = self._forward_nlp
            elif isinstance(single_task, ICTaskInput):
                forward_fn = self._forward_ic
            elif isinstance(single_task, VQATaskInput):
                forward_fn = self._forward_vqa

            (
                single_embedding,
                single_loss_mask,
                single_attn_mask,
                single_label,
            ) = forward_fn(single_task)

            batch_embeddings.append(single_embedding)
            batch_loss_maks.append(single_loss_mask)
            # batch_attention_masks.append(single_attn_mask)
            batch_labels.append(single_label)

        loss_masks = torch.cat(batch_loss_maks, dim=0) if compute_loss else None
        # attn_masks = torch.cat(batch_attention_masks, dim=0)
        labels = torch.cat(batch_labels, dim=0).long() if compute_loss else None
        hidden_states = torch.cat(batch_embeddings, dim=0)
        hidden_states = self.drop(hidden_states)

        qlen = hidden_states.size(1)
        mlen = mems[0].size(1) if mems is not None else 0
        klen = mlen + qlen

        if self.same_length:
            all_ones = hidden_states.new_ones((qlen, klen), dtype=torch.uint8)
            mask_len = klen - self.mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            attention_mask = (
                torch.triu(all_ones, 1 + mlen) + torch.tril(all_ones, -mask_shift_len)
            )[
                None, :, :
            ]  # -1
        else:
            attention_mask = torch.triu(
                hidden_states.new_ones((qlen, klen), dtype=torch.uint8),
                diagonal=1 + mlen,
            )[None, :, :]

        pos_seq = torch.arange(
            klen - 1, -1, -1.0, device=hidden_states.device, dtype=hidden_states.dtype
        )
        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)
        pos_emb = self.pos_emb(pos_seq)
        pos_emb = self.drop(pos_emb)

        hids = []
        for i, block in enumerate(self.h):
            hids.append(hidden_states)
            mems_i = None if mems is None else mems[i]
            outputs = block(
                hidden_states,
                pos_emb,
                mems=mems_i,
                attention_mask=attention_mask,
                head_mask=None,
                output_attentions=False,
                deepnorm_alpha=self.deepnorm_alpha
            )

            hidden_states = outputs[0]
        # hidden_states = self.ln_f(hidden_states)
        if self.share_input_output_embedding:
            # bsz x L x D, V x D
            lm_logits = F.linear(hidden_states, self.word_embedding.weight)
            assert lm_logits.shape[:-1] == hidden_states.shape[:-1]
        else:
            lm_logits = self.lm_head(hidden_states)

        new_mems = self._update_mem(hids, mems, mlen, qlen)

        if compute_loss:
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            loss_masks = loss_masks.float()
            loss = loss.float()
            # in order to make avoid exceeding of float16
            loss = (loss * loss_masks.view(-1)).sum() / loss_masks.sum()
            if loss.isnan() or loss.isinf():
                print_with_rank("WARNING: Loss Overflow.")
        else:
            loss = None

        res = (lm_logits, loss)
        if mems is not None:
            res = res + (new_mems,)

        return res

    def _forward_rl(self, rl_input: RLTaskInput):
        input_tensor = rl_input.tensor_seq
        bsz, text_len = input_tensor.size()
        label = rl_input.label

        # [bsz, seq_len, n_embed]
        valid_word_emb = self.word_embedding(input_tensor[input_tensor >= 0])
        word_emb = valid_word_emb.new_zeros(bsz, text_len, self.n_embed)
        word_emb[input_tensor >= 0] = valid_word_emb
        if rl_input.vision_seq is not None:
            input_img = rl_input.vision_seq
            vis_emb = self.vision_encoder(input_img.view(-1, *input_img.shape[-3:]))
            vis_emb_flat = vis_emb.reshape(bsz, -1, self.n_embed)

            # def substitute_values(arr, flags, value):
            #     arr[flags==-1] = value[:(flags==-1).sum()]
            # for i in range(bsz):
            #     substitute_values(word_emb[i], input_tensor[i], vis_emb_flat[i])
            l = (input_tensor == -1).sum(1)[0]
            word_emb[input_tensor == -1] = (
                vis_emb_flat[:, :l].contiguous().view(-1, self.n_embed)
            )

            if label is not None:
                label[label == -1] = 0
        local_positional_embedding = self.rl_local_timestep_embedding(
            rl_input.position_id
        )
        rl_hidden_states = word_emb + local_positional_embedding
        # pos_emb = self.word_positional_embedding(
        #     torch.arange(input_tensor.size(1), device=input_tensor.device)
        # )
        # rl_hidden_states = rl_hidden_states + pos_emb

        return (
            rl_hidden_states,
            rl_input.loss_mask,
            rl_input.attention_mask,
            label,
        )

    def _forward_nlp(self, nlp_input: NLPTaskInput):
        input_ids = nlp_input.text_seq
        # bsz, text_len = input_ids.size()
        text_emb = self.word_embedding(input_ids)

        return (
            text_emb,
            nlp_input.loss_mask,
            nlp_input.attention_mask,
            nlp_input.label,
        )

    def _forward_ic(self, ic_input: ICTaskInput):
        input_encodings = []
        # prompt
        prompt_emb = self.word_embedding(ic_input.prompt_seq)

        # image part
        image_raw = ic_input.img_seq
        # print("image_raw", image_raw.shape)
        vis_emb = self.ic_encoder(image_raw)
        # bs = vis_emb.shape[0]
        # seq_length = vis_emb.shape[1]

        text_emb = self.word_embedding(ic_input.text_seq)
        # text_pos_id = ic_input.position_id
        # text_pos_emb = self.word_positional_embedding(text_pos_id)

        # text_emb = text_emb + text_pos_emb
        input_encodings.append(prompt_emb)
        input_encodings.append(vis_emb)
        input_encodings.append(text_emb)

        input_encodings = torch.cat(input_encodings, dim=1)
        # input_encodings = input_encodings + text_pos_emb

        return (
            input_encodings,
            ic_input.loss_mask,
            ic_input.attention_mask,
            ic_input.label,
        )

    def _forward_vqa(self, vqa_input: VQATaskInput):

        input_encodings = []
        # prompt
        prompt_emb = self.word_embedding(vqa_input.prompt_seq)
        # print("prompt_emb", vqa_input.prompt_seq.shape)

        # image part
        image_raw = vqa_input.img_seq

        vis_emb = self.ic_encoder(image_raw)
        # vis_emb = torch.zeros((image_raw.shape[0], 324, 768), device=prompt_emb.device, dtype=image_raw.dtype)
        before = vis_emb.shape[1] + vqa_input.prompt_seq.shape[1]

        before = vqa_input.ques_len[0] + before
        # print(f"ques: {vqa_input.text_seq[0, vqa_input.ques_len[0]-3: vqa_input.ques_len[0]+23]}")
        # print(f"label {vqa_input.label[0, before-3: before+23]}, loss {vqa_input.loss_mask[0, before-3: before+23]}")
        # print(f"loss sum: {torch.sum(vqa_input.loss_mask, dim=1)}")

        text_emb = self.word_embedding(vqa_input.text_seq)
        # print("text_seq", vqa_input.text_seq.shape)
        text_pos_id = vqa_input.position_id

        input_encodings.append(prompt_emb)
        input_encodings.append(vis_emb)
        input_encodings.append(text_emb)

        input_encodings = torch.cat(input_encodings, dim=1)

        # text_pos_id = vqa_input.position_id
        # if text_pos_id is None:
        #     text_pos_id = torch.arange(
        #             input_encodings.size(1), dtype=torch.long,
        #             device=input_encodings.device
        #         )
        # text_pos_emb = self.word_positional_embedding(text_pos_id)
        # input_encodings = input_encodings + text_pos_emb

        return (
            input_encodings,
            vqa_input.loss_mask,
            vqa_input.attention_mask,
            vqa_input.label,
        )
