from tkinter import Variable
from unittest.mock import NonCallableMagicMock
import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel,GPT2Tokenizer
from typing import Tuple, Optional
from enum import Enum
from swin_transformer import SwinTransformer
import numpy as np
from muge_data.xbert import BertOnlyMLMHead,MaskedLMOutput
from torch.nn import NLLLoss, CrossEntropyLoss
from muge_data.wordpiece import BertTokenizer
from muge_data.data import TextField
from tools.visualize import heatmap,tsne
def position_embedding(input, d_model):
    input = input.view(-1, 1)
    dim = torch.arange(d_model // 2, dtype=torch.float32, device=input.device).view(1, -1)
    sin = torch.sin(input / 10000 ** (2 * dim / d_model))
    cos = torch.cos(input / 10000 ** (2 * dim / d_model))

    out = torch.zeros((input.shape[0], d_model), device=input.device)
    out[:, ::2] = sin
    out[:, 1::2] = cos
    return out


def sinusoid_encoding_table(max_len, d_model, padding_idx=None):
    pos = torch.arange(max_len, dtype=torch.float32)
    out = position_embedding(pos, d_model)

    if padding_idx is not None:
        out[padding_idx] = 0
    return out


def cross_entropy(logits, target, size_average=True):
    if size_average:
        return torch.mean(torch.sum(- target * nnf.log_softmax(logits, -1), -1))
    else:
        return torch.sum(torch.sum(- target * nnf.log_softmax(logits, -1), -1))
 
 
class NpairLoss(nn.Module):
    """the multi-class n-pair loss"""
    def __init__(self, l2_reg=0.02):
        super(NpairLoss, self).__init__()
        self.l2_reg = l2_reg
 
    def forward(self, anchor, positive, target):
        '''  
        anchor and positve are pair data， which are from the same class and target indicate their class
        '''
        batch_size = anchor.size(0)
        target = target.view(target.size(0), 1)
 
        target = (target == torch.transpose(target, 0, 1)).float()
        target = target / torch.sum(target, dim=1, keepdim=True).float()
 
        logit = torch.matmul(anchor, torch.transpose(positive, 0, 1))
        # heatmap(logit.detach().cpu().numpy())
        # tsne(positive.detach().cpu().numpy(),anchor.detach().cpu().numpy())
        loss_ce = cross_entropy(logit, target)
        l2_loss = torch.sum(anchor**2) / batch_size + torch.sum(positive**2) / batch_size
 
        loss = loss_ce + self.l2_reg*l2_loss*0.25
        return loss

class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'
    CONV = 'conv'


class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x.float())

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)
       
class MlpTransformer(nn.Module):
    def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, act=nnf.relu, dropout=0.):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class MultiHeadAttention(nn.Module):

    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        # b n h dh
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        # b m 2 h dh
        keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention


class TransformerLayer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        x_, attention = self.attn(self.norm1(x), y, mask)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, attention

    def forward(self, x, y=None, mask=None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x

    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=nnf.relu,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)


class Transformer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask)
            attentions.append(att)
        return x, attentions

    def forward(self, x, y=None, mask=None):
        intermediate=[]
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec: # cross
                x = layer(x, y)
            elif self.enc_dec:  # self
                x = layer(x, x, mask)
            else:  # self or cross
                x = layer(x, y, mask)
            if self.return_intermediate:
                intermediate.append(self.norm(x))
        if self.norm is not None:
            output = self.norm(x)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        if self.return_intermediate:
            return torch.stack(intermediate)
        return x

    def __init__(self, dim_self: int, num_heads: int, num_layers: int, dim_ref: Optional[int] = None,
                 mlp_ratio: float = 2., act=nnf.relu, norm_layer: nn.Module = nn.LayerNorm, enc_dec: bool = False,return_intermediate=False):
        super(Transformer, self).__init__()
        self.return_intermediate = return_intermediate
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.norm = norm_layer(dim_self)
        # self.norm = nn.Tanh()
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:  # cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            elif enc_dec:  # self
                layers.append(TransformerLayer(dim_self, dim_self, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            else:  # self or cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
        self.layers = nn.ModuleList(layers)


class TransformerMapper(nn.Module):

    def forward(self, x):
        x = self.linear(x)
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        # prefix = self.linear(x)
        
        if self.return_intermediate:
            # (layer_num,batch_size,prefix+clip_legnth,dim)(3,40,100,768)
            out = self.transformer(prefix)[:,:, self.prefix_length:]
            # out = self.transformer(prefix)
        else:
            # out = self.transformer(prefix)
            out = self.transformer(prefix)[:, self.prefix_length:]
        
        # x = self.linear(x)
        # prefix_query = self.prefix_query.unsqueeze(0).expand(x.shape[0], *self.prefix_query.shape)
        # out = self.transformer(prefix_query,x)
        # out = self.transformer(x,prefix_query)
        return out

    def __init__(self, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int, num_layers: int = 3,return_intermediate=False):
        super(TransformerMapper, self).__init__()
        self.prefix_length = prefix_length
        self.clip_length = clip_length
        self.return_intermediate = return_intermediate
        self.transformer = Transformer(dim_embedding, 8, num_layers,return_intermediate=return_intermediate)
        self.linear = nn.Linear(dim_clip,dim_embedding)
        # self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)


class ClipCaptionModel(nn.Module):
    def generate(self,image,sample=False, num_beams=1, max_length=30, min_length=10, top_p=0.9,
                repetition_penalty=1.0, num_return_sequences=1, greedy=False):
        image_fea = self.swin(image) # B,L,1024
        prefix = self.clip_project(image_fea) # B,L,768
        
        if num_beams > 1:
            assert (sample is False) and (num_return_sequences == 1)
            prefix = prefix.repeat_interleave(num_beams, dim=0)

        if num_return_sequences > 1:
            assert (sample is True) and (num_beams == 1)
            prefix = prefix.repeat_interleave(num_return_sequences, dim=0)

        prefix = prefix.to(image.device)

        if greedy:
            # greedy generation from OSCAR
            assert (num_beams == 1) and (num_return_sequences == 1)
            outputs, logprobs = self.generate_no_beam_search(input=prefix, cur_len=0, max_length=max_length,
                                          do_sample=False, temperature=1,
                                          top_k=0, top_p=1, repetition_penalty=repetition_penalty,
                                          eos_token_ids=[self.stop_token_id],
                                          batch_size=prefix.size(0))

            return self._get_captions(outputs)

        elif sample:
            # sampling from OSCAR
            outputs, logprobs = self.generate_no_beam_search(input=prefix, cur_len=0, max_length=max_length,
                                          do_sample=True, temperature=1,
                                          top_k=0, top_p=1, repetition_penalty=repetition_penalty,
                                          eos_token_ids=[self.stop_token_id],
                                          batch_size=prefix.size(0))
            

            # outputs: (bs x num_return_sequences, max_length)
            # logprobs: (bs x num_return_sequences,)

            return self._get_captions(outputs), logprobs

        else:
            # beam search from huggingface
            # seq_lengths = torch.ones(prefix.shape[0],device = prefix.device) 
            # is_stopped = torch.zeros(prefix.shape[0],device= prefix.device, dtype = torch.bool)
            # outputs = self.generate_beam_search(prefix=prefix,
            #                             max_length=max_length,
            #                             seq_lengths = seq_lengths,
            #                             is_stopped = is_stopped,
            #                             num_beams=num_beams,
            #                             eos_token_id=self.stop_token_id,
            #                             repetition_penalty=repetition_penalty,
            #                             )

            # return self._get_captions(outputs)
            outputs = self.generate_test(embed=prefix)
            return outputs
            # return self._get_captions(outputs)

    def generate_no_beam_search(self,input,cur_len,max_length,do_sample,temperature,top_k,top_p,repetition_penalty,
                                eos_token_ids,batch_size):

        """ Generate sequences for each example without beam search (num_beams == 1).
            All returned sequence are generated independantly.
        """
        # current position / max lengths / length of generated sentences / unfinished sentences
        unfinished_sents = []
        cur_unfinished = torch.tensor(input.new(batch_size).fill_(1),dtype=torch.int16)
        
        # log of scores for each sentence in the batch
        logprobs = []
        tokens = torch.tensor([],device=input.device)
        past_key_values=None
        while cur_len < max_length:
            outputs = self.gpt(inputs_embeds = input,past_key_values=past_key_values)
            next_token_logits = outputs.logits[:,-1,:]
            past_key_values = outputs.past_key_values
            # repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for previous_token in set(input[i].tolist()):
                        # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                        if next_token_logits[i, previous_token] < 0:
                            next_token_logits[i, previous_token] *= repetition_penalty
                        else:
                            next_token_logits[i, previous_token] /= repetition_penalty

            if do_sample:
                # Temperature (higher temperature => more likely to sample low probability tokens)
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                # Top-p/top-k filtering
                next_token_logits = self.top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                # Sample
                next_token = torch.multinomial(nnf.softmax(next_token_logits, dim=-1), num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1)

            # Compute scores
            _scores = nnf.log_softmax(next_token_logits, dim=-1)  # (batch_size, vocab_size)
            _scores = torch.gather(_scores, -1, next_token.unsqueeze(-1))  # (batch_size, 1)
            logprobs.append(_scores)  # (batch_size, 1)
            unfinished_sents.append(cur_unfinished)
     
            # update generations and finished sentences
            
            tokens_to_add = next_token * cur_unfinished
            tokens_to_add_embed = self.gpt.transformer.wte(tokens_to_add)
            input = tokens_to_add_embed.unsqueeze(1)
            # input = torch.cat([input, tokens_to_add_embed.unsqueeze(1)], dim=1)
            tokens = torch.cat([tokens, tokens_to_add.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1

            for eos_token_id in eos_token_ids:
                cur_unfinished = cur_unfinished.mul(tokens_to_add.ne(eos_token_id).long())

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if cur_unfinished.max() == 0:
                break

        # add eos_token_ids to unfinished sentences
        if cur_len == max_length:
            tokens[:, -1].masked_fill_(cur_unfinished.to(dtype=torch.bool), eos_token_ids[0])
  
        logprobs = torch.cat(logprobs, dim=1)
        unfinished_sents = torch.stack(unfinished_sents, dim=1).float()
        sum_logprobs = (logprobs * unfinished_sents).sum(dim=1)
        # return logprobs to keep consistent with beam search output
        logprobs = sum_logprobs / unfinished_sents.sum(dim=1)

        
        return tokens, logprobs

  
    def top_k_top_p_filtering(self,logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
        """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
            Args:
                logits: logits distribution shape (batch size, vocabulary size)
                if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
                if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                    Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
                Make sure we keep at least min_tokens_to_keep per batch example in the output
            From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        """
        if top_k > 0:
            top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs > top_p
            if min_tokens_to_keep > 1:
                # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
                sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value
        return logits

    def _get_captions(self,caption_ids):
        captions = []
        for output in caption_ids:
            caption = self.tokenizer.decode(output, skip_special_tokens=True)
            caption = caption.replace('!','')
            captions.append(caption)
            
        return captions

    def generate_test(self,
        beam_size: int = 2,
        embed=None,
        prefix_mask=None,
        entry_length=30,
        temperature=1.0,
        generate_prefix=False,
        stop_token: str = ".",
        
    ):
        result_list = []
        stop_token_index = self.tokenizer.encode(stop_token)[0]
        device = embed.device
        
        for item in range(embed.shape[0]):
            tokens = None
            scores = None
            seq_lengths = torch.ones(beam_size, device=device)
            is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
            if prefix_mask is not None:
                if len(prefix_mask.shape) == 1:
                    prefix_mask = prefix_mask.unsqueeze(0)
                mask = prefix_mask[item]
            else:
                mask = None
            with torch.no_grad():
                if embed is not None:
                    generated = embed[item].unsqueeze(0)
                for i in range(entry_length):
                    if mask is not None:
                        outputs = self.gpt(inputs_embeds=generated,attention_mask=mask)
                    else:
                        outputs = self.gpt(inputs_embeds=generated)
                    logits = outputs.logits
                    if generate_prefix is True:
                        tokens = torch.argmax(logits,-1)
                        if mask is not None:
                            mask = torch.tensor(mask,device=mask.device,dtype=torch.bool)
                            tokens = torch.masked_select(tokens,mask)
                        tokens = tokens.cpu().numpy()
                        output_prefix = self.tokenizer.decode(tokens)
                        
                        return [output_prefix]
                    logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                    logits = logits.softmax(-1).log()
                    if scores is None:
                        scores, next_tokens = logits.topk(beam_size, -1) # 1,beam_size
                        generated = generated.expand(beam_size, *generated.shape[1:])
                        if mask is not None:
                            mask = mask.expand(beam_size,*mask.shape)
                        next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                        if tokens is None:
                            tokens = next_tokens
                        else:
                            tokens = tokens.expand(beam_size, *tokens.shape[1:])
                            tokens = torch.cat((tokens, next_tokens), dim=1)
                    else:
                        logits[is_stopped] = -float(np.inf)
                        logits[is_stopped, 0] = 0
                        scores_sum = scores[:, None] + logits
                        seq_lengths[~is_stopped] += 1
                        scores_sum_average = scores_sum / seq_lengths[:, None]
                        scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(
                            beam_size, -1
                        )
                        next_tokens_source = next_tokens // scores_sum.shape[1]
                        seq_lengths = seq_lengths[next_tokens_source]
                        next_tokens = next_tokens % scores_sum.shape[1]
                        next_tokens = next_tokens.unsqueeze(1)
                        tokens = tokens[next_tokens_source]
                        tokens = torch.cat((tokens, next_tokens), dim=1)
                        generated = generated[next_tokens_source]
                        scores = scores_sum_average * seq_lengths
                        is_stopped = is_stopped[next_tokens_source]
                    next_token_embed = self.gpt.transformer.wte(next_tokens.squeeze()).view(
                        generated.shape[0], 1, -1
                    )
                    generated = torch.cat((generated, next_token_embed), dim=1)
                    if mask is not None:
                        mask = torch.cat((mask,torch.ones((beam_size,1),device=mask.device)),dim=1)
                    is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
                    if is_stopped.all():
                        break
            scores = scores / seq_lengths
            output_list = tokens.cpu().numpy()
            output_texts = [
                self.tokenizer.decode(output[: int(length)])
                for output, length in zip(output_list, seq_lengths)
            ]
            order = scores.argsort(descending=True)
            output_texts = [output_texts[i] for i in order]
            result_list.append(output_texts[0])
        return result_list

    def generate_eval(
        self,
        embed = None,
        prefix_mask=None,
        entry_length=30,
        temperature=1.0,
        top_p = 0.8,
        stop_token: str = ".",
        ):
        
        tokens = None
        stop_token_index = self.tokenizer.encode(stop_token)[0]
          
        result_list = []
        for item in range(embed.shape[0]):
            filter_value = -float('Inf')
            if prefix_mask is not None:
                if len(prefix_mask.shape) == 1:
                    prefix_mask = prefix_mask.unsqueeze(0)
                mask = prefix_mask[item]
            else:
                mask = None
            generated = embed[item].unsqueeze(0)
            for i in range(entry_length):
                if mask is not None:
                    outputs = self.gpt(inputs_embeds = generated,attention_mask=mask)
                else: 
                    outputs = self.gpt(inputs_embeds = generated)
                logits = outputs.logits
                # 取最后一行向量
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    nnf.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0
    
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = self.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if mask is not None:
                    mask = torch.cat((mask,torch.ones(1,device=mask.device)))
                if stop_token_index == next_token.item():
                    break
            try:
                output_list = list(tokens.squeeze().cpu().numpy())
            except TypeError:
                output_list = [13]
            # output_texts = tokenizer.decode(output_list)
            # # generated_list.append(output_text)
            output_text = self.tokenizer.decode(output_list)
            result_list.append(output_text)
        return result_list

    def SwinModel(self,load_ckpt=False):
        model =  SwinTransformer(img_size=224,
                                patch_size=4,
                                in_chans=3,
                                embed_dim=128,
                                depths=[ 2, 2, 18, 2 ],
                                num_heads=[ 4, 8, 16, 32 ],
                                window_size=7,
                                mlp_ratio=4.,
                                qkv_bias=True,
                                drop_rate=0.0,
                                drop_path_rate=0.1,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=False)
        if load_ckpt:
            ckpt = 'data/coco/swin_pretrain.th'
            state_dict = torch.load(ckpt,map_location='cpu')
            model.load_state_dict(state_dict)
        for params in model.parameters():
            params.requires_grad = False
        return model
               
    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)
   
    def get_contrastive_loss(self,image_feat,text_feat):
        """
        Args:
            image_feat, text_feat: normalized

        Returns: contrastive loss

        """
        loss_i2t = 0
        loss_t2i = 0
        
        assert image_feat.size(-1) == text_feat.size(-1)
        if len(image_feat.shape) == 3:
            image_feat = image_feat.reshape(-1,*image_feat.shape)
        layer_num=image_feat.shape[0]
        bsz = image_feat.shape[1]
        prefix_num=image_feat.shape[2]
        text_feat = text_feat.expand(image_feat.shape[0],*text_feat.shape)
        image_feat = image_feat.reshape(image_feat.shape[0],-1,image_feat.shape[-1])
        text_feat = text_feat.reshape(text_feat.shape[0],-1,text_feat.shape[-1])
        logits = image_feat @ text_feat.transpose(1,2) / self.temp
        logits = torch.sum(logits.reshape(layer_num,bsz,prefix_num,bsz,-1),dim=-1)
        logits = torch.sum(logits,dim=2)
        labels = torch.arange(bsz, device=image_feat.device)
        for i in range(image_feat.shape[0]):
            loss_i2t += nnf.cross_entropy(logits[i], labels)
            loss_t2i += nnf.cross_entropy(logits[i].t(), labels)
        return (loss_i2t + loss_t2i) / 2
    
    def get_eusian_loss(self,image_feat,text_feat):
        image_feat = torch.mean(image_feat,dim=2)
        text_feat = torch.mean(text_feat,dim=1)
        assert image_feat.size(-1) == text_feat.size(-1)
        text_feat = text_feat.expand(image_feat.shape[0],*text_feat.shape)
        loss = self.mse(image_feat,text_feat)
        return loss
    
    def get_n_pair_loss(self,image_feat,text_feat):
        assert image_feat.size(-1) == text_feat.size(-1)
        image_feat_mean = torch.mean(image_feat,dim=1)
        text_feat_mean = torch.mean(text_feat[:,:20,:],dim=1)
        target = torch.range(1,image_feat.shape[0],device = image_feat.device)
        loss = self.npair_loss(anchor=text_feat_mean,positive = image_feat_mean,target=target)
        return loss
    
    # def get_maskv2(self,image_prefix):
    #     q = self.linear3(image_prefix)
    #     k,v = self.linear1(image_prefix),self.linear2(image_prefix)
    #     w = torch.matmul(q, k.transpose(1,2))
    #     w = nn.Softmax(dim=-1)(w)
    #     w = w / np.sqrt(v.size(-1))
    #     new_v = torch.matmul(w, v)
    #     prefix_mask_prob = self.sigmoid(self.norm(new_v))
    #     mask = torch.where(prefix_mask_prob > 0.5,1,0)
    #     return mask

    
    def get_mask(self,image_prefix):
        # batch_size = image_prefix.shape[0]
        # prefix_mask = self.prefix_mask.expand(batch_size,*self.prefix_mask.shape) # bs,50
        # v = prefix_mask.unsqueeze(-1) # bs,50,1
        # q,k = image_prefix, image_prefix 
        # # k,v = self.linear1(image_prefix),self.linear2(image_prefix) # bs,50,50
        # attention = q @ k.transpose(1,2) # bs,50,50
        # prefix_mask_score =  torch.bmm(attention,v).squeeze() # bs,1,50 -> bs,50
        # prefix_mask_prob = self.sigmoid(self.norm(prefix_mask_score))
        # mask = torch.where(prefix_mask_prob > 0.5,1,0)
        
        batch_size = image_prefix.shape[0]
        prefix_mask = self.prefix_mask.expand(batch_size,*self.prefix_mask.shape) # bs,50
        q = prefix_mask.unsqueeze(1) # bs,1,50
        k,v = self.linear1(image_prefix),self.linear2(image_prefix) # bs,50,50
        attention = q @ k.transpose(1,2) # bs,1,50
        prefix_mask_score =  torch.bmm(attention,v).squeeze() # bs,1,50 -> bs,50
        prefix_mask_prob = self.sigmoid(self.norm(prefix_mask_score))
        mask = torch.where(prefix_mask_prob > self.threshold,1,0)
        return mask
    def forward(self, image: torch.Tensor, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        prefix = self.swin(image)
        embedding_text = self.gpt.transformer.wte(tokens)
        # if self.use_aux_loss and not self.use_sparce_mask:
        if self.use_aux_loss:
            # (layer_num,batch_size,prefix+clip_legnth,dim)(3,40,100,768)
            prefix_projections = self.clip_project(prefix)
            # aux_loss = self.get_n_pair_loss(prefix_projections,embedding_text)
            # aux_loss = self.get_contrastive_loss(image_feat=prefix_projections,text_feat=embedding_text[:,:20,:])
            # aux_loss = self.get_eusian_loss(image_feat=prefix_projections,text_feat=embedding_text)
            if self.return_intermediate:
                embedding_cat = torch.cat((prefix_projections[-1], embedding_text), dim=1)
            else:
                embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
            
            if self.use_sparce_mask:
                # prefix,prefix_mask = self.get_prefix(prefix_projections)
                prefix_mask = self.get_mask(prefix_projections)
                mask = torch.cat((prefix_mask,mask[:,self.prefix_length:]),dim=1)
                # prefix_mask_expand = prefix_mask.unsqueeze(-1).expand(prefix_projections.size())
                # prefix_projections = prefix_projections * prefix_mask_expand
            aux_loss = self.get_n_pair_loss(prefix_projections,embedding_text)
            if labels is not None:
                dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
                labels = torch.cat((dummy_token, tokens), dim=1)
            
            out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
            return out,aux_loss
        else:
            prefix_projections = self.clip_project(prefix)
            if self.use_sparce_mask:
                # prefix,prefix_mask = self.get_prefix(prefix_projections)
                prefix_mask = self.get_mask(prefix_projections)
                mask = torch.cat((prefix_mask,mask[:,self.prefix_length:]),dim=1)
            embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
            if labels is not None:
                dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
                labels = torch.cat((dummy_token, tokens), dim=1)
            out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
            return out

    def __init__(self, prefix_length: int=50, clip_length: Optional[int] = 50, prefix_size: int = 1024,
                 num_layers: int = 8, mapping_type: MappingType = MappingType.MLP, batch_size:int = 40,
                 temp=10,threshold=0.5,use_aux_loss=False,use_sparce_mask=False,return_intermediate=False,load_swin_ckpt=False):
        super(ClipCaptionModel, self).__init__()
        self.batch_size = batch_size
        self.prefix_length = prefix_length
        self.clip_length = clip_length
        self.use_aux_loss = use_aux_loss
        self.use_sparce_mask = use_sparce_mask
        # self.temp = temp
        self.threshold = threshold
        self.return_intermediate = return_intermediate
        self.gpt = GPT2LMHeadModel.from_pretrained('data/gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        self.prefix_mask = nn.Parameter(torch.randn(prefix_length), requires_grad=True)
        self.npair_loss = NpairLoss()
        self.linear1 = nn.Linear(self.gpt_embedding_size,prefix_length)
        self.linear2 = nn.Linear(self.gpt_embedding_size,prefix_length)
        # self.linear3 = nn.Linear(self.gpt_embedding_size,1)
        # self.mse = nn.MSELoss()
        self.swin = self.SwinModel(load_swin_ckpt)
        self.tokenizer = GPT2Tokenizer.from_pretrained('data/gpt2')
        self.stop_token_id = self.tokenizer.encode('.')[0]
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.LayerNorm(prefix_length)
        if mapping_type == MappingType.MLP:
            # origin
            # self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 4,
            #                          self.gpt_embedding_size * prefix_length))
            # v1.0
            # self.clip_project = MLP((prefix_size,
            #                          self.gpt_embedding_size))
            # v2.0
            self.clip_project = MLP((prefix_size,self.gpt_embedding_size*4,
                                     self.gpt_embedding_size))
        elif mapping_type == MappingType.Transformer:
            self.clip_project = TransformerMapper(prefix_size, self.gpt_embedding_size, prefix_length,
                                                                     clip_length, num_layers,return_intermediate=return_intermediate)
        # elif mapping_type == MappingType.CONV:
        #     self.clip_project = ConvNet(prefix_size, self.gpt_embedding_size, prefix_length)
        else:
            assert KeyError('the key does not exisit')
                
class ClipCaptionPrefix(ClipCaptionModel):

    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self

class MugeModel(ClipCaptionModel):
    def __init__(self,vocab_path,dict_path, use_aux_loss =False, use_kd=False, language_model=None,vocab_size=5659,max_length=40):
        super(MugeModel, self).__init__()
        self.max_length = max_length
        self.vocab_size = vocab_size
        if language_model is not None:
            self.lan_model = language_model
            self.lan_model.return_hidden_state = False
        else:
            self.lan_model=None
        self.gpt.transformer.wte = nn.Embedding(self.vocab_size,self.gpt_embedding_size)
        self.use_aux_loss = use_aux_loss
        self.text_field = TextField(vocab_path, dict_path)
        self.tokenizer = self.text_field.tokenizer
        self.vocab_size = vocab_size
        self.gpt.lm_head = nn.Linear(self.gpt_embedding_size,self.vocab_size)
        # if use_kd:
        #     self.gpt.lm_head = self.lan_model
        # else:
        #     self.gpt.lm_head = nn.Linear(self.gpt_embedding_size,self.vocab_size)
    def forward(self, image: torch.Tensor, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        prefix = self.swin(image)
        if self.lan_model is not None:
            with torch.no_grad():
                embedding_text = self.lan_model.word_emb(tokens)[:,1:,:]
        else:
            embedding_text = self.gpt.transformer.wte(tokens)
        
        if self.use_aux_loss:
            # (layer_num,batch_size,prefix+clip_legnth,dim)(3,40,100,768)
            prefix_projections = self.clip_project(prefix)
            embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
            aux_loss = self.get_n_pair_loss(prefix_projections,embedding_text)
            if labels is not None:
                dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
                labels = torch.cat((dummy_token, tokens), dim=1)
            out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
            return out,aux_loss    
        else:
            prefix_projections = self.clip_project(prefix)
            embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
            if labels is not None:
                dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
                labels = torch.cat((dummy_token, tokens), dim=1)
            out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
            return out
        
 

    def generate_test_muge(self,
        beam_size: int = 2,
        embed=None,
        prefix_mask=None,
        entry_length=150,
        temperature=1.0,
        generate_prefix=False,
        stop_token: str = "[eos]",
        
    ):
        result_list = []
        stop_token_index = 1
        device = embed.device
        
        for item in range(embed.shape[0]):
            tokens = None
            scores = None
            seq_lengths = torch.ones(beam_size, device=device)
            is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
            if prefix_mask is not None:
                if len(prefix_mask.shape) == 1:
                    prefix_mask = prefix_mask.unsqueeze(0)
                mask = prefix_mask[item]
            else:
                mask = None
            with torch.no_grad():
                if embed is not None:
                    generated = embed[item].unsqueeze(0)
                for i in range(entry_length):
                    if mask is not None:
                        outputs = self.gpt(inputs_embeds=generated,attention_mask=mask)
                    else:
                        outputs = self.gpt(inputs_embeds=generated)
                    logits = outputs.logits
                    if generate_prefix is True:
                        tokens = torch.argmax(logits,-1)
                        if mask is not None:
                            mask = torch.tensor(mask,device=mask.device,dtype=torch.bool)
                            tokens = torch.masked_select(tokens,mask)
                        tokens = tokens.cpu().numpy()
                        output_prefix = self.tokenizer.decode(tokens)
                        
                        return [output_prefix]
                    logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                    logits = logits.softmax(-1).log()
                    if scores is None:
                        scores, next_tokens = logits.topk(beam_size, -1) # 1,beam_size
                        generated = generated.expand(beam_size, *generated.shape[1:])
                        if mask is not None:
                            mask = mask.expand(beam_size,*mask.shape)
                        next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                        if tokens is None:
                            tokens = next_tokens
                        else:
                            tokens = tokens.expand(beam_size, *tokens.shape[1:])
                            tokens = torch.cat((tokens, next_tokens), dim=1)
                    else:
                        logits[is_stopped] = -float(np.inf)
                        logits[is_stopped, 0] = 0
                        scores_sum = scores[:, None] + logits
                        seq_lengths[~is_stopped] += 1
                        scores_sum_average = scores_sum / seq_lengths[:, None]
                        scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(
                            beam_size, -1
                        )
                        next_tokens_source = next_tokens // scores_sum.shape[1]
                        seq_lengths = seq_lengths[next_tokens_source]
                        next_tokens = next_tokens % scores_sum.shape[1]
                        next_tokens = next_tokens.unsqueeze(1)
                        tokens = tokens[next_tokens_source]
                        tokens = torch.cat((tokens, next_tokens), dim=1)
                        generated = generated[next_tokens_source]
                        scores = scores_sum_average * seq_lengths
                        is_stopped = is_stopped[next_tokens_source]
                    next_token_embed = self.gpt.transformer.wte(next_tokens.squeeze()).view(
                        generated.shape[0], 1, -1
                    )
                    generated = torch.cat((generated, next_token_embed), dim=1)
                    if mask is not None:
                        mask = torch.cat((mask,torch.ones((beam_size,1),device=mask.device)),dim=1)
                    is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
                    if is_stopped.all():
                        break
            scores = scores / seq_lengths
            output_list = tokens.cpu().numpy()
            output_texts = [
                self.tokenizer.convert_ids_to_tokens(output[: int(length)])
                for output, length in zip(output_list, seq_lengths)
            ]
            order = scores.argsort(descending=True)
            output_texts = [output_texts[i] for i in order]
            result_list.append(output_texts[0])
        return result_list

    def generate_eval_muge(
        self,
        embed = None,
        prefix_mask=None,
        entry_length=150,
        temperature=1.0,
        top_p = 0.8,
        stop_token: str = "[eos]",
        ):
        
        tokens = None
        stop_token_index = 1
          
        result_list = []
        for item in range(embed.shape[0]):
            filter_value = -float('Inf')
            if prefix_mask is not None:
                if len(prefix_mask.shape) == 1:
                    prefix_mask = prefix_mask.unsqueeze(0)
                mask = prefix_mask[item]
            else:
                mask = None
            generated = embed[item].unsqueeze(0)
            for i in range(entry_length):
                if mask is not None:
                    outputs = self.gpt(inputs_embeds = generated,attention_mask=mask)
                else: 
                    outputs = self.gpt(inputs_embeds = generated)
                logits = outputs.logits
                # 取最后一行向量
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    nnf.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0
    
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                # next_token_embed = self.lan_model(next_token)
                next_token_embed = self.gpt.transformer.wte(next_token)
                # next_token_embed = self.lan_model.word_emb(next_token) + self.lan_model.pos_emb(i+1)
                
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if mask is not None:
                    mask = torch.cat((mask,torch.ones(1,device=mask.device)))
                if stop_token_index == next_token.item():
                    break
            try:
                output_list = list(tokens.squeeze().cpu().numpy())
            except TypeError:
                output_list = [13]
            # output_texts = tokenizer.decode(output_list)
            # # generated_list.append(output_text)
            output_text = self.tokenizer.convert_ids_to_tokens(output_list)
            result_list.append(output_text)
        return result_list

        
class MugeModelPrefix(MugeModel):
    def parameters(self, recurse: bool = True):
            return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(MugeModelPrefix, self).train(mode)
        self.gpt.eval()
        return self

class Language_model(nn.Module):
    
    def __init__(self, bert, config, vocab_size, fix_bert=True, return_hidden_state=False) -> None:
        super(Language_model, self).__init__()
        self.bert = bert
        self.config = config
        self.config.vocab_size = vocab_size
        self.return_hidden_state = return_hidden_state
        
        # self.linear = nn.Linear(768, 768)
        self.word_emb = nn.Embedding(vocab_size, 768, padding_idx=0)  
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(150, 768, 0), freeze=True)
        self.cls = BertOnlyMLMHead(self.config)
        if fix_bert:
            self.grad()
        self.loss_fn =  CrossEntropyLoss()
    def grad(self):
        for p in self.bert.parameters():
            p.requires_grad = False

    def forward(self, x,seq_=None):
        
        bs, seq_len = x.shape[:2]
        mask_queries = (x != 0).unsqueeze(-1).float()
        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(bs, -1).to(x.device)
        seq = seq.masked_fill(mask_queries.squeeze(-1) == 0, 0)
        if seq_ is not None:
            seq = seq_
        with torch.no_grad():
            x1 = self.word_emb(x) + self.pos_emb(seq)
        outputs = self.bert.bert(inputs_embeds=x1)
        sequence_output = outputs[0]
        # sequence_output = self.linear(sequence_output)
        if self.return_hidden_state:
            return sequence_output
        
        prediction_scores = self.cls(sequence_output)
        masked_lm_loss = self.loss_fn(prediction_scores.view(-1, self.config.vocab_size), x.view(-1))
        out = MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        return out