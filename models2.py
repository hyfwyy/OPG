import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel
from convnext import convnext_tiny
from typing import Tuple, Optional
from enum import Enum

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

class ConvNet(nn.Module):
    def __init__(self, prefix_dim=1024, gpt_embedding_size=768, prefix_length=20,prefix_size=50):
        super().__init__()
        self.l1 = nn.Linear(prefix_dim,prefix_dim*4)
        # self.dwconv = nn.Conv1d(prefix_size,prefix_size,kernel_size=3,groups=prefix_size,padding=1)
        self.l2 = nn.Linear(prefix_dim*4,gpt_embedding_size)
        self.l3 = nn.Conv1d(prefix_size,prefix_length,kernel_size=1)
        # self.norm = nn.LayerNorm()
        # self.model = nn.Sequential(self.l1,self.dwconv,self.l2,self.l3)
        nn.Conv1d
    def forward(self,x):
        x = self.l1(x)
        # x = self.dwconv(x)
        x = self.l2(x)
        x = self.l3(x)
        return x
        # return self.model(x)

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

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)
    def get_mask(self,image_prefix):
        # prefix_mask = self.prefix_mask.expand(self.batch_size,*self.prefix_mask.shape) # bs,50
        # q = prefix_mask.unsqueeze(1) # bs,1,50
        # k,v = self.linear1(prefix_projections),self.linear2(prefix_projections) # bs,50,50
        # attention = q @ k.transpose(1,2) # bs,1,50
        # prefix_mask_score =  torch.bmm(attention,v).squeeze() # bs,1,50 -> bs,50
        # prefix_mask_prob = self.sigmoid(self.norm(prefix_mask_score))
        # mask = torch.where(prefix_mask_prob > 0.5,1,0)
        
        prefix_mask = self.prefix_mask.expand(self.batch_size,*self.prefix_mask.shape).unsqueeze(-1)
        attention = image_prefix @ image_prefix.transpose(1,2)
        prefix_mask_score =  torch.bmm(attention,prefix_mask).squeeze()
        prefix_mask_prob = self.sigmoid(self.norm(prefix_mask_score))
        mask = torch.where(prefix_mask_prob > 0.5,1,0)
        return mask
    def get_prefix(self,prefix_all):
        prefix_clip = torch.full_like(prefix_all,0,device=prefix_all.device)
        prefix_mask = torch.zeros((self.batch_size,self.prefix_length),device=prefix_all.device)
        cos_sim = prefix_all @ prefix_all.transpose(-1,-2)
        values,indices = torch.max(cos_sim,dim=1)
        mask = indices.eq(torch.range(0,self.prefix_length-1,device=prefix_all.device))
        for i in range(mask.shape[0]):
            temp = prefix_all[i,mask[i]]
            prefix_clip[i,50-temp.shape[0]:] = temp
            prefix_mask[i,50-temp.shape[0]:] = torch.ones(temp.shape[0])
        return prefix_clip,prefix_mask
   
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
    
    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        
        # prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        # weight = self.relu(self.weight).to(prefix_projections.device)
        # weight = self.sigmoid(self.weight).to(prefix_projections.device)
        # prefix_projections = weight.unsqueeze(2)*prefix_projections
        if self.use_aux_loss:
            # (layer_num,batch_size,prefix+clip_legnth,dim)(3,40,100,768)
            prefix_projections = self.clip_project(prefix)
            aux_loss = self.get_n_pair_loss(prefix_projections,embedding_text)
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
            if labels is not None:
                dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
                labels = torch.cat((dummy_token, tokens), dim=1)
            
            out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
            return out,aux_loss
        else:
            prefix_projections = self.clip_project(prefix)
            if self.use_sparce_mask:
                prefix,prefix_mask = self.get_prefix(prefix_projections)
                # prefix_mask = self.get_mask(prefix_projections)
                mask = torch.cat((prefix_mask,mask[:,self.prefix_length:]),dim=1)
            embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
            if labels is not None:
                dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
                labels = torch.cat((dummy_token, tokens), dim=1)
            out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
            return out

    def __init__(self, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 1024,
                 num_layers: int = 8, mapping_type: MappingType = MappingType.MLP, batch_size:int = 40,
                 temp=10,use_aux_loss=False,use_sparce_mask=False,return_intermediate=False):
        super(ClipCaptionModel, self).__init__()
        self.batch_size = batch_size
        self.prefix_length = prefix_length
        self.clip_length = clip_length
        self.use_aux_loss = use_aux_loss
        self.use_sparce_mask = use_sparce_mask
        self.temp = temp
        self.return_intermediate = return_intermediate
        self.gpt = GPT2LMHeadModel.from_pretrained('data/gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        self.prefix_mask = nn.Parameter(torch.randn(prefix_length), requires_grad=True)
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.LayerNorm(prefix_length)
        self.npair_loss = NpairLoss()
        # self.linear1 = nn.Linear(self.gpt_embedding_size,prefix_length)
        # self.linear2 = nn.Linear(self.gpt_embedding_size,prefix_length)
        self.mse = nn.MSELoss()
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
        elif mapping_type == MappingType.CONV:
            self.clip_project = ConvNet(prefix_size, self.gpt_embedding_size, prefix_length)
        else:
            assert KeyError('the key does not exisit')
                


class ClipCaptionPrefix(ClipCaptionModel):

    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self


class ConvCapModel(nn.Module):
    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: torch.Tensor, image: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix = self.convnext(image)
        prefix = prefix.transpose(1,2)
        embedding_cat = torch.cat((prefix, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_size:int,prefix_length: int):
        super(ConvCapModel, self).__init__()
        self.prefix_length = prefix_length
        self.prefix_size = prefix_size
        
        self.gpt = GPT2LMHeadModel.from_pretrained('data/gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        self.convnext = convnext_tiny(pretrained=True)
        self.l1 = nn.Conv1d(prefix_size,prefix_length,kernel_size=1,groups=prefix_length)


