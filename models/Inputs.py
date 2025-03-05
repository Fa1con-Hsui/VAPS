from calendar import c
import logging
import math
import pdb
from email_validator import ValidatedEmail
from sympy import flatten
import torch
import torch.nn as nn
import numpy as np
from utils import const
import torch.nn.functional as F
from .layers import PositionalEmbedding, Qwen2MLPSwiGLU


# ====================================
# Author: 2025/01/05


class MySwiGLU(nn.Module):
    def __init__(self, in_size, out_size) -> None:
        super().__init__()
        self.down_proj = nn.Linear(in_size, out_size)
        
    def forward(self, x):
        return self.down_proj(x * F.silu(x))
    

class TryMlp(nn.Module):
    def __init__(self, in_size, out_size, act_func):
        super().__init__()
        self.linear = nn.Linear(in_size, out_size)
        if act_func == 'try_relu':
            self.act = nn.ReLU()
        elif act_func == 'try_gelu':
            self.act = nn.GELU()
        elif act_func == 'try_prelu':
            self.act = nn.PReLU()
        elif act_func == 'try_silu':
            self.act = nn.SiLU()
        elif act_func == 'try_tanh':
            self.act = nn.Tanh()
        elif act_func == 'try_sigmoid':
            self.act = nn.Sigmoid()
        elif act_func == 'try_swiglu':
            self.act = MySwiGLU(out_size, out_size)
        else:
            raise NotImplementedError
    def forward(self, x):
        return self.act(self.linear(x))



class LLMTokenEncoder(nn.Module):
    def __init__(self, llm,
                 mid_emb_size=768,
                 output_size=64):
        super(LLMTokenEncoder, self).__init__()
        embeddings = llm.get_input_embeddings().weight.data.clone().detach()
        num_embeddings, embedding_dim = embeddings.shape
        
        self.embedding_layer = nn.Embedding.from_pretrained(
            embeddings,
            freeze=True  
        )
        
        self.num_embeddings = num_embeddings
        if const.args.use_LLMTokenEncoder_mlp:
            self.size = output_size
            self.mlp = Qwen2MLPSwiGLU(
                        in_size=embedding_dim, # 3584
                        mid_emb_size=mid_emb_size, 
                        out_size=self.size)
        else:
            self.size = embedding_dim
    
    def forward(self, input_ids):
        """
        input_ids: Tensor of shape (..., query_num, max_query_word_len) tokenids, LLMtoken
        output: Tensor of shape (..., query_num, max_query_word_len, final_emb_size)
        """
        # 
        embeddings = self.embedding_layer(input_ids)  # (..., query_num, max_query_word_len, embedding_dim)

        # MLP
        if const.args.use_LLMTokenEncoder_mlp:
            output = self.mlp(embeddings)  # (..., query_num, max_query_word_len, final_emb_size)
        else:
            output = embeddings
        return output # pad_token_id
    


class TextFeat(nn.Module):
    def __init__(self, device: torch.device, use_llm=False) -> None:
        super().__init__()
        
        assert hasattr(const, 'llm') and const.llm is not None
        assert hasattr(const, 'PAD_TOKEN_ID') and const.PAD_TOKEN_ID is not None
        assert hasattr(const, 'mid_word_id_dim') and const.mid_word_id_dim is not None
        
        self.device = device
        
        if use_llm:
            self.token_encoder = LLMTokenEncoder(const.llm, 
                                                mid_emb_size=const.mid_word_id_dim,
                                                output_size=const.word_id_dim,
                                                ) 
            word_emb_size = self.token_encoder.size
            self.num_embeddings = self.token_encoder.num_embeddings
        else:
            self.token_encoder = nn.Embedding(
                num_embeddings=len(const.token_map.word2id), 
                embedding_dim=const.word_id_dim, padding_idx=const.PAD_TOKEN_ID)
            nn.init.xavier_normal_(self.token_encoder.weight.data)
        
            word_emb_size = const.word_id_dim
            self.num_embeddings = len(const.token_map.word2id)

        # self.query_trans = nn.Linear(const.word_id_dim, const.final_emb_size) # (word_id_dim, final_emb_size)
        
        
        if const.args.text_feat_act == 'qwen2mlp':
            self.trans = Qwen2MLPSwiGLU(
                    in_size=word_emb_size,  
                    mid_emb_size = (word_emb_size + const.final_emb_size) // 2, 
                    out_size=const.final_emb_size) # word_id_dim -> final_emb_size
        elif const.args.text_feat_act.startswith('try'):
            self.trans = TryMlp(word_emb_size, const.final_emb_size, const.args.text_feat_act)
        else: 
            raise ValueError
        self.word_emb_size = word_emb_size
        self.size = const.final_emb_size
        

    def forward(self, sample: torch.Tensor):
        '''
        input NOTE embedding, mean pooling.
        sample: (..., text num, token_ids_max_len)
        -> (..., text num, token_ids_max_len, final_emb_size)
        -> output: (..., text num, final_emb_size)
        '''
        
        flatten_sample = sample.reshape((-1, sample.size(-1)))
        token_emb: torch.Tensor = self.token_encoder(flatten_sample) # (-1, token_ids_max_len) -> (-1, token_ids_max_len, word_id_dim) 3-d
        
        seqs_mask = (flatten_sample == const.PAD_TOKEN_ID) # 2-d (-1, token_ids_max_len)
        token_emb = token_emb.masked_fill(seqs_mask.unsqueeze(2), 0) # 3-d  (-1, token_ids_max_len, word_id_dim)
        seqs_len = (~seqs_mask).sum(1, keepdim=True) # 2-d  (-1, 1)
  
        
        modified_seqs_len = seqs_len.clone()
        modified_seqs_len[modified_seqs_len == 0] = 1 # 0,0
        mean_emb = torch.sum(token_emb, dim=1) / modified_seqs_len # 2-d (-1, word_id_dim) 
        
        mean_emb = mean_emb.masked_fill(seqs_len == 0, 0) # text(token)textembset 0
        
        text_emb = mean_emb.reshape((*sample.shape[:-1], -1)) # (-1, word_id_dim) -> (..., text num, word_id_dim)
        text_emb = self.trans(text_emb) # (query num, word_id_dim) -> (query num, final_emb_size)
        

        
        return text_emb




# Author: 2025/02/06

class AttentionPoolingExpert_v1(nn.Module):
    
    def __init__(self, emb_dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(emb_dim))  # 
        self.key = nn.Linear(emb_dim, emb_dim, bias=False)  # 
        
        # 
        nn.init.normal_(self.query, mean=0, std=1/math.sqrt(emb_dim))
        nn.init.xavier_normal_(self.key.weight)

    def forward(self, x, mask):
        # x: (batch, seq_len, emb_dim)
        # mask: (batch, seq_len)
        keys = self.key(x)  # 
        scores = torch.matmul(keys, self.query)  # (batch, seq_len)
        
        # padding mask
        scores = scores.masked_fill(mask, float('-inf'))  
        scores = scores.masked_fill(torch.all(mask,dim=1).unsqueeze(1), 1.0) # padding,attn_weights1, -infsoftmaxnan
        attn_weights = F.softmax(scores, dim=-1)  # (batch, seq_len)
        
        # 
        return torch.sum(x * attn_weights.unsqueeze(-1), dim=1)  # (batch, emb_dim)





class TextFeat_MoEv1(TextFeat):
    def __init__(self, device: torch.device, use_llm=False, use_moe=False) -> None:
        super().__init__(device,use_llm)
        self.moe_num_experts = const.args.moev1_num_experts
        self.moe_top_k = const.args.moev1_top_k
        self.use_moe = use_moe
        if use_moe:
            # 
            self.experts = nn.ModuleList([
                AttentionPoolingExpert_v1(self.word_emb_size) 
                for _ in range(self.moe_num_experts)
            ])
            
            # 
            self.gate = nn.Linear(self.word_emb_size, self.moe_num_experts)
            # nn.init.normal_(self.gate.weight, std=0.02)
            nn.init.xavier_normal_(self.gate.weight)
            
    def _moe_pooling(self, token_emb, mask):
        """MoE"""
        # token_emb: (batch, seq_len, emb_dim)
        # mask: (batch, seq_len)
        # seqs_len: (batch, 1) 
        
        batch_size = token_emb.size(0)
        
        # 
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(token_emb, mask)  # (batch, emb_dim)
            expert_outputs.append(expert_out)
        expert_outputs = torch.stack(expert_outputs, dim=1)  # (batch, K, emb_dim)
        
        # 
        gate_input = token_emb.mean(dim=1)  #  # (batch, emb_dim)
        gate_scores = self.gate(gate_input)  # (batch, K)
        
        # top-k
        topk_scores, topk_indices = torch.topk(
            gate_scores, self.moe_top_k, dim=1)  # (batch, top_k)
        
        # 
        gate_weights = F.softmax(topk_scores, dim=-1) # (batch, top_k)
        
        # 
        full_weights = torch.zeros_like(gate_scores) # (batch, K)
        full_weights.scatter_(1, topk_indices, gate_weights) # (batch, K)
        
        # 
        weighted_emb = torch.einsum('bk,bkd->bd', full_weights, expert_outputs)
        
        return weighted_emb

    def forward(self, sample: torch.Tensor, use_text_moe=False, output_word_dim_emb=False):
        '''
        input  embedding, mean pooling
        sample: (..., text num, token_ids_max_len)
        -> (..., text num, token_ids_max_len, final_emb_size)
        -> output: (..., text num, final_emb_size)
        '''
        
        flatten_sample = sample.reshape((-1, sample.size(-1)))
        token_emb: torch.Tensor = self.token_encoder(flatten_sample) # (-1, token_ids_max_len) -> (-1, token_ids_max_len, word_id_dim) 3-d
        
        seqs_mask = (flatten_sample == const.PAD_TOKEN_ID) # 2-d (-1, token_ids_max_len)
        token_emb = token_emb.masked_fill(seqs_mask.unsqueeze(2), 0) # 3-d  (-1, token_ids_max_len, word_id_dim)
        seqs_len = (~seqs_mask).sum(1, keepdim=True) # 2-d  (-1, 1)
  
        
        if use_text_moe and self.use_moe:
            # MoE
            pooled_emb = self._moe_pooling(token_emb, seqs_mask)
        else:
            modified_seqs_len = seqs_len.masked_fill(seqs_len == 0, 1)
            pooled_emb = torch.sum(token_emb, dim=1) / modified_seqs_len
            
        pooled_emb = pooled_emb.masked_fill(seqs_len == 0, 0.0)
        text_emb = pooled_emb.reshape((*sample.shape[:-1], -1)) # (-1, word_id_dim) -> (..., text num, word_id_dim)
        final_text_emb = self.trans(text_emb) # (..., text num, word_id_dim) -> (..., text num, final_emb_size)
        
        if output_word_dim_emb:
            return final_text_emb, text_emb
        else:
            return final_text_emb

    

class SelfAttentionPoolingExpert_v1(nn.Module):
    def __init__(self, emb_dim, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = math.sqrt(emb_dim // num_heads)  # 


        self.q_proj = nn.Linear(emb_dim, emb_dim, bias=False)  # Query 
        self.k_proj = nn.Linear(emb_dim, emb_dim, bias=False)  # Key 
        self.v_proj = nn.Linear(emb_dim, emb_dim, bias=False)  # Value 
        self.out_proj = nn.Linear(emb_dim, emb_dim, bias=False)  

        nn.init.xavier_normal_(self.q_proj.weight)
        nn.init.xavier_normal_(self.k_proj.weight)
        nn.init.xavier_normal_(self.v_proj.weight)
        nn.init.xavier_normal_(self.out_proj.weight)

    def forward(self, x, mask):
        """
        x: (batch, seq_len, emb_dim) - token embedding
        mask: (batch, seq_len) - padding mask
        """
        batch_size, seq_len, emb_dim = x.shape
        head_dim = emb_dim // self.num_heads  # 

        #  Query, Key, Value
        Q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)  # (batch, heads, seq_len, head_dim)
        K = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)  # (batch, heads, seq_len, head_dim)
        V = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)  # (batch, heads, seq_len, head_dim)

        # 
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch, heads, seq_len, seq_len)
        attn_scores = attn_scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))  # mask 
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, heads, seq_len, seq_len)

        # 
        attn_output = torch.matmul(attn_weights, V)  # (batch, heads, seq_len, head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, emb_dim)  # (batch, seq_len, emb_dim)

        # 
        output = self.out_proj(attn_output.mean(dim=1))  # (batch, emb_dim)

        return output


class CrossAttentionPoolingExpert(nn.Module):
    
    def __init__(self, emb_dim, cross_att_dim=None):
        super().__init__()
        self.emb_dim = emb_dim
        self.cross_att_dim = cross_att_dim
        
        self.key = nn.Linear(emb_dim, emb_dim, bias=False)  # 
        if cross_att_dim is not None:
            self.cross_attention_query_weight = nn.Linear(cross_att_dim, emb_dim, bias=False)
        
        nn.init.xavier_normal_(self.key.weight)
        if cross_att_dim is not None:
            nn.init.xavier_normal_(self.cross_attention_query_weight.weight)

    def setup_cross_attention(self, cross_query):
        bs = cross_query.size(0)
        # emb_dim
        transformed_query = self.cross_attention_query_weight(cross_query)  # (bs, 1, emb_dim)
        return transformed_query
    
    def cross_attention_pooling(self, x, cross_query, mask):

        keys = self.key(x)  
 
        scores = torch.matmul(keys, cross_query.transpose(1, 2))  # (batch, seq_len, 1)
        scores = scores.squeeze(-1)  # (batch, seq_len)
        
        # mask
        scores = scores.masked_fill(mask, float('-inf'))
        # -inf
        scores = scores.masked_fill(torch.all(mask, dim=1).unsqueeze(1), 1.0)
        attn_weights = F.softmax(scores, dim=-1)  # (batch, seq_len)
        
        return torch.sum(x * attn_weights.unsqueeze(-1), dim=1)  # (batch, emb_dim)
    
    def forward(self, x, mask, cross_query=None):

        if cross_query is not None:
            # cross_queryemb_dim
            transformed_query = self.setup_cross_attention(cross_query)
            return self.cross_attention_pooling(x, transformed_query, mask)
        else:
            raise ValueError("Cross attention query is required for this expert.")



class CrossAttentionPoolingExpert_v1(nn.Module):
    def __init__(self, word_emb_dim, query_dim):
        """
        word_emb_dim: token embedding 
        query_dim:  query 
        """
        super().__init__()
        self.query_proj = nn.Linear(query_dim, word_emb_dim, bias=False)  #  query
        self.key_proj = nn.Linear(word_emb_dim, word_emb_dim, bias=False)  #  key
        
        nn.init.xavier_normal_(self.query_proj.weight)
        nn.init.xavier_normal_(self.key_proj.weight)

    def forward(self, x, mask, query):
        """
        x: (batch, seq_len, word_emb_dim) - token embedding
        mask: (batch, seq_len) - padding mask
        query: (batch, 1, query_dim) - 
        """
        query = self.query_proj(query)  # (batch, 1, word_emb_dim)
        keys = self.key_proj(x)  # (batch, seq_len, word_emb_dim)

        scores = torch.matmul(query, keys.transpose(-2, -1)).squeeze(1)  # (batch, seq_len)
        scores = scores.masked_fill(mask, float('-inf'))
        scores = scores.masked_fill(torch.all(mask, dim=1).unsqueeze(1), 1.0)
        attn_weights = F.softmax(scores, dim=-1)  # (batch, seq_len)

        return torch.sum(x * attn_weights.unsqueeze(-1), dim=1)  # (batch, word_emb_dim)


class TextFeat_MoEv2(TextFeat):
    def __init__(self, device: torch.device, use_llm=False, use_moe=False) -> None:
        super().__init__(device, use_llm)
        self.moe_num_experts = const.args.moev1_num_experts
        self.moe_top_k = const.args.moev1_top_k
        self.use_moe = use_moe

        if use_moe:
            assert self.moe_num_experts % 2 == 0
            self.experts = nn.ModuleList([
                AttentionPoolingExpert_v1(self.word_emb_size) 
                for _ in range(self.moe_num_experts // 2)
            ] + [
                CrossAttentionPoolingExpert_v1(self.word_emb_size, self.word_emb_size)
                for _ in range(self.moe_num_experts // 2)
            ])

            self.gate = nn.Linear(self.word_emb_size, self.moe_num_experts)
            nn.init.xavier_normal_(self.gate.weight)

    def _moe_pooling(self, token_emb, mask, moe_extra_query=None):
        expert_outputs = []

        for i, expert in enumerate(self.experts):
            if isinstance(expert, CrossAttentionPoolingExpert_v1):
                if moe_extra_query is None:
                    continue
                expert_out = expert(token_emb, mask, moe_extra_query)  
            else:
                expert_out = expert(token_emb, mask)  
            expert_outputs.append(expert_out)

        expert_outputs = torch.stack(expert_outputs, dim=1)  # (batch, K, emb_dim)

        gate_input = token_emb.mean(dim=1)
        gate_scores = self.gate(gate_input)  # (batch, K)
        topk_scores, topk_indices = torch.topk(gate_scores, self.moe_top_k, dim=1)  
        gate_weights = F.softmax(topk_scores, dim=-1)  

        full_weights = torch.zeros_like(gate_scores)
        full_weights.scatter_(1, topk_indices, gate_weights)

        weighted_emb = torch.einsum('bk,bkd->bd', full_weights, expert_outputs)
        return weighted_emb

    def forward(self, sample: torch.Tensor, use_text_moe=False, moe_extra_query=None, output_word_dim_emb=False):
        '''
        if sample is (bs,text_num,token_num), moe_extra_query must be (bs,text_num,1,self.word_emb_siz)
        if sample is (bs,token_num), moe_extra_query must be (bs,1,self.word_emb_siz)
        moe_query_tensorssample word_emb_size 
        '''
        
        flatten_sample = sample.reshape((-1, sample.size(-1))) # (bs*text_num,token_num)
        if moe_extra_query is not None:
            moe_extra_query = moe_extra_query.reshape((flatten_sample.size(0), 1, self.word_emb_size))
        token_emb = self.token_encoder(flatten_sample)  
        
        seqs_mask = (flatten_sample == const.PAD_TOKEN_ID)  
        token_emb = token_emb.masked_fill(seqs_mask.unsqueeze(2), 0)  
        seqs_len = (~seqs_mask).sum(1, keepdim=True)  

        if use_text_moe and self.use_moe: 
            pooled_emb = self._moe_pooling(token_emb, seqs_mask, moe_extra_query)
        else:
            modified_seqs_len = seqs_len.masked_fill(seqs_len == 0, 1)
            pooled_emb = torch.sum(token_emb, dim=1) / modified_seqs_len
            
        pooled_emb = pooled_emb.masked_fill(seqs_len == 0, 0.0)
        text_emb = pooled_emb.reshape((*sample.shape[:-1], -1))  
        final_text_emb = self.trans(text_emb)  

        if output_word_dim_emb:
            return final_text_emb, text_emb
        else:
            return final_text_emb



class TextFeat_MoEv3(TextFeat_MoEv2):
    def __init__(self, device: torch.device, use_llm=False, use_moe=False) -> None:
        super().__init__(device, use_llm)
        self.moe_num_experts = const.args.moev1_num_experts
        self.moe_top_k = const.args.moev1_top_k
        self.use_moe = use_moe

        if use_moe:
            assert self.moe_num_experts % 3 == 0
            self.experts = nn.ModuleList([
                AttentionPoolingExpert_v1(self.word_emb_size) 
                for _ in range(self.moe_num_experts // 3)
            ] + [
                CrossAttentionPoolingExpert_v1(self.word_emb_size, self.word_emb_size)
                for _ in range(self.moe_num_experts // 3)
            ] + [
                SelfAttentionPoolingExpert_v1(self.word_emb_size)
                for _ in range(self.moe_num_experts // 3)
            ])

            self.gate = nn.Linear(self.word_emb_size, self.moe_num_experts)
            nn.init.xavier_normal_(self.gate.weight)



class ObjectFeat(nn.Module):
    def __init__(self, map_vocab, text_emb_lay, 
                 category_features,
                 text_features,
                 category_features_num, active_features,
                 padding_idx=None, list_category_features=[],
                 
                 llm_map_vocab=None,llm_text_features=[],llm_text_emb_lay=None,
                 ) -> None:
        super().__init__()
        self.padding_idx = padding_idx # 0pad
        self.map_vocab = map_vocab
        self.text_emb_lay = text_emb_lay
        self.size = const.final_emb_size
        self.active_features = active_features
        
        # extra llm emb
        self.llm_map_vocab = llm_map_vocab 
        self.llm_text_features = llm_text_features
        self.llm_text_emb_lay = llm_text_emb_lay
        
        self.list_category_features = list_category_features
        self.concat_size = 0
        self.category_features = category_features
        for attr in self.category_features:
            if attr not in self.active_features:
                continue
            setattr(
                self, f'{attr}_emb',
                nn.Embedding(
                    num_embeddings=category_features_num[attr],
                    embedding_dim=const.category_feature_dim,
                    padding_idx=padding_idx,
                )
            )
            nn.init.xavier_normal_(getattr(self, f'{attr}_emb').weight.data)
            if self.padding_idx is not None:
                getattr(self, f'{attr}_emb').weight.data[self.padding_idx, :] = 0
            self.concat_size += const.category_feature_dim

        # self.text_features = const.user_text_features
        self.text_features = text_features
        for attr in self.text_features:
            if attr not in self.active_features:
                continue
            self.concat_size += self.text_emb_lay.size
            
        # extra llm emb
        if self.llm_map_vocab is not None: 
            assert self.llm_text_emb_lay is not None
            for attr in self.llm_text_features:
                if attr not in self.active_features:
                    continue
                self.concat_size += self.llm_text_emb_lay.size

        if const.args.object_feat_act == 'qwen2mlp':
            self.trans = Qwen2MLPSwiGLU(
                    in_size=self.concat_size,  
                    mid_emb_size = (self.concat_size + self.size) // 2, 
                    out_size=self.size) # (size, final_emb_size)
        elif const.args.object_feat_act.startswith('try'):
            self.trans = TryMlp(self.concat_size, self.size, const.args.object_feat_act)
        else:
            raise ValueError


    ##############################################################################3
    

    def index_access_text_emb(self, sample, use_text_moe=False, **kwargs):
        text_feats_ls = []
        for attr in self.text_features:
            if attr not in self.active_features:
                continue
            index = self.map_vocab[attr][sample] # (*sample_size, max_token_id_len)
            text_feats_ls.append(
                self.text_emb_lay(index,use_text_moe=use_text_moe, **kwargs)  # (*sample_size, self.text_emb_lay.size)
            )
        sample_text_emb = torch.cat(text_feats_ls, dim=-1)
        return sample_text_emb
    def index_access_llm_text_emb(self, sample, use_text_moe=False, **kwargs):
        llm_text_feats_ls = []
        for attr in self.llm_text_features:
            if attr not in self.active_features:
                continue
            index = self.llm_map_vocab[attr][sample]
            llm_text_feats_ls.append(
                self.llm_text_emb_lay(index,use_text_moe=use_text_moe, **kwargs)  # (*sample_size, self.llm_text_emb_lay.size)
            ) 
        sample_llm_text_emb = torch.cat(llm_text_feats_ls, dim=-1)
        return sample_llm_text_emb
        
    def get_text_emb(self, sample, use_text_moe=False, **kwargs):
        if self.padding_idx is not None:
            new_sample = sample.reshape((-1,)) # (batch_size*item_num,) flatten
            result_emb = torch.zeros(
                (new_sample.shape[0], self.size), device=sample.device) 
            # (batch_size*item_num, final_emb_size) like (1024*5, 64)
            
            sub_mask = new_sample != self.padding_idx
            if sub_mask.sum() > 0:
                sub_sample = new_sample[sub_mask] # 1dint tensor 
                if 'moe_extra_query' in kwargs and kwargs['moe_extra_query'] is not None:
                    moe_extra_query = kwargs['moe_extra_query']
                    new_moe_extra_query = moe_extra_query.reshape((new_sample.shape[0],*moe_extra_query.shape[-2:]))
                    kwargs['moe_extra_query'] = new_moe_extra_query[sub_mask]
                result_emb[sub_mask] = self.index_access_text_emb(sub_sample, use_text_moe=use_text_moe, **kwargs)
            return result_emb.reshape((*sample.shape, self.size))
        else:
            sample_emb = self.index_access_text_emb(sample, use_text_moe=use_text_moe, **kwargs)       
            return sample_emb
        
    def get_llm_text_emb(self, sample, use_text_moe=False, **kwargs):
        assert self.llm_map_vocab is not None and self.llm_text_emb_lay is not None
        if self.padding_idx is not None:
            new_sample = sample.reshape((-1,)) # (batch_size*item_num,) flatten
            result_emb = torch.zeros(
                (new_sample.shape[0], self.size), device=sample.device) 
            # (batch_size*item_num, final_emb_size) like (1024*5, 64)
            
            sub_mask = new_sample != self.padding_idx
            if sub_mask.sum() > 0:
                sub_sample = new_sample[sub_mask] # 1dint tensor 
                if 'moe_extra_query' in kwargs and kwargs['moe_extra_query'] is not None:
                    moe_extra_query = kwargs['moe_extra_query']
                    new_moe_extra_query = moe_extra_query.reshape((new_sample.shape[0],*moe_extra_query.shape[-2:]))
                    kwargs['moe_extra_query'] = new_moe_extra_query[sub_mask]
                result_emb[sub_mask] = self.index_access_llm_text_emb(sub_sample, use_text_moe=use_text_moe, **kwargs)
            return result_emb.reshape((*sample.shape, self.size))
        else:
            sample_emb = self.index_access_llm_text_emb(sample, use_text_moe=use_text_moe, **kwargs)       
            return sample_emb
        
        
    ##############################################################################3
        
    def index_access_emb(self, sample, use_text_moe=False, **kwargs):
        feats_ls = []
        for attr in self.category_features:
            if attr not in self.active_features:
                continue
            index = self.map_vocab[attr][sample]
            if attr in self.list_category_features:
                feats_ls.append(
                    getattr(self, f'{attr}_emb')(index).sum(dim=-2) #  (*sample_size, const.category_feature_dim)
                )
            else:    
                feats_ls.append(
                    getattr(self, f'{attr}_emb')(index) # # (*sample_size, const.category_feature_dim)
                )
        for attr in self.text_features:
            if attr not in self.active_features:
                continue
            index = self.map_vocab[attr][sample] # (*sample_size, max_token_id_len)
            feats_ls.append(
                self.text_emb_lay(index,use_text_moe=use_text_moe, **kwargs)  # (*sample_size, self.text_emb_lay.size)
            )
            
        if self.llm_map_vocab is not None:
            for attr in self.llm_text_features:
                if attr not in self.active_features:
                    continue
                index = self.llm_map_vocab[attr][sample]
                feats_ls.append(
                    self.llm_text_emb_lay(index,use_text_moe=use_text_moe, **kwargs)  # (*sample_size, self.llm_text_emb_lay.size)
                ) 
        
        sample_emb = torch.cat(feats_ls, dim=-1) # (*sample_size, self.concat_size) 
        return self.trans(sample_emb)      
    def forward(self, sample, use_text_moe=False, **kwargs):
        if self.padding_idx is not None:
            
            new_sample = sample.reshape((-1,)) # (batch_size*item_num,) flatten
            result_emb = torch.zeros(
                (new_sample.shape[0], self.size), device=sample.device) 
            # (batch_size*item_num, final_emb_size) like (1024*5, 64)
            
            sub_mask = new_sample != self.padding_idx
            if sub_mask.sum() > 0:
                sub_sample = new_sample[sub_mask] # 1dint tensor 
                if 'moe_extra_query' in kwargs and kwargs['moe_extra_query'] is not None:
                    moe_extra_query = kwargs['moe_extra_query']
                    new_moe_extra_query = moe_extra_query.reshape((new_sample.shape[0],*moe_extra_query.shape[-2:]))
                    kwargs['moe_extra_query'] = new_moe_extra_query[sub_mask]
                result_emb[sub_mask] = self.index_access_emb(sub_sample, use_text_moe=use_text_moe, **kwargs)
            return result_emb.reshape((*sample.shape, self.size))
        else:
            sample_emb = self.index_access_emb(sample, use_text_moe=use_text_moe, **kwargs)      
            return sample_emb
    
        

    

class SrcSessionFeat(nn.Module):
    # def __init__(self, device: torch.device) -> None:
    def __init__(self, query_feat, item_feat, user_feat,  map_vocab) -> None:
        super().__init__()
        self.query_feat = query_feat
        self.item_feat = item_feat
        self.user_feat = user_feat
        self.pad_item_id = const.PAD_ITEM_ID
        self.pad_src_session_id = const.PAD_SEARCH_SESSION_ID

        self.map_vocab = map_vocab

    def get_user_emb(self, sample, use_text_moe=False, **kwargs):
        return self.user_feat(sample, use_text_moe=use_text_moe, **kwargs)

    def get_item_emb(self, sample, use_text_moe=False, **kwargs):
        return self.item_feat(sample, use_text_moe=use_text_moe, **kwargs)

    def get_query_emb(self, sample, use_text_moe=False, **kwargs):
        # pooling,,final_emb_size,(max_token_ids_len)
        return self.query_feat(sample, use_text_moe=use_text_moe, **kwargs)

    def forward(self, sample):
        """
        only for unisar from unisar
        """
        # padsessionemb 
        new_sample = sample.reshape((-1,)) 
        sub_mask = new_sample != self.pad_src_session_id

        result_query_emb = torch.zeros(
            (new_sample.shape[0], const.final_emb_size), device=sample.device)
        result_item_emb = torch.zeros(
            (new_sample.shape[0], const.max_session_item_len, const.final_emb_size), device=sample.device)
        result_item_mask = torch.zeros(
            (new_sample.shape[0], const.max_session_item_len), device=sample.device).bool()

        if sub_mask.sum() > 0:
            sub_sample = new_sample[sub_mask] # (big session num,)
            sub_query_id = self.map_vocab['query'][sub_sample] # (big session num,)
            sub_click_item_ls = self.map_vocab['pos_items'][sub_sample] # (big session num, max_session_item_len)
            sub_query_emb = self.get_query_emb(sub_query_id) # (big session num, emb_d)
            
            # sub_click_item_mask = torch.where(
            #     sub_click_item_ls == 0, 0, 1).bool()  # Author equally modify
            sub_click_item_mask = sub_click_item_ls != self.pad_item_id  # (big session num, max_session_item_len)
            
            sub_click_item_emb = self.get_item_emb(sub_click_item_ls)  # (big session num, max_session_item_len, emb_d)
            
            # 
            result_query_emb[sub_mask] = sub_query_emb 
            result_item_emb[sub_mask] = sub_click_item_emb
            result_item_mask[sub_mask] = sub_click_item_mask

        result_query_emb = result_query_emb.reshape(
            (*sample.shape, const.final_emb_size))
        result_item_emb = result_item_emb.reshape(
            (*sample.shape, const.max_session_item_len, const.final_emb_size))
        result_item_mask = result_item_mask.reshape(
            (*sample.shape, const.max_session_item_len))

        return [result_query_emb, result_item_emb, result_item_mask]



class ConvSrcSessionFeat(SrcSessionFeat):
    def __init__(self, query_feat, item_feat, user_feat, conv_feat, map_vocab) -> None:
        super().__init__(query_feat, item_feat, user_feat, map_vocab)
        self.conv_feat = conv_feat
        
    def get_conv_emb(self, sample,use_text_moe=False, **kwargs):
        return self.conv_feat(sample,use_text_moe=use_text_moe, **kwargs)  
    
    def get_map_attr(self, src_session_id: torch.Tensor, attr_name):
        return self.map_vocab[attr_name][src_session_id]
    def get_query_emb_based_on_src_session_id(self, src_session_id: torch.Tensor, use_text_moe=False, **kwargs):
        '''
        src_session_id: tensor (batch_size,)
        
        output:
            query_emb: tensor (batch_size, final_emb_size)
        '''
        if 'query' in self.map_vocab and 'query_cut' in self.map_vocab:
            if const.args.use_query_cut_mode=='only_no_cut':
                return self.get_query_emb(self.map_vocab['query'][src_session_id],use_text_moe=use_text_moe, **kwargs)
            elif const.args.use_query_cut_mode=='only_cut':
                return self.get_query_emb(self.map_vocab['query_cut'][src_session_id],use_text_moe=use_text_moe, **kwargs)
            elif const.args.use_query_cut_mode=='add':
                query = self.get_query_emb(self.map_vocab['query'][src_session_id],use_text_moe=use_text_moe, **kwargs)
                query_cut = self.get_query_emb(self.map_vocab['query_cut'][src_session_id],use_text_moe=use_text_moe, **kwargs)
                return query+query_cut
            else:
                raise NotImplementedError
        else:
            return self.get_query_emb(self.map_vocab['query'][src_session_id],use_text_moe=use_text_moe, **kwargs)
        # query_token_ids = self.map_vocab['query'][src_session_id] # (batch_size, token_ids_max_len)
        # query_embs = self.get_query_emb(self.map_vocab['query'][src_session_id]) # (batch_size, final_emb_size)
        # return query_embs # (batch_size, final_emb_size)
    
    def get_link_conv_embs_based_on_src_session_id(self, src_session_id: torch.Tensor, use_text_moe=False, **kwargs):
        '''
        src_session_id: tensor (batch_size,)
        
        output:
            link_conv_embs: tensor (batch_size, pad conv num, final_emb_size), In descending order of time
        '''
        assert 'conv_record_idx' in self.map_vocab.keys()
        related_conv_reidx = self.map_vocab['conv_record_idx'][src_session_id] # (batch_size, pad conv num)
        return self.get_conv_emb(related_conv_reidx, use_text_moe=use_text_moe, **kwargs)# (batch_size, pad conv num, final_emb_size)
        
        
        