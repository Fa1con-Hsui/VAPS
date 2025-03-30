import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):

    def __init__(self, max_len, dim):
        super().__init__()

        # Compute the positional encodings once in log space.
        self.pe = nn.Embedding(max_len, dim)
        nn.init.xavier_normal_(self.pe.weight.data)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)
    
class PositionalEmbedding_v2(nn.Module):

    def __init__(self, max_len, dim):
        super().__init__()

        # Compute the positional encodings once in log space.
        self.pe = nn.Embedding(max_len, dim)
        nn.init.xavier_normal_(self.pe.weight.data)

    def forward(self, x):
        batch_size = x.size(0)
        _len = x.size(1)
        return self.pe.weight[:_len,:].unsqueeze(0).repeat(batch_size, 1, 1)


class FeedForward(nn.Module):
    """
    Point-wise feed-forward layer is implemented by two dense layers.
    Args:
        input_tensor (torch.Tensor): the input of the point-wise feed-forward layer
    Returns:
        hidden_states (torch.Tensor): the output of the point-wise feed-forward layer
    """

    def __init__(self, hidden_size, inner_size, hidden_dropout_prob, layer_norm_eps=0.00001):
        super().__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = nn.LeakyReLU()

        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor) # skip

        return hidden_states
    
    
# class FeedForwardSwiGLU(nn.Module):
#     '''
#     MLP with SwiGLU
#     https://zhuanlan.zhihu.com/p/650237644
#     '''
#     def __init__(self, hidden_size: int, inner_size: int, multiple_of: int, hidden_dropout_prob: float):
#         super().__init__()
#         inner_size = multiple_of * ((2 * inner_size // 3 + multiple_of - 1) // multiple_of)
#         self.w1 = nn.Linear(hidden_size, inner_size)
#         self.w2 = nn.Linear(inner_size, hidden_size)
#         self.w3 = nn.Linear(hidden_size, inner_size)
#         self.dropout = nn.Dropout(hidden_dropout_prob)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # 
#         return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Qwen2MLPSwiGLU(nn.Module):
    '''
    MLP with SwiGLU
    Modified from transformers.models.mistral.modeling_mistral.MistralMLP with Mistral->Qwen2
    input in_size tensor, output out_size tensor
    
    '''
    def __init__(self, mid_emb_size, in_size, out_size):
        super().__init__()

        self.gate_proj = nn.Linear(in_size, mid_emb_size, bias=False)
        self.up_proj = nn.Linear(in_size, mid_emb_size, bias=False)
        self.down_proj = nn.Linear(mid_emb_size, out_size, bias=False)
        self.size = out_size

        # # 
        # self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        # self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, x):
        output = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        
        # # 
        # output = self.dropout(output)
        # output = self.LayerNorm(output + x) # skip
        return output



class MultiHeadAttention(nn.Module):

    def __init__(self, n_heads, hidden_size, hidden_dropout_prob, layer_norm_eps):
        super().__init__()
        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=n_heads, batch_first=True
        )
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_tensor,  attention_mask):
        attention_output, _ = self.multi_head_attention(
            query=input_tensor, key=input_tensor, value=input_tensor,
            key_padding_mask=attention_mask,  # ignore padded places with True
            need_weights=False
        )

        attention_output = self.dropout(attention_output)
        return self.LayerNorm(input_tensor + attention_output)


class TransformerLayer(nn.Module):
    """
    One transformer layer consists of a multi-head self-attention layer and a point-wise feed-forward layer.
    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer
    Returns:
        feedforward_output (torch.Tensor): The output of the point-wise feed-forward sublayer,
                                           is the output of the transformer layer.
    """

    def __init__(
        self, n_heads, hidden_size, intermediate_size, hidden_dropout_prob, layer_norm_eps
    ):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(
            n_heads, hidden_size, hidden_dropout_prob, layer_norm_eps)
        self.feed_forward = FeedForward(
            hidden_size, intermediate_size, hidden_dropout_prob, layer_norm_eps)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.multi_head_attention(
            hidden_states, attention_mask)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output


class TransformerEncoder(nn.Module):
    r""" One TransformerEncoder consists of several TransformerLayers.
    Args:
        n_layers(num): num of transformer layers in transformer encoder. Default: 1
        n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        hidden_size(num): the input and output hidden size. Default: 64
        inner_size(num): the dimensionality in feed-forward layer. Default: 256
        hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.2
        layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12
    """

    def __init__(
        self,
        n_layers=1,
        n_heads=2,
        hidden_size=60,
        inner_size=64,
        hidden_dropout_prob=0.2,
        layer_norm_eps=1e-8
    ):

        super().__init__()
        layer = TransformerLayer(
            n_heads, hidden_size, inner_size, hidden_dropout_prob, layer_norm_eps
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                   for _ in range(n_layers)])

    def forward(self, hidden_states, attention_mask):
        """
        Args
            hidden_states (torch.Tensor): the input of the TransformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output
        Returns:
        """
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)

        return hidden_states


class infoNCE(nn.Module):

    def __init__(self, temp_init, hdim):
        super().__init__()
        self.temp = nn.Parameter(torch.ones([]) * temp_init)

        self.weight_matrix = nn.Parameter(torch.randn((hdim, hdim)))
        nn.init.xavier_normal_(self.weight_matrix)

        self.tanh = nn.Tanh()

    def calculate_loss(self, query, item, neg_item):

        positive_logit = torch.sum(
            (query @ self.weight_matrix) * item, dim=1, keepdim=True)
        negative_logits = (
            query @ self.weight_matrix) @ neg_item.transpose(-2, -1)

        positive_logit, negative_logits = self.tanh(
            positive_logit), self.tanh(negative_logits)

        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(
            len(logits), dtype=torch.long, device=query.device)

        return F.cross_entropy(logits / self.temp, labels, reduction='mean')

    def forward(self, query, click_item, neg_item, neg_query):
        '''
        Args:
            query: matrix (#item, dim) 
            click item: matrix (#item, dim)
            neg item: matrix (#neg, dim)
            neg query: matrix (#neg, dim)

        Returns: loss
            infoNCE loss: (,)
        '''

        query_loss = self.calculate_loss(query, click_item, neg_item)
        item_loss = self.calculate_loss(click_item, query, neg_query)

        return 0.5 * (query_loss + item_loss)


class feature_align(nn.Module):
    def __init__(self, temp_init, hdim):
        super().__init__()
        self.infoNCE_loss = infoNCE(temp_init, hdim)

    def filter_user_src_his(self, qry_his_emb, click_item_mask, click_item_emb):
        '''process data to construct query-clicking item pairs.
            For issued queries, expand query embeddings to all their clicking items and filter out paddings.
            For clicked items, filter out paddings

        Args:
            qry_his_emb: (B, seq_len, dim)
            click_item_mask: (B, seq_len, max_click_item_num)
            click_item_emb: (B, seq_len, max_click_item_num, dim)

        Returns:
            src_his_query_emb, src_his_click_item_emb: (#item, dim)
        '''

        qry_his_emb = qry_his_emb.unsqueeze(
            2).expand(-1, -1, click_item_mask.size(2), -1)

        src_his_query_emb = torch.masked_select(
            qry_his_emb.clone(), click_item_mask.unsqueeze(-1)).reshape(-1, qry_his_emb.size(-1))
        src_his_click_item_emb = torch.masked_select(click_item_emb.clone(), click_item_mask.unsqueeze(-1))\
            .reshape(-1, click_item_emb.size(-1))

        return src_his_query_emb, src_his_click_item_emb

    def forward(self, align_loss_input, query_emb, click_item_mask, q_click_item_emb):
        neg_item_emb, neg_query_emb = align_loss_input
        src_his_query_emb, src_his_click_item_emb = self.filter_user_src_his(
            query_emb, click_item_mask, q_click_item_emb)

        align_loss = self.infoNCE_loss(
            src_his_query_emb, src_his_click_item_emb, neg_item_emb, neg_query_emb)

        return align_loss


class Dice(nn.Module):
    """The Data Adaptive Activation Function in DIN,which can be viewed as a generalization of PReLu and can adaptively adjust the rectified point according to distribution of input data.
    Input shape:
        - 2 dims: [batch_size, embedding_size(features)]
        - 3 dims: [batch_size, num_features, embedding_size(features)]
    Output shape:
        - Same shape as input.
    References
        - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
        - https://github.com/zhougr1993/DeepInterestNetwork, https://github.com/fanoping/DIN-pytorch
    """

    def __init__(self, emb_size, dim=2, epsilon=1e-8):
        super(Dice, self).__init__()
        assert dim == 2 or dim == 3

        self.bn = nn.BatchNorm1d(emb_size, eps=epsilon)
        self.sigmoid = nn.Sigmoid()
        self.dim = dim

        # wrap alpha in nn.Parameter to make it trainable
        if self.dim == 2:
            self.alpha = nn.Parameter(torch.zeros((emb_size,)))
        else:
            self.alpha = nn.Parameter(torch.zeros((emb_size, 1)))

    def forward(self, x):
        assert x.dim() == self.dim
        if self.dim == 2:
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
        else:
            x = torch.transpose(x, 1, 2)
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
            out = torch.transpose(out, 1, 2)
        return out


class FullyConnectedLayer(nn.Module):
    def __init__(self, input_size, hidden_unit, batch_norm=False, activation='relu', sigmoid=False, dropout=None, layer_norm=False,
                 dice_dim=None):
        super(FullyConnectedLayer, self).__init__()
        assert len(hidden_unit) >= 1
        self.sigmoid = sigmoid

        layers = []
        layers.append(nn.Linear(input_size, hidden_unit[0]))

        for i, h in enumerate(hidden_unit[:-1]):
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_unit[i]))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_unit[i]))

            if activation.lower() == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation.lower() == 'tanh':
                layers.append(nn.Tanh())
            elif activation.lower() == 'leakyrelu':
                layers.append(nn.LeakyReLU())
            elif activation.lower() == 'prelu':
                layers.append(nn.PReLU())
            elif activation.lower() == 'dice':
                assert dice_dim
                layers.append(Dice(hidden_unit[i], dim=dice_dim))
            else:
                raise NotImplementedError

            if dropout is not None:
                layers.append(nn.Dropout(dropout))

            layers.append(nn.Linear(hidden_unit[i], hidden_unit[i+1]))

        self.fc = nn.Sequential(*layers)
        if self.sigmoid:
            self.output_layer = nn.Sigmoid()

    def forward(self, x):
        return self.output_layer(self.fc(x)) if self.sigmoid else self.fc(x)


class AttentionSequencePoolingLayer(nn.Module):
    def __init__(self, embedding_dim=4, activation='dice'):
        super(AttentionSequencePoolingLayer, self).__init__()

        self.local_att = LocalActivationUnit(
            hidden_unit=[64, 16], embedding_dim=embedding_dim, batch_norm=False, activation=activation)

    def forward(self, query_ad, user_behavior, mask=None):
        # query ad            : size -> batch_size * num_items * dim
        # user behavior       : size -> batch_size * seq_len * dim
        # mask                : size -> batch_size * seq_len
        # output              : size -> batch_size * num_items * dim

        # [batch_size, seq_len, num_items]
        attention_score = self.local_att(query_ad, user_behavior)

        # [batch_size, num_items, seq_len]
        attention_score = torch.transpose(attention_score, 1, 2)

        if mask is not None:
            attention_score = attention_score.masked_fill(
                mask.unsqueeze(1), torch.tensor(0))

        # multiply weight
        # [batch_size,num_items,dim]
        output = torch.matmul(attention_score, user_behavior)

        return output


class LocalActivationUnit(nn.Module):
    def __init__(self, hidden_unit=[80, 40], embedding_dim=4, batch_norm=False, activation='dice'):
        super(LocalActivationUnit, self).__init__()
        self.fc1 = FullyConnectedLayer(input_size=4*embedding_dim,
                                       hidden_unit=hidden_unit,
                                       batch_norm=batch_norm,
                                       sigmoid=False,
                                       activation=activation,
                                       dice_dim=3)

        self.fc2 = nn.Linear(hidden_unit[-1], 1)

    # @torchsnooper.snoop()
    def forward(self, query: torch.Tensor, user_behavior: torch.Tensor):
        # query ad            : size -> batch_size * num_items * dim
        # user behavior       : size -> batch_size * seq_len * dim
        batch_size, num_items, embed_size = query.size()
        _, seq_len, _ = user_behavior.size()

        queriy_expand = query.unsqueeze(1).expand(-1, seq_len, -1, -1)
        user_behavior_expand = user_behavior.unsqueeze(
            2).expand(-1, -1, num_items, -1)

        attention_input = torch.cat([queriy_expand, user_behavior_expand, queriy_expand-user_behavior_expand,
                                    queriy_expand*user_behavior_expand], dim=-1)  # as the source code, subtraction simulates verctors' difference

        attention_output = self.fc1(attention_input.reshape(
            (batch_size, seq_len*num_items, -1)))
        # [batch_size, seq_len, num_items, 1]
        attention_score = self.fc2(attention_output)

        return attention_score.squeeze().reshape((batch_size, seq_len, num_items))


class Target_Attention(nn.Module):
    def __init__(self, hid_dim1, hid_dim2):
        super().__init__()

        self.W = nn.Parameter(torch.randn((1, hid_dim1, hid_dim2)))
        nn.init.xavier_normal_(self.W)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, seq_emb, target, mask):
        '''
        Args:
            seq_emb: batch, seq_length, dim1
            target: batch, num_items, dim2
            mask: batch, seq_length. True means padding
        '''

        score = torch.matmul(seq_emb, self.W)  # batch, seq, dim2
        # batch, seq, num_items
        score = torch.matmul(score, target.transpose(-2, -1))

        all_score = score.masked_fill(mask.unsqueeze(-1), torch.tensor(-1e16))
        all_weight = self.softmax(
            all_score.transpose(-2, -1))  # batch,num_items,seq
        all_vec = torch.matmul(all_weight, seq_emb)  # batch, num_items, dim1

        return all_vec


class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True, batch_norm=False):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            if batch_norm:
                layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)


class MMoELayer(torch.nn.Module):
    """
    A pytorch implementation of MMoE Model.

    Reference:
        Ma, Jiaqi, et al. Modeling task relationships in multi-task learning with multi-gate mixture-of-experts. KDD 2018.
    """

    def __init__(self, embed_dim, bottom_mlp_dims, tower_mlp_dims, task_num, expert_num, dropout):
        super().__init__()
        self.embed_output_dim = embed_dim
        self.task_num = task_num
        self.expert_num = expert_num

        self.expert = torch.nn.ModuleList([MultiLayerPerceptron(
            self.embed_output_dim, bottom_mlp_dims, dropout, output_layer=False) for i in range(expert_num)])
        self.tower = torch.nn.ModuleList([MultiLayerPerceptron(
            bottom_mlp_dims[-1], tower_mlp_dims, dropout) for i in range(task_num)])
        self.gate = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(
            self.embed_output_dim, expert_num), torch.nn.Softmax(dim=1)) for i in range(task_num)])

    def forward(self, emb):
        gate_value = [self.gate[i](emb).unsqueeze(1)
                      for i in range(self.task_num)]
        fea = torch.cat([self.expert[i](emb).unsqueeze(1)
                        for i in range(self.expert_num)], dim=1)
        task_fea = [torch.bmm(gate_value[i], fea).squeeze(1)
                    for i in range(self.task_num)]

        results = [torch.sigmoid(self.tower[i](task_fea[i]).squeeze(1))
                   for i in range(self.task_num)]
        return results


class PLE_layer(nn.Module):
    def __init__(self, orig_input_dim, bottom_mlp_dims, tower_mlp_dims, task_num, shared_expert_num, specific_expert_num, dropout) -> None:
        super().__init__()
        self.embed_output_dim = orig_input_dim
        self.task_num = task_num
        self.shared_expert_num = shared_expert_num
        self.specific_expert_num = specific_expert_num
        self.layers_num = len(bottom_mlp_dims)

        self.batch_norm = False
        logging.info("PLE_layer batch_norm:{}".format(self.batch_norm))
        logging.info("PLE_layer bottom_mlp_dims:{}".format(bottom_mlp_dims))
        logging.info("PLE_layer tower_mlp_dims:{}".format(tower_mlp_dims))
        logging.info("PLE_layer shared_expert_num:{}".format(shared_expert_num))
        logging.info("PLE_layer specific_expert_num:{}".format(specific_expert_num))

        self.task_experts = [
            [0] * self.task_num for _ in range(self.layers_num)]
        self.task_gates = [[0] * self.task_num for _ in range(self.layers_num)]
        self.share_experts = [0] * self.layers_num
        self.share_gates = [0] * self.layers_num
        for i in range(self.layers_num):
            input_dim = self.embed_output_dim if 0 == i else bottom_mlp_dims[i - 1]
            self.share_experts[i] = torch.nn.ModuleList([MultiLayerPerceptron(input_dim, [
                                                        bottom_mlp_dims[i]], dropout, output_layer=False, batch_norm=self.batch_norm) for k in range(self.shared_expert_num)])
            self.share_gates[i] = torch.nn.Sequential(torch.nn.Linear(
                input_dim, shared_expert_num + task_num * specific_expert_num), torch.nn.Softmax(dim=1))
            for j in range(task_num):
                self.task_experts[i][j] = torch.nn.ModuleList([MultiLayerPerceptron(input_dim, [
                                                              bottom_mlp_dims[i]], dropout, output_layer=False, batch_norm=self.batch_norm) for k in range(self.specific_expert_num)])
                self.task_gates[i][j] = torch.nn.Sequential(torch.nn.Linear(
                    input_dim, shared_expert_num + specific_expert_num), torch.nn.Softmax(dim=1))
            self.task_experts[i] = torch.nn.ModuleList(self.task_experts[i])
            self.task_gates[i] = torch.nn.ModuleList(self.task_gates[i])

        self.task_experts = torch.nn.ModuleList(self.task_experts)
        self.task_gates = torch.nn.ModuleList(self.task_gates)
        self.share_experts = torch.nn.ModuleList(self.share_experts)
        self.share_gates = torch.nn.ModuleList(self.share_gates)

        self.tower = torch.nn.ModuleList([MultiLayerPerceptron(
            bottom_mlp_dims[-1], tower_mlp_dims, dropout, output_layer=False, batch_norm=self.batch_norm) for i in range(task_num)])

    def forward(self, emb):
        task_fea = [emb for i in range(self.task_num + 1)]
        for i in range(self.layers_num):
            share_output = [expert(task_fea[-1]).unsqueeze(1)
                            for expert in self.share_experts[i]]
            task_output_list = []
            for j in range(self.task_num):
                task_output = [expert(task_fea[j]).unsqueeze(1)
                               for expert in self.task_experts[i][j]]
                task_output_list.extend(task_output)
                mix_ouput = torch.cat(task_output+share_output, dim=1)
                gate_value = self.task_gates[i][j](task_fea[j]).unsqueeze(1)
                task_fea[j] = torch.bmm(gate_value, mix_ouput).squeeze(1)
            if i != self.layers_num-1:  # share expert 
                gate_value = self.share_gates[i](task_fea[-1]).unsqueeze(1)
                mix_ouput = torch.cat(task_output_list + share_output, dim=1)
                task_fea[-1] = torch.bmm(gate_value, mix_ouput).squeeze(1)

        results = [self.tower[i](task_fea[i]).squeeze(1)
                   for i in range(self.task_num)]
        return results
