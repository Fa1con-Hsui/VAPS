import pdb
import torch
import torch.nn as nn
import logging
from utils import const

from ..BaseModel import BaseModel
from ..layers import PositionalEmbedding, FullyConnectedLayer, PositionalEmbedding_v2


class WeightedAverage3(nn.Module):
    def __init__(self, initial_weights=None, learable=False):
        super().__init__()
        if initial_weights is None:
            initial_weights = [1.0, 1.0, 1.0]
        
        self.log_weights = torch.log(torch.tensor(initial_weights, dtype=torch.float32))
        if learable:
            self.log_weights = nn.Parameter(self.log_weights)
    def forward(self, x1, x2, x3):
        weights = torch.softmax(self.log_weights, dim=0)
        return x1 * weights[0] + x2 * weights[1] + x3 * weights[2]


def ta_nan2valid(emb, valid_emb, mask):
    '''
    emb: (bs, 1, d)
    mask: (bs,) True nan
    '''
    # zeros = torch.zeros_like(query_emb)
    mask = mask.unsqueeze(-1).unsqueeze(-1)
    return  torch.where(mask, valid_emb, emb)


class TEM_CS_0206_Final(BaseModel):

    @staticmethod
    def parse_model_args(parser):
        # parser.add_argument('--runner', type=str, default="SrcRunner")

        parser.add_argument('--num_layers',
                            type=int,
                            default=1,
                            help='Number of self-attention layers.')
        parser.add_argument('--num_heads',
                            type=int,
                            default=2,
                            help='Number of attention heads.')

        parser.add_argument('--adhoc_weight', type=int, default=0)

        parser.add_argument('--pred_mlp', type=int, default=0)
        parser.add_argument('--pred_hid_units', type=str, default="200, 80, 1")
        
        
        # Conv:
        parser.add_argument('--conv_seq_pad_mode', type=str, default="left",
                            choices=['left', 'right'], 
                            help='Conv seq padding on left or right')
        parser.add_argument('--conv_num_layers', type=int, default=1)
        parser.add_argument('--src_session_num_layers', type=int, default=1)
        # wa_learable
        parser.add_argument('--wa_learable', type=int, default=0, help='wa_learable, ')
        parser.add_argument('--wa_initial_weights', type=str, default="1.0, 1.0, 1.0",
                            help='wa_initial_weights,'\
                            'query_emb, query_emb_src_session_attn_emb, query_emb_conv_attn_emb')

        # add_user_emb_mode
        parser.add_argument('--add_user_emb_mode', type=str, default="",
                            choices=['first','','uni_query_emb','seq_output'], 
                            help='user_emb')
        
        return BaseModel.parse_model_args(parser)

    def __init__(self, args):
        super().__init__(args)
        
        self.args = args
        self.use_preprocessed_conv_his_of_src_session = args.use_preprocessed_conv_his_of_src_session
        self.conv_seq_pad_mode = args.conv_seq_pad_mode
        self.conv_num_layers = args.conv_num_layers
        self.src_session_num_layers = args.src_session_num_layers
        self.wa_initial_weights = [float(k) for k in args.wa_initial_weights.split(',')]
        self.wa_learable = bool(args.wa_learable)
        
        self.adhoc_weight = args.adhoc_weight
        logging.info("adhoc_weight:{}".format(self.adhoc_weight))
        if self.adhoc_weight:
            self.src_weight_layer = nn.Sequential(
                nn.Linear(self.user_size + self.item_size + self.query_size,
                          128), nn.ReLU(), nn.Linear(128, 2),
                nn.Softmax(dim=-1))

        self.num_layers = args.num_layers
        self.num_heads = args.num_heads

        
        # emb
        self.item_pos_embedding = PositionalEmbedding(
            const.max_src_his_len + 1,
            self.item_size)
        
        if self.use_preprocessed_conv_his_of_src_session:
            max_conv_his_len = self.conv_src_session_embedding.map_vocab['conv_record_idx'].shape[-1]
        else: 
            max_conv_his_len = const.max_conv_his_len
        self.conv_pos_embedding = PositionalEmbedding_v2(
            max_conv_his_len + 1, 
            self.text_size)
        
        self.src_session_pos_embedding = PositionalEmbedding_v2(
            const.max_src_session_his_len + 1, 
            self.text_size)
        

        self.his_transformer_layer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.item_size,
                nhead=self.num_heads,
                dim_feedforward=self.item_size,
                dropout=self.dropout,
                batch_first=True), 
            num_layers=self.num_layers,)
        
        self.his_transformer_layer_conv = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.text_size,
                nhead=self.num_heads,
                dim_feedforward=self.text_size,
                dropout=self.dropout,
                batch_first=True), 
            num_layers=self.conv_num_layers,)
        
        self.his_transformer_layer_src_session = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.text_size,
                nhead=self.num_heads,
                dim_feedforward=self.text_size,
                dropout=self.dropout,
                batch_first=True), 
            num_layers=self.src_session_num_layers,)
        

        self.wa3 = WeightedAverage3(
            initial_weights=self.wa_initial_weights, learable=self.wa_learable,
        )
        
        self.pred_mlp = args.pred_mlp
        if self.pred_mlp:
            self.hidden_unit = [int(x) for x in args.pred_hid_units.split(',')]
            self.src_fc_layer = FullyConnectedLayer(
                input_size=2 * self.item_size + self.user_size + self.query_size,
                hidden_unit=self.hidden_unit,
                batch_norm=False,
                sigmoid=True,
                activation='relu',
                dropout=self.dropout)

        self.sigmoid = nn.Sigmoid()
        self.loss_fn = nn.BCELoss()

        self._init_weights()
        self.to(self.device)

   
    def pair_forward(self, inputs):
        user, src_his, pos_item, neg_items\
            = inputs['user'], inputs['src_his'], inputs['item'], inputs['neg_items'] 

        user_emb = self.conv_src_session_embedding.get_user_emb(user).unsqueeze(1)
        item_his_emb = self.conv_src_session_embedding.get_item_emb(src_his)
        batch_size = item_his_emb.size(0)
        
        # query_emb = self.conv_src_session_embedding.get_query_emb(inputs['query']).unsqueeze(1)
        # Author NOTE
 
        src_session_id = inputs['src_session_id'] # (bs,)
        
        if const.args.use_moev1 and (const.args.switch2MoEv2 or const.args.switch2MoEv3):
            # query
            query_emb, query_ori_emb = self.conv_src_session_embedding\
                .get_query_emb_based_on_src_session_id(src_session_id, 
                                                       use_text_moe=False,
                                                       output_word_dim_emb=True) 
            query_emb = query_emb.unsqueeze(1) # (bs, 1, dim)  
            query_ori_emb = query_ori_emb.unsqueeze(1).unsqueeze(1) # (bs, 1, 1, self.word_emb_siz)
            
            # conv
            if self.use_preprocessed_conv_his_of_src_session:
                conv_his = self.conv_src_session_embedding\
                    .map_vocab['conv_record_idx'][src_session_id] # (batch_size, conv_max_len)
            else:
                conv_his = inputs['conv_his']
            query_ori_emb_4conv=torch.repeat_interleave(query_ori_emb, repeats=conv_his.size(1), dim=1)
            conv_his_emb = self.conv_src_session_embedding\
                .get_conv_emb(conv_his, 
                              use_text_moe=const.args.use_moev1,
                              moe_extra_query=query_ori_emb_4conv) # (batch_size, conv_max_len, final_emb_size)
            if self.conv_seq_pad_mode == 'left': #  
                conv_his_emb = torch.flip(conv_his_emb, [1]) #(bs, conv_max_len, final_emb_size)
            else:
                conv_his_emb = conv_his_emb # 
                
            # src_session_his
            src_session_his = inputs['src_session_his'] # (bs, src_session_his_len)
            query_ori_emb_4src=torch.repeat_interleave(query_ori_emb, repeats=src_session_his.size(1), dim=1)
            src_session_his_emb = self.conv_src_session_embedding\
                .get_query_emb_based_on_src_session_id(src_session_his, 
                                                       use_text_moe=const.args.use_moev1,
                                                       moe_extra_query=query_ori_emb_4src) # (bs, src_session_his_len, dim)    
        else:
            # query
            query_emb = self.conv_src_session_embedding\
                .get_query_emb_based_on_src_session_id(src_session_id, use_text_moe=const.args.use_moev1).unsqueeze(1) # (bs, 1, dim)
            # conv
            if self.use_preprocessed_conv_his_of_src_session:
                conv_his = self.conv_src_session_embedding\
                    .map_vocab['conv_record_idx'][src_session_id] # (batch_size, conv_max_len)
            else:
                conv_his = inputs['conv_his']
            conv_his_emb = self.conv_src_session_embedding\
                .get_conv_emb(conv_his, use_text_moe=const.args.use_moev1) # (batch_size, conv_max_len, final_emb_size)
            if self.conv_seq_pad_mode == 'left': # 
                conv_his_emb = torch.flip(conv_his_emb, [1]) #(bs, conv_max_len, final_emb_size)
            else:
                conv_his_emb = conv_his_emb # 
            # src_session_his
            src_session_his = inputs['src_session_his'] # (bs, src_session_his_len)
            src_session_his_emb = self.conv_src_session_embedding\
                .get_query_emb_based_on_src_session_id(src_session_his, use_text_moe=const.args.use_moev1) # (bs, src_session_his_len, dim)
        
        
        if self.args.add_user_emb_mode == 'first':
            query_emb = query_emb + user_emb
            src_session_his_emb = src_session_his_emb + user_emb
            conv_his_emb = conv_his_emb + user_emb


        query_mask = torch.zeros((batch_size, 1), device=self.device, dtype=torch.bool) # False (bs,1)

        #  conv 
        # NOTE queryconv
        conv_his_query_emb = torch.cat(
            [query_emb, conv_his_emb], dim=1) # (bs, conv_max_len+1, final_emb_size)
        conv_his_mask = torch.cat(
            [query_mask, conv_his == self.conv_embedding.padding_idx], dim=1) # (bs, conv_max_len+1,)
        conv_his_query_emb += self.conv_pos_embedding(conv_his_query_emb)
        conv_his_query_emb = self.his_transformer_layer_conv(
            src=conv_his_query_emb, src_key_padding_mask=conv_his_mask) # (bs, conv_max_len+1, final_emb_size)
        seq_output_conv = conv_his_query_emb[:, 0, :].unsqueeze(1) # (bs, 1, final_emb_size)

        
        src_session_his_query_emb = torch.cat(
            [query_emb, src_session_his_emb], dim=1) 
        src_session_his_mask = torch.cat(
            [query_mask, src_session_his == const.PAD_SEARCH_SESSION_ID], dim=1) 
        src_session_his_query_emb += self.src_session_pos_embedding(src_session_his_query_emb)
        src_session_his_query_emb = self.his_transformer_layer_src_session(
            src=src_session_his_query_emb, src_key_padding_mask=src_session_his_mask)
        seq_output_src_session = src_session_his_query_emb[:, 0, :].unsqueeze(1) # (bs, 1, final_emb_size)
        
        uni_query_emb = self.wa3(query_emb, seq_output_conv, seq_output_src_session)
        
        
        if self.args.add_user_emb_mode == 'uni_query_emb':
            uni_query_emb = uni_query_emb + user_emb

        item_his_query_emb = torch.cat(
            [uni_query_emb, item_his_emb], dim=1)  
        item_his_mask = torch.cat(
            [query_mask, src_his == const.PAD_ITEM_ID], dim=1) 
        item_his_query_emb += self.item_pos_embedding(item_his_query_emb)
        item_his_query_emb = self.his_transformer_layer(
            src=item_his_query_emb, src_key_padding_mask=item_his_mask)
        seq_output = item_his_query_emb[:, 0, :].unsqueeze(1) # (bs, 1, dim)
        
        if self.args.add_user_emb_mode == 'seq_output':
            seq_output = seq_output + user_emb


        items = torch.cat([pos_item.unsqueeze(1), neg_items], dim=1) # bs, candidiate item num
        items_emb = self.conv_src_session_embedding.get_item_emb(items) # (bs, candidiate item num, dim)
        
        
        if self.pred_mlp:
            personalized_score = self.src_fc_layer(
                torch.cat([
                    seq_output.expand(-1, items.size(1), -1), items_emb,
                    query_emb.expand(-1, items.size(1), -1),
                    user_emb.expand(-1, items.size(1), -1)
                ], -1)) # (bs, candidiate item num, dim * 4) -> (bs, candidiate item num, 1)
        else:
            personalized_score = self.sigmoid(
                (seq_output * items_emb).sum(-1, keepdim=True)) # (bs, candidiate item num, 1)
        if self.adhoc_weight:
            ad_hoc_score = self.sigmoid(
                (query_emb * items_emb).sum(-1, keepdim=True)) # (bs, candidiate item num, 1)

            # logits = self.adhoc_weight * ad_hoc_score + \
            #     (1-self.adhoc_weight) * personalized_score
            logits = self.src_final_layer(
                torch.cat([personalized_score, ad_hoc_score], dim=-1))
        else:
            logits = personalized_score

        # pdb.set_trace()
    

        return logits.reshape((batch_size, -1))

    def src_pair_loss(self, inputs):
        logits = self.pair_forward(inputs)

        labels = torch.zeros_like(logits).to(self.device)
        labels[:, 0] = 1.0

        logits = logits.reshape((-1, ))
        labels = labels.reshape((-1, ))
        
        # align_loss
        
        loss = self.loss_fn(logits, labels)
        
        InfoNCE_loss = 0.0
        if const.args.InfoNCE_kw2item_alpha > 0.0:
            InfoNCE_loss = self.kw2item_infonce_loss(inputs)
        
        multi_align_loss = 0.0
        if const.args.multi_align_alpha > 0.0:
            multi_align_loss = self.align_loss(inputs)
            
        
        loss = loss * (1-const.args.InfoNCE_kw2item_alpha-const.args.multi_align_alpha) \
            + const.args.InfoNCE_kw2item_alpha * InfoNCE_loss \
            + const.args.multi_align_alpha * multi_align_loss

            # return {"total_loss": self.loss_fn(logits, labels) * (1-const.args.InfoNCE_kw2item_alpha) \
            #         +  const.args.InfoNCE_kw2item_alpha * self.kw2item_infonce_loss(inputs)}
            
        return {"total_loss": loss}
        # else:
        #     return {"total_loss": self.loss_fn(logits, labels)}

    def src_pair_predict(self, inputs):
        logits = self.pair_forward(inputs)

        return logits
