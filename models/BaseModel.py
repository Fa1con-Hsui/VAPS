import logging
import os
from typing import Dict

import torch
import torch.nn as nn

from utils import const, utils
from collections import deque
from .Inputs import *


class BaseModel(nn.Module):
    ParameterDict = {
        'use_src_session': None,
        'use_rec_his': None,
        'use_src_his': None,
        'use_src_session_his': None,
        'use_ts': None,
    }
    runner = None
    dataset = {}
    saved_epochs = deque()

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--model_path',
                            type=str,
                            default='',
                            help='Model save path.')
        parser.add_argument('--dropout', type=float, default=0.1)

        return parser

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.device = args.device
        self.model_path = args.model_path
        self.dropout = args.dropout
        self.saved_epochs_num = args.saved_epochs_num

        # logging.info("load: {}".format(const.user_map_vocab))
        # user_map_vocab = utils.load_pickle(const.user_map_vocab)
        # self.user_map_vocab = {
        #     k: torch.from_numpy(v).to(self.device)
        #     for k, v in user_map_vocab.items()
        # }  # const.user_vocab_np

        # logging.info("load: {}".format(const.item_map_vocab))
        # item_map_vocab = utils.load_pickle(const.item_map_vocab)
        # self.item_map_vocab = {
        #     k: torch.from_numpy(v).to(self.device)
        #     for k, v in item_map_vocab.items()
        # } # const.item_vocab_np
 
        # add the mask token map
        # self.item_map_vocab = self.add_mask_token(self.item_map_vocab)

        # logging.info("load: {}".format(const.session_map_vocab))
        # session_map_vocab = utils.load_pickle(const.session_map_vocab)
        # self.session_map_vocab = {
        #     k: torch.from_numpy(v).to(self.device)
        #     for k, v in session_map_vocab.items()
        # } # const.src_session_vocab_np

        # add the mask token map
        # self.session_map_vocab = self.add_mask_token(self.session_map_vocab)
        
        # XXX Author 
        # if args.use_moev1:
        #     self.text_embedding = TextFeat_MoEv1(self.device)
        # else:
        #     self.text_embedding = TextFeat(self.device)
        
        
        self.llm_text_embedding = None
        if args.use_extra_llm_emb:
            if args.switch2MoEv3:
                self.llm_text_embedding = TextFeat_MoEv3(self.device, use_llm=True, use_moe=args.use_moev1)
            elif args.switch2MoEv2:
                self.llm_text_embedding = TextFeat_MoEv2(self.device, use_llm=True, use_moe=args.use_moev1)
            else:
                self.llm_text_embedding = TextFeat_MoEv1(self.device, use_llm=True, use_moe=args.use_moev1) 
        
        if args.switch2MoEv3:
            self.text_embedding = TextFeat_MoEv3(self.device, 
                                             use_llm=const.args.use_llm_token_emb_mode, 
                                             use_moe=args.use_moev1)    
        elif args.switch2MoEv2:
            self.text_embedding = TextFeat_MoEv2(self.device, 
                                             use_llm=const.args.use_llm_token_emb_mode, 
                                             use_moe=args.use_moev1)    
        else:
            self.text_embedding = TextFeat_MoEv1(self.device, 
                                             use_llm=const.args.use_llm_token_emb_mode, 
                                             use_moe=args.use_moev1)  
            
        self.user_embedding = ObjectFeat(
            const.user_vocab_np, self.text_embedding,
            category_features=const.user_category_features,
            text_features=const.user_text_features,
            category_features_num=const.user_category_features_num,
            active_features=const.active_features_dict['user'],
            padding_idx=0 if 'PersonalWAB' in const.args.data else None,
            list_category_features = const.list_category_features,
        )
        self.item_embedding = ObjectFeat(
            const.item_vocab_np, self.text_embedding,
            category_features=const.item_category_features,
            text_features=const.item_text_features,
            category_features_num=const.item_category_features_num,
            active_features=const.active_features_dict['item'],
            padding_idx=0,
            list_category_features = const.list_category_features,
            
            llm_text_features=args.llm_text_features if args.use_extra_llm_emb else [], 
            llm_map_vocab=const.llm_process_text_data['item_vocab_np'] if args.use_extra_llm_emb else None,
            llm_text_emb_lay=self.llm_text_embedding if args.use_extra_llm_emb else None,
        )
        self.conv_embedding = ObjectFeat(
            const.conv_vocab_np, self.text_embedding,
            category_features=const.conv_category_features,
            text_features=const.conv_text_features,
            category_features_num=const.conv_category_features_num,
            active_features=const.active_features_dict['conv'],
            padding_idx=0,
            list_category_features = const.list_category_features,
        )
        # for UniSAR, NOT USERD USUALLY
        # self.session_embedding = SrcSessionFeat(
        #     query_feat=self.text_embedding,
        #     item_feat=self.item_embedding,
        #     user_feat=self.user_embedding,
        #     map_vocab=const.src_session_vocab_np)
        
        # for CS
        self.conv_src_session_embedding = ConvSrcSessionFeat(
            query_feat=self.text_embedding,
            item_feat=self.item_embedding,
            user_feat=self.user_embedding,
            conv_feat=self.conv_embedding,
            map_vocab=const.src_session_vocab_np)
    

        logging.info("final emb size:{}".format(const.final_emb_size))
        self.user_size = const.final_emb_size
        self.item_size = const.final_emb_size
        self.query_size = const.final_emb_size
        self.text_size = const.final_emb_size
        self.conv_size = const.final_emb_size

        self.query_item_alignment = False
        
        
        self.temp_log = nn.Parameter(torch.ones([]) * np.log(1/const.args.InfoNCE_kw2item_temp)).to(self.device)

    def align_loss(self,inputs):
        src_session_id = inputs['src_session_id'] # (bs,)
        pos_item, neg_items = inputs['item'], inputs['neg_items'] 
        
        if const.args.use_moev1 and (const.args.switch2MoEv2 or const.args.switch2MoEv3) and const.args.use_item_moe:
            query_emb, query_ori_emb = self.conv_src_session_embedding\
                .get_query_emb_based_on_src_session_id(src_session_id, use_text_moe=False, output_word_dim_emb=True)
            query_emb = query_emb.unsqueeze(1)
            query_ori_emb = query_ori_emb.unsqueeze(1).unsqueeze(1)
            
            items = torch.cat([pos_item.unsqueeze(1), neg_items], dim=1) # bs, 1+num_negs
            query_ori_emb_4items = torch.repeat_interleave(query_ori_emb, repeats=items.size(1), dim=1)
            items_emb = self.conv_src_session_embedding.get_item_emb(items,use_text_moe=True,
                                                                     moe_extra_query=query_ori_emb_4items) # (bs, 1+num_negs, d)
            
            item_llm_text_emb = self.item_embedding.get_llm_text_emb(items,use_text_moe=True,
                                                                    moe_extra_query=query_ori_emb_4items,) # (bs, 1+num_negs, d)
            item_text_emb = self.item_embedding.get_text_emb(items,use_text_moe=True,
                                                                    moe_extra_query=query_ori_emb_4items,) # (bs, 1+num_negs, d)
        else:
            if const.args.use_moev1 and not (const.args.switch2MoEv2 or const.args.switch2MoEv3):
                query_emb = self.conv_src_session_embedding\
                    .get_query_emb_based_on_src_session_id(src_session_id, use_text_moe=True).unsqueeze(1) 
            else:
                query_emb = self.conv_src_session_embedding\
                    .get_query_emb_based_on_src_session_id(src_session_id, use_text_moe=False).unsqueeze(1) 
            
            items = torch.cat([pos_item.unsqueeze(1), neg_items], dim=1) # bs, 1+num_negs
            items_emb = self.conv_src_session_embedding.get_item_emb(items) # (bs, 1+num_negs, d)
            
            
            item_llm_text_emb = self.item_embedding.get_llm_text_emb(items,use_text_moe=False) # (bs, 1+num_negs, d)
            item_text_emb = self.item_embedding.get_text_emb(items,use_text_moe=False) # (bs, 1+num_negs, d)
            
        #  [CLS]
        pos_features = torch.cat([
            query_emb,  # 
            items_emb[:, 0:1],  # 
            item_llm_text_emb[:, 0:1],  # LLM
            item_text_emb[:, 0:1]  # 
        ], dim=1)  # (bs,4,d)

        #  [NEG]
        neg_features = torch.cat([
            items_emb[:, 1:],  # 
            item_llm_text_emb[:, 1:],  # LLM
            item_text_emb[:, 1:]  # 
        ], dim=1)  # (bs,3*num_negs,d)

        if const.args.multi_align_emb_l2:
            pos_features = F.normalize(pos_features, p=2, dim=-1)
            neg_features = F.normalize(neg_features, p=2, dim=-1)

        all_features = torch.cat([pos_features, neg_features], dim=1)  # (bs,4+3N,d)
        sim_matrix = torch.matmul(pos_features, all_features.transpose(-1, -2))  # (bs,4,4+3N)

        batch_size, num_anchors = pos_features.shape[:2]
        device = sim_matrix.device
        
        pos_mask = ~torch.eye(num_anchors, dtype=torch.bool,device=device).unsqueeze(0)  # (1, num_anchors, num_anchors)
        pos_mask = pos_mask.expand(batch_size, -1, -1)  # (batch_size, num_anchors, num_anchors)
        zeros_part = torch.zeros(batch_size, num_anchors, 3*neg_items.shape[1],device=device).bool()
        pos_mask = torch.cat([pos_mask, zeros_part], dim=2)
        
        
        neg_mask = torch.cat([
            torch.zeros(batch_size, num_anchors, num_anchors, dtype=torch.bool, device=device),
            torch.ones(batch_size, num_anchors, 3*neg_items.shape[1], dtype=torch.bool, device=device)
        ], dim=2)

        temperature = const.args.multi_align_temperature
        pos_logits = sim_matrix[pos_mask].view(batch_size, num_anchors, -1)  # (bs,4,3)
        neg_logits = sim_matrix[neg_mask].view(batch_size, num_anchors, -1)  # (bs,4,3N)

        exp_pos = torch.exp(pos_logits / temperature).sum(2)
        exp_neg = torch.exp(neg_logits / temperature).sum(2)
        
        ratio = exp_pos / (exp_pos + exp_neg + 1e-8)
        if bool(torch.any(torch.isnan(ratio))):
            logging.error("ratio has nan")
            torch.save(dict(exp_neg=exp_neg.cpu(),exp_pos=exp_pos.cpu(),neg_logits=neg_logits.cpu(),
                            ratio=ratio.cpu(), pos_logits=pos_logits.cpu(), neg_mask=neg_mask.cpu(), pos_mask=pos_mask.cpu(),
                            query_emb=query_emb.cpu(), items_emb=items_emb.cpu(), item_llm_text_emb=item_llm_text_emb.cpu(),
                            item_text_emb=item_text_emb.cpu(), sim_matrix=sim_matrix.cpu(), pos_features=pos_features.cpu(),
                            neg_features=neg_features.cpu(), all_features=all_features.cpu(),
                ),'dbg.pth')
        
        ratio = torch.nan_to_num(ratio)
        contrastive_loss = -torch.log(ratio).mean()
        
        return contrastive_loss
    
    
    
    
    def kw2item_infonce_loss(self, inputs):
        
        # (B,num) (B,)
        kw2item_token_ids,kw2item_item_id = inputs['kw2item_token_ids'],inputs['kw2item_item_id']
        token_embs = self.text_embedding(kw2item_token_ids) # (B,D)
        item_embs = self.item_embedding(kw2item_item_id) # (B,D)
        batch_size = item_embs.size(0)
        
        # TODO l2
        if const.args.InfoNCE_kw2item_l2:
            token_embs = F.normalize(token_embs, p=2, dim=-1)  # (B, D)
            item_embs = F.normalize(item_embs, p=2, dim=-1)    # (B, D)
            
        # pdb.set_trace()
        
        # in-batch negative sampling
        if ('kw2item_neg' not in inputs or inputs['kw2item_neg'] is None) and \
                ('item2kw_neg' not in inputs or inputs['item2kw_neg'] is None):
            sim_matrix = torch.matmul(token_embs, item_embs.T) * self.temp_log.exp() # (B, B)
            labels = torch.arange(batch_size, device=self.device)  # (B,)
            loss_kw2item = F.cross_entropy(sim_matrix, labels)
            loss_item2kw = F.cross_entropy(sim_matrix.T, labels)  # 
            loss = const.args.loss_kw2item_lambda * loss_kw2item \
                + const.args.loss_item2kw_lambda * loss_item2kw
            return loss
        
        # sampler'kw2item_neg''item2kw_neg'
        else:
            labels = torch.zeros(batch_size, dtype=torch.long, 
                                device=self.device)
            
            loss_kw2item,loss_item2kw = None,None
            if 'kw2item_neg' in inputs and inputs['kw2item_neg'] is not None:
                neg_item_ids = inputs['kw2item_neg']  # (B, num_neg) item_id
                #  (B, 1+num_neg)
                logits = torch.einsum('bd,bnd->bn', 
                    token_embs, 
                    torch.cat([
                        item_embs.unsqueeze(1),  # (B,1,D)
                        self.item_embedding(neg_item_ids)  # (B,num_neg,D)
                    ], dim=1) # (B, 1+num_neg, D)
                ) * self.temp_log.exp()
                loss_kw2item = F.cross_entropy(logits, labels)
            if 'item2kw_neg' in inputs and inputs['item2kw_neg'] is not None:
                neg_kw_ids = inputs['item2kw_neg'] # (B, num_neg, token num)
                reverse_logits = torch.einsum('bd,bnd->bn',
                    item_embs,
                    torch.cat([
                        token_embs.unsqueeze(1), # (B,1,D)
                        self.text_embedding(neg_kw_ids) # (B, num_neg, D)
                    ], dim=1)
                ) * self.temp_log.exp()
                loss_item2kw = F.cross_entropy(reverse_logits, labels)
            if loss_item2kw and loss_kw2item:
                loss = const.args.loss_kw2item_lambda * loss_kw2item \
                    + const.args.loss_item2kw_lambda * loss_item2kw
            elif loss_item2kw:
                loss = loss_item2kw
            else:
                loss = loss_kw2item
                
        return loss
        
        
        

    def _init_weights(self):
        # weight initialization xavier_normal (a.k.a glorot_normal in keras, tf)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.Embedding):
                # BERTembedding
                continue
                # nn.init.xavier_normal_(m.weight.data)
            # elif isinstance(m, nn.Parameter):
            #     nn.init.xavier_normal_(m)

    def save_model(self, epoch, model_path=None):
        if model_path is None:
            model_path = self.model_path
            
        ckpt_path = os.path.join(model_path, "epoch_{}.pt".format(epoch))
       
        utils.check_dir(ckpt_path)
        logging.info("save model to: {}".format(ckpt_path))
        torch.save(self.state_dict(), ckpt_path)
        
        
        self.saved_epochs.append(epoch)
        if len(self.saved_epochs) == self.saved_epochs_num + 1:
            epoch_to_delete = self.saved_epochs.popleft()
            os.remove(os.path.join(model_path, "epoch_{}.pt".format(epoch_to_delete)))
            

    def load_model(self, ckpt_path=None, epoch=None):
        if ckpt_path is None:
            assert epoch is not None
            ckpt_path = os.path.join(self.model_path, "epoch_{}.pt".format(epoch))
        logging.info("load model from: {}".format(ckpt_path))
        self.load_state_dict(torch.load(ckpt_path, map_location=self.device, weights_only=True))

    def count_variables(self) -> int:
        total_parameters = 0
        for name, p in self.named_parameters():
            if p.requires_grad:
                num_p = p.numel()
                total_parameters += num_p
                logging.info("model.count_variables: name:{} size:{} num_parameters:{}".format(
                    name, p.size(), num_p))

        return total_parameters

    def customize_parameters(self) -> list:
        # customize optimizer settings for different parameters
        weight_p, bias_p = [], []
        for name, p in filter(lambda x: x[1].requires_grad,
                              self.named_parameters()):
            if 'bias' in name:
                bias_p.append(p)
            else:
                weight_p.append(p)
        optimize_dict = [{
            'params': weight_p
        }, {
            'params': bias_p,
            'weight_decay': 0
        }]
        return optimize_dict

    def warm_rec(self, freeze=True):
        raise NotImplementedError

    def warm_src(self, freeze=True):
        raise NotImplementedError

    def loss(self, inputs):
        if inputs['search']:
            return self.src_loss(inputs)
        else:
            return self.rec_loss(inputs)

    def predict(self, inputs):
        if inputs['search']:
            return self.src_predict(inputs)
        else:
            return self.rec_predict(inputs)

    def rec_loss(self, inputs):
        raise NotImplementedError

    def rec_predict(self, inputs):
        raise NotImplementedError

    # def src_w_neg_predict(self, inputs):
    #     raise NotImplementedError

    # def src_w_neg_loss(self, inputs):
    #     raise NotImplementedError

    def src_pair_loss(self, inputs):
        raise NotImplementedError

    def src_point_loss(self, inputs):
        raise NotImplementedError

    def src_loss(self, inputs):
        if inputs['pairwise']:
            return self.src_pair_loss(inputs)
        else:
            return self.src_point_loss(inputs)

    def src_pair_predict(self, inputs):
        raise NotImplementedError

    def src_point_predict(self, inputs):
        raise NotImplementedError

    def src_predict(self, inputs):
        if inputs['pairwise']:
            return self.src_pair_predict(inputs)
        else:
            return self.src_point_predict(inputs)

    # def sar_loss(self, inputs):
    #     raise NotImplementedError

    # def sar_predict(self, inputs):
    #     raise NotImplementedError


# class RecModel(BaseModel):
#     runner = 'RecRunner'

#     def __init__(self, args):
#         super().__init__(args)

# class SrcModel(BaseModel):
#     runner = 'SrcRunner'

#     def __init__(self, args):
#         super().__init__(args)

# class SarModel(BaseModel):
#     runner = 'SarRunner'

#     def __init__(self, args):
#         super().__init__(args)

#     def rec_predict(self, inputs):
#         pass

#     def rec_loss(self, inputs):
#         pass

#     def src_predict(self, inputs):
#         pass

#     def src_loss(self, inputs):
#         pass
