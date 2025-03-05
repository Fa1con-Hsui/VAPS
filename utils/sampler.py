import logging
import math
import random
from typing import List

import numpy as np
import pandas as pd

from utils import const, utils



def get_pad_negs_for_train(neg_item_list:List[int], neg_items_for_pad:List[int], num_negs):
        random.shuffle(neg_item_list)
        
        neg_item_list = neg_item_list[: num_negs]
        if len(neg_item_list) < num_negs:
            assert neg_items_for_pad and \
                len(neg_items_for_pad) > num_negs-len(neg_item_list)
            neg_item_list += random.sample(neg_items_for_pad, 
                                           num_negs-len(neg_item_list))
        return neg_item_list



def pad_seq(seq, max_len, is_seq_pad_left, pad_item):
    if len(seq) < max_len:
        if is_seq_pad_left:
            seq = [pad_item] * (max_len - len(seq)) + seq
        else:
            seq += [pad_item] * (max_len - len(seq))
    return seq


class Sampler(object):
    def __init__(self, args, is_search, train_mode, 
                 **kwargs) -> None:
        self.args = args
        self.num_negs = args.num_negs
        self.num_candidates = args.num_candidates
        self.is_seq_pad_left = args.is_seq_pad_left
        self.is_search = is_search
        self.train_mode = train_mode
        self.user_vocab = const.user_vocab
        self.preSampleNeg4Infer = args.preSampleNeg4Infer
        self.user_get_all_his_unisar = args.user_get_all_his_unisar
        self.aug_his = args.aug_his
        
        assert train_mode in ['train', 'test', 'val']

        self.data = None # df
        if self.is_search:
            if self.train_mode == 'train':
                self.data = const.src_train
            elif self.train_mode == 'test':
                self.data = const.src_test
            elif self.train_mode == 'val':
                self.data = const.src_val
        else:
            if self.train_mode == 'train':
                self.data = const.rec_train
            elif self.train_mode == 'test':
                self.data = const.rec_test
            elif self.train_mode == 'val':
                self.data = const.rec_val
        
        if self.train_mode in args.data_dbg_str:
            self.data = self.data.head(10)

    def sample(self, index):
        feed_dict = {}
        line = self.data.iloc[index]

        user_id = int(line['user_id']) 
        feed_dict['user'] = user_id # NOTE int
        feed_dict['item'] = int(line['item_id']) # NOTE int
        
        # get feed_dict['neg_items']
        if 'PersonalWAB' in self.args.data:
            if self.train_mode == 'train':
                feed_dict['neg_items'] = random.sample(line['neg_items_all'], self.num_negs)
            else:
                if self.preSampleNeg4Infer:
                    if self.num_candidates > 0:
                        feed_dict['neg_items'] = line['neg_items_sampled'][:self.num_candidates]
                    else:
                        feed_dict['neg_items'] = line['neg_items_sampled']
                else:
                    if self.num_candidates > 0:
                        feed_dict['neg_items'] = line['neg_items_all'][:self.num_candidates]
                    else:
                        feed_dict['neg_items'] = line['neg_items_all']

        else:
            raise ValueError

        feed_dict['search'] = self.is_search # NOTE bool

        if self.is_search:
            feed_dict['src_session_id'] = int(line['search_session_id']) # NOTE int


            
        rec_his_num = int(line['rec_his']) # NOTE int
        src_session_his_num = int(line['src_session_his']) # NOTE int
        src_his_num = int(line['src_his'])
        conv_record_his_num = int(line['conv_record_his'])
        feed_dict['rec_his'] = self.get_his(user_id, 'rec_his', 
                                rec_his_num, 
                                max_len=const.max_rec_his_len,
                                pad_item=const.PAD_ITEM_ID) # List[int] rec items

        feed_dict['src_session_his'] = self.get_his(user_id, 'src_session_his', 
                                src_session_his_num, 
                                max_len=const.max_src_session_his_len,
                                pad_item=const.PAD_SEARCH_SESSION_ID) # List[int] of src session_ids

        feed_dict['src_his'] = self.get_his(user_id, 'src_his', 
                                src_his_num, 
                                max_len=const.max_src_his_len,
                                pad_item=const.PAD_ITEM_ID) # List[int] of src items
        
        feed_dict['conv_his'] = self.get_his(user_id, 'conv_record_his', 
                                conv_record_his_num, 
                                max_len=const.max_conv_his_len,
                                pad_item=const.PAD_CONV_REIDX) # List[int] of s
        
        
        if self.user_get_all_his_unisar:
            feed_dict.update( # New key: 'all_his_type', 'all_his_ts', 'all_his'; Value: List[int/float]
                self.get_all_his_unisar(user_id, rec_his_num, src_session_his_num))
        
        


        if self.aug_his != '' and self.train_mode == 'train':
            if 'src' in self.aug_his:
                aug_src_his_1 = self.aug_seq(feed_dict['src_his'],
                                                mask_token=const.item_mask_token)
                feed_dict['aug_src_his_1'] = pad_seq(aug_src_his_1, const.max_src_his_len,
                            self.is_seq_pad_left, pad_item=const.PAD_ITEM_ID) 

                aug_src_his_2 = self.aug_seq(feed_dict['src_his'],
                                                mask_token=const.item_mask_token)
                feed_dict['aug_src_his_2'] = pad_seq(aug_src_his_2, const.max_src_his_len,
                            self.is_seq_pad_left, pad_item=const.PAD_ITEM_ID) 
            if 'rec' in self.aug_his:
                aug_rec_his_1 = self.aug_seq(feed_dict['rec_his'],
                                                mask_token=const.item_mask_token)
                feed_dict['aug_rec_his_1'] = pad_seq(aug_rec_his_1, const.max_rec_his_len,
                            self.is_seq_pad_left, pad_item=const.PAD_ITEM_ID)
                aug_rec_his_2 = self.aug_seq(feed_dict['rec_his'],
                                                mask_token=const.item_mask_token)
                feed_dict['aug_rec_his_2'] = pad_seq(aug_rec_his_2, const.max_rec_his_len,
                            self.is_seq_pad_left, pad_item=const.PAD_ITEM_ID)
            if 'src_session' in self.aug_his:
                aug_src_session_his_1 = self.aug_seq(feed_dict['src_session_his'],
                                                mask_token=const.PAD_SEARCH_SESSION_ID)
                feed_dict['aug_src_session_his_1'] = pad_seq(aug_src_session_his_1, const.max_src_session_his_len,
                            self.is_seq_pad_left, pad_item=const.PAD_SEARCH_SESSION_ID)
                aug_src_session_his_2 = self.aug_seq(feed_dict['src_session_his'],
                                                mask_token=const.PAD_SEARCH_SESSION_ID)
                feed_dict['aug_src_session_his_2'] = pad_seq(aug_src_session_his_2, const.max_src_session_his_len,
                            self.is_seq_pad_left, pad_item=const.PAD_SEARCH_SESSION_ID)
            if 'conv' in self.aug_his:  
                aug_conv_his_1 = self.aug_seq(feed_dict['conv_his'],
                                                mask_token=const.PAD_CONV_REIDX)
                feed_dict['aug_conv_his_1'] = pad_seq(aug_conv_his_1, const.max_conv_his_len,
                            self.is_seq_pad_left, pad_item=const.PAD_CONV_REIDX) 
                aug_conv_his_2 = self.aug_seq(feed_dict['conv_his'],
                                                mask_token=const.PAD_CONV_REIDX)
                feed_dict['aug_conv_his_2'] = pad_seq(aug_conv_his_2, const.max_conv_his_len,
                            self.is_seq_pad_left, pad_item=const.PAD_CONV_REIDX)             
    
            
        feed_dict['pairwise'] = True
        return feed_dict
    
    def get_his(self, user_id, his_name, his_num, max_len, pad_item):
        
        his = self.user_vocab[user_id][his_name][:his_num][-max_len:] 
        his = pad_seq(his, max_len,
                        self.is_seq_pad_left, pad_item=pad_item) 
        return his
    def get_all_his_unisar(self, user_id, rec_his_num, src_his_num):

        rec_his_item :list = self.get_his(user_id, 'rec_his', rec_his_num, 
                        max_len=const.max_rec_his_len,
                        pad_item=const.PAD_ITEM_ID)
        rec_his_ts :list  = self.get_his(user_id, 'rec_his_ts', rec_his_num, 
                        max_len=const.max_rec_his_len,
                        pad_item=np.inf)
        rec_his_type = [1] * len(rec_his_item)
        rec_his = list(zip(rec_his_item, rec_his_ts, rec_his_type)) # [(item_id, ts, type),(item_id, ts, type),...]

        src_his_item = self.get_his(user_id, 'src_session_his', src_his_num, 
                        max_len=const.max_src_session_his_len,
                        pad_item=const.PAD_SEARCH_SESSION_ID)
        src_his_ts = self.get_his(user_id, 'src_session_his_ts', src_his_num, 
                        max_len=const.max_src_session_his_len,
                        pad_item=np.inf)
        
        src_his_type = [2] * len(src_his_item)
        src_his = list(zip(src_his_item, src_his_ts, src_his_type)) # [(search_session_id, ts, type),(search_session_id, ts, type),...]

        all_his = rec_his + src_his

        sorted_all_his = sorted(all_his, key=lambda x: x[1])
        sorted_all_his_item = [x[0] for x in sorted_all_his]
        sorted_all_his_time = [x[1] for x in sorted_all_his]
        sorted_all_his_type = [x[2] for x in sorted_all_his]

        return {
            "all_his": sorted_all_his_item,
            "all_his_ts": sorted_all_his_time,
            "all_his_type": sorted_all_his_type
        }

    def aug_seq(self, seqs, mask_token):
        seqs_len = len(seqs)
        if seqs_len > 1:
            aug_type = random.choice(range(3)) # 
            if aug_type == 0:
                num_left = math.floor(seqs_len * self.args.crop_ratio)
                crop_begin = random.randint(0, seqs_len - num_left)
                aug_seqs = seqs[crop_begin:crop_begin + num_left]
            elif aug_type == 1:
                num_mask = math.floor(seqs_len * self.args.mask_ratio)
                mask_index = random.sample(range(seqs_len), k=num_mask)
                aug_seqs = []
                for i in range(seqs_len):
                    if i in mask_index:
                        aug_seqs.append(mask_token)
                    else:
                        aug_seqs.append(seqs[i])
            elif aug_type == 2:
                num_reorder = math.floor(seqs_len * self.args.reorder_ratio)
                reorder_begin = random.randint(0, seqs_len - num_reorder)
                aug_seqs = seqs[:]
                sub_seqs = seqs[reorder_begin:reorder_begin + num_reorder]
                random.shuffle(sub_seqs)
                aug_seqs[reorder_begin:reorder_begin + num_reorder] = sub_seqs
            return aug_seqs
        else:
            return seqs

