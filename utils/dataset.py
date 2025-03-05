import pdb
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from utils import const
from utils.sampler import *


class BaseDataSet(Dataset):
    def __init__(self) -> None:
        super().__init__()

    def __len__(self):
        return self.sampler.data.shape[0]

    def __getitem__(self, index) -> Any:
        return super().__getitem__(index)

    def collate_batch(self, feed_dicts: List[dict]) -> dict:
        '''
        Collate a batch according to the list of feed dicts, and pad the "align_neg_text"
        For 2-d list or 1-d list or int

        
        input:feed_dicts = [
            {"key0": 9, "key1": [9], "key2": [1, 2], "key3": [[1, 2]], 'k4': [[5,4]]},
            {"key0": 3, "key1": [3], "key2": [6], "key3": [[6]], 'k4': [[4],[7,7]]},
            {"key0": 8, "key1": [8], "key2": [5, 5], "key3": [[5, 5]], 'k4': [[0],[9,2,3]]},
        ]
        output :{'key0': tensor([9, 3, 8]), # (3,)
            'key1': tensor([[9],
                            [3],
                            [8]]),  # (3,1)
            'key2': tensor([[1, 2],
                            [6, 0],
                            [5, 5]]), # (3,2)
            'key3': tensor([[[1, 2]],
                            [[6, 0]],
                            [[5, 5]]]), # (3,1,2)
            'k4': tensor([[[5, 4, 0], [0, 0, 0]],
                            [[4, 0, 0], [7, 7, 0]],
                            [[0, 0, 0], [9, 2, 3]]])} # (3,2,3)
        '''
        # bool_key_dic = {'is_search':False}
        
        result = {} 
        for key in feed_dicts[0].keys():
            pad_id = const.PAD_TOKEN_ID
            if key in ['neg_items']:
                pad_id = const.PAD_ITEM_ID
            
            
            values = [d[key] for d in feed_dicts]
            if isinstance(values[0], bool):
                result[key] = values[0] 
            elif isinstance(values[0], int):
                result[key] = torch.tensor(values)
            elif isinstance(values[0], list):
                # float for timestamp (np.inf)
                if all(type(x) in [int, np.int32, np.int64, float] for x in values[0]):  # 2: {'key':[1,2]} {'key':[1]}
                    max_len = max(len(v) for v in values)
                    padded_values = []
                    for v in values:
                        if len(v) < max_len:
                            v.extend([pad_id] * (max_len - len(v)))
                        padded_values.append(v)
                    result[key] = torch.tensor(padded_values)
                elif all(isinstance(x, list) for x in values[0]): # 2: {'key':[[1,2]]} {'key':[[1],[2,3]]}
                    max_v_len = max(len(v) for v in values)
                    x_lens = [len(x) for v in values for x in v]
                    max_x_len = max(x_lens)
                    # print(max_v_len)
                    # print(max_x_len)
                    padded_sub_lists = []
                    for v in values:
                        padded_sub_list = []
                        for x in v:
                            if len(x) < max_x_len:
                                x.extend([pad_id] * (max_x_len - len(x)))
                            padded_sub_list.append(x)
                        
                        if len(v) < max_v_len:
                            padded_sub_list.extend([[pad_id] * max_x_len] * (max_v_len - len(v)))
                        padded_sub_lists.append(padded_sub_list)
                    try:
                        result[key] = torch.tensor(padded_sub_lists)
                    except:
                        raise ValueError(f"Can't be a tensor for key '{key}'\n\t{padded_sub_lists}")
                else:
                    raise ValueError(f"Unsupported data type for key '{key}'\n\t{type(values[0])}\n\t{values[0]}"
                        f"\n\t{[isinstance(x, int) for x in values[0]]}"
                        f"\n\t{[type(x) for x in values[0]]}"
                        )
            else:
                raise ValueError(f"Unsupported data type for key '{key}'\n\t{type(values[0])}\n\t{values[0]}")
        result['batch_size'] = len(feed_dicts)

        return result
    
    
   


class RecDataSet(BaseDataSet):
    def __init__(self, args, train_mode, **kwargs) -> None:
        super().__init__()
        self.sampler = Sampler(
                args=args, 
                train_mode=train_mode, 
                is_search=False, **kwargs)
    def __getitem__(self, index) -> Any:
        return self.sampler.sample(index)


class SrcDataSet(BaseDataSet):
    def __init__(self, args, train_mode, **kwargs) -> None:
        super().__init__()
        self.sampler = Sampler(
                args=args, 
                train_mode=train_mode, 
                is_search=True, **kwargs)

    def __getitem__(self, index) -> Any:
        return self.sampler.sample(index)


# class SAR_concat_DataSet(BaseDataSet):


class SAR_Random_DataSet(BaseDataSet):
    def __init__(self, args, train_mode='train', **kwargs) -> None:
        super().__init__()
        self.rec_sampler = Sampler(
                args=args, 
                train_mode=train_mode, 
                is_search=False, **kwargs)
        self.src_sampler =  Sampler(
                args=args, 
                train_mode=train_mode, 
                is_search=True, **kwargs)

        self.rec_len = self.rec_sampler.data.shape[0]
        self.src_len = self.src_sampler.data.shape[0]

    def __len__(self):
        return int(1e10)

    def __getitem__(self, index) -> Any:
        rec_index = random.randint(0, self.rec_len-1)
        src_index = random.randint(0, self.src_len-1)
        rec_data = self.rec_sampler.sample(rec_index)
        src_data = self.src_sampler.sample(src_index)
        sample_dict = {"rec": rec_data, "src": src_data}
        return sample_dict

    # Collate a batch according to the list of feed dicts
    def collate_batch(self, feed_dicts: List[Dict]) -> Dict:
        rec_feed_dicts = []
        src_feed_dicts = []
        for d in feed_dicts:
            rec_feed_dicts.append(d['rec'])
            src_feed_dicts.append(d['src'])
        result_dicts = {}
        result_dicts['rec'] = super().collate_batch(rec_feed_dicts)
        result_dicts['src'] = super().collate_batch(src_feed_dicts)
        return result_dicts




class KwItemInfoNCEDataset(BaseDataSet):
    def __init__(self, token_id_all_num) -> None:
        super().__init__()

        # 
        self.item_id_all = list(const.item_vocab.keys()) # 1-d list like: [1,2,3...]
        # 
        self.token_id_all = list(range(token_id_all_num)) # 1-d list like: [1,2,3...]
        self.item_id_all = [i for i in self.item_id_all if i != const.PAD_ITEM_ID]
        self.token_id_all = [i for i in self.token_id_all if i != const.PAD_TOKEN_ID]
        
        
        self.kw_item_data = const.kw_item_data
        self.token_map = const.token_map
        self.kw2items = self.kw_item_data['kw_items_dict']
        
        if hasattr(const.token_map, 'word2id'):
            self.kw2items = {k:v for k,v in self.kw2items.items() if k in self.token_map.word2id}
        
        #  token_ids_tuple: item_list
        self.token_ids_2_item_ids = {
            tuple(self._convert_kw_to_ids(kw)):v for kw,v in self.kw2items.items()}
      
        self.data_list = [] # Commercial_August_v0 len: 31133
        for token_ids_tuple, item_list in self.token_ids_2_item_ids.items():
            for item_id in item_list:
                self.data_list.append((token_ids_tuple,item_id))
        np.random.shuffle(self.data_list)
    
    def __len__(self):
        return 10000000000000 

    def __getitem__(self, idx):
        
        token_ids_tuple, item_id = self.data_list[idx % len(self.data_list)]
        
        if const.args.kw2item_neg_sample_mode == 'in-batch':
            return {"kw2item_token_ids": list(token_ids_tuple),
                    "kw2item_item_id": item_id,
                    } 
        else:
            neg_sample_ty = const.args.kw2item_neg_sample_mode.split(':')[-1] # for example: 'random'
            neg_item_list,neg_kw_list = None,None
            if 'item' in const.args.kw2item_neg_sample_mode:
                neg_item_list = self.get_neg_item_list(item_id, 
                                                       neg_sample_ty, 
                                                       neg_num=const.args.kw2item_neg_num) # List[int]
            if 'kw' in const.args.kw2item_neg_sample_mode:
                neg_kw_list = self.get_neg_kw_list(token_ids_tuple, 
                                                   neg_sample_ty, 
                                                   neg_num=const.args.kw2item_neg_num) # List[List[int]]: [[4,0],[3,9],...]
                
            data =  {"kw2item_token_ids": list(token_ids_tuple),
                    "kw2item_item_id": item_id,}
            if neg_item_list is not None:
                data['kw2item_neg'] = neg_item_list
            if neg_kw_list is not None:
                data['item2kw_neg'] = neg_kw_list
                
            return data
            
    def get_neg_item_list(self, pos_item_id, sample_type, neg_num):
        """"""
        candidate_items = [x for x in self.item_id_all if x != pos_item_id]
        
        if sample_type == 'random':
            # 
            return np.random.choice(candidate_items, size=neg_num, replace=False).tolist()
       
        
        else:
            raise ValueError(f"Unsupported item sample type: {sample_type}")

    def get_neg_kw_list(self, token_ids_tuple:tuple, sample_type, neg_num):
        """"""
        # candidate_tokens = [k for k in self.token_id_all if k not in pos_token_ids]
        candidate_kws = [k for k in self.token_ids_2_item_ids.keys() if k != token_ids_tuple]
        
        if sample_type == 'random':
            # 
            sampled_kws = np.random.choice(candidate_kws, size=neg_num, replace=False)
            return [self._convert_kw_to_ids(kw) for kw in sampled_kws]
        
    
        else:
            raise ValueError(f"Unsupported kw sample type: {sample_type}")

    def _convert_kw_to_ids(self, kw_str):
        """token id list"""
        if const.args.use_llm_token_emb_mode:
            return self.token_map.llm_tokenizer(kw_str).input_ids
        else:
            return [self.token_map.word2id[kw_str]]
       
     


class InfoNCEDataset(BaseDataSet):
    def __init__(self) -> None:
        super().__init__()

        item_id_set = set(const.item_vocab.keys())-set([const.PAD_ITEM_ID])
        self.item_id_all = list(item_id_set) # 1-d list like: [1,2,3...]
        
        self.PAD_TOKEN_ID = const.PAD_TOKEN_ID
        
        self.word_list_all = list(const.token_ids_all) # 2-d list like: [[4,0],[3,9],...] , no duplicates
        
        np.random.shuffle(self.word_list_all)
        np.random.shuffle(self.item_id_all)

    def __len__(self):
        return 10000000000000 



    def __getitem__(self, index):
        item = self.item_id_all[index % len(self.item_id_all)] # item_id, int: 4

        query = self.word_list_all[index % len(self.word_list_all)] # str list: [4,8]

        return {"align_neg_item": item,
                "align_neg_text": query} 
