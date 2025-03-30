from ast import Not
import glob
import json
import logging
# from multiprocessing import process
import os
import pdb
import pickle
import shutil
import sys
from typing import Union,List

import numpy as np
import pandas as pd
from sympy import Dict
import torch
from tqdm import tqdm
import runpy
# from Py_projects.llama3.llama import tokenizer





def init_setting(args):
    """
    **Function Name:** `init_setting`

    **Function Description:**  
    This function is primarily used to initialize specific settings and load related data. Based on the passed parameter `args`, it configures and integrates various information related to the data. This includes reading pre-stored pickle files, setting data-related flags, paths, and dimensions of different data structures, among others. Finally, it returns a consolidated data dictionary.

    **Parameters:**  
    `args`: An object containing a series of configuration parameters. 
    It is used within the function to retrieve information such as the data name,
    whether the source data is paired, random seed, and other related configuration details.

    **Return Value:**  
    `const_data`: A dictionary that integrates various data-related information.
    It includes data paths, flags for whether certain features are used, 
    specific data for different partitions, as well as vocabularies related to users, 
    items, and other entities, along with settings for data structure dimensions and other aspects.
    """
    
    
    if 'PersonalWAB' in args.data:
        load_path = f"data/{args.data}"

               
        const_data = pd.read_pickle(os.path.join(load_path,'data_process_info.pkl'))
        const_data['args'] = args
        const_data['load_path'] = load_path
        const_data['use_bins'] = False
        const_data['use_bert_encoder_query'] = False
    
        
        const_data['random_seed'] = args.random_seed
        const_data['pred_workers'] = 2
        const_data['shuffle_train_data'] = True
        
        VOCAB_DATA_DIR = const_data['VOCAB_DATA_DIR']
        SPLIT_DATASET_DIR_NAME = const_data['SPLIT_DATASET_DIR_NAME']
        KW_ITEMS_DIR = 'cs_inv_index'
        
        
        
        const_data.update(dict(
            user_vocab = pd.read_pickle(os.path.join(load_path, f'{VOCAB_DATA_DIR}/user_vocab.pkl')),
            item_vocab = pd.read_pickle(os.path.join(load_path, f'{VOCAB_DATA_DIR}/item_vocab.pkl')),
            src_train = pd.read_pickle(os.path.join(load_path, f'{SPLIT_DATASET_DIR_NAME}/src_train.pkl')),
            src_val = pd.read_pickle(os.path.join(load_path, f'{SPLIT_DATASET_DIR_NAME}/src_val.pkl')),
            src_test = pd.read_pickle(os.path.join(load_path, f'{SPLIT_DATASET_DIR_NAME}/src_test.pkl')),
            rec_train = pd.read_pickle(os.path.join(load_path, f'{SPLIT_DATASET_DIR_NAME}/rec_train.pkl')),
            rec_val = pd.read_pickle(os.path.join(load_path, f'{SPLIT_DATASET_DIR_NAME}/rec_val.pkl')),
            rec_test = pd.read_pickle(os.path.join(load_path, f'{SPLIT_DATASET_DIR_NAME}/rec_test.pkl')),
            rec_item_set = pd.read_pickle(os.path.join(load_path, f'{VOCAB_DATA_DIR}/interacted_rec_item_set.pkl')),
            src_item_set = pd.read_pickle(os.path.join(load_path, f'{VOCAB_DATA_DIR}/interacted_src_item_set.pkl')),
            
            user_vocab_np = pd.read_pickle(os.path.join(load_path, f'{VOCAB_DATA_DIR}/user_vocab_np.pkl')),
            item_vocab_np = pd.read_pickle(os.path.join(load_path, f'{VOCAB_DATA_DIR}/item_vocab_np.pkl')),
            src_session_vocab_np = pd.read_pickle(os.path.join(load_path, f'{VOCAB_DATA_DIR}/search_session_vocab_np.pkl')),
            conv_vocab_np = pd.read_pickle(os.path.join(load_path, f'{VOCAB_DATA_DIR}/conv_inter_vocab_np.pkl')),

            item_mask_token = const_data['PAD_ITEM_ID'], 
  
            word_id_dim = 32,
            
            mid_word_id_dim = 128, # for llm token mlp
            
            final_emb_size = 64,
            max_rec_his_len = 30,
            max_src_session_his_len = 30,
            max_src_his_len = 30 ,
            max_conv_his_len = 30, 
            
            # hyper-param
            max_conv_text_len = 400, 
            max_query_text_len = 70, 
            max_item_text_len = 70, 
            
            
            active_features_dict = runpy.run_path(
                os.path.join(load_path,"active_features.py")
            )[args.active_features_dict_name],
                                

            kw_item_data = dict(kw_items_dict=\
                pd.read_pickle(os.path.join(load_path, f'{KW_ITEMS_DIR}/{args.kw_items_file_name}')))\
                    if args.InfoNCE_kw2item_alpha>0.0 else None,
            
        ))
        # Try to load
        process_text_data_names = ['process_text_data_step1',
                                   'process_text_data_step2',
                                   'process_text_data_step3',
                                   ]
            
        for name in process_text_data_names:
            if os.path.exists(os.path.join(load_path, f'process_text_data/{name}_{args.llm_name}.pkl')):
                const_data[name] = pd.read_pickle(
                    os.path.join(load_path, f'process_text_data/{name}_{args.llm_name}.pkl'))
    else:
        raise NotImplementedError

    return const_data


 



def list_or_numpy_to_torch_and_to_device(input_dict, device, active_features=[]):
    """
    This function converts values in a dictionary that are lists or NumPy arrays 
    into PyTorch tensors and places them on the specified device (e.g., CPU or GPU).
    """
    result_dict = {}
    for key, value in input_dict.items():
        if active_features and  key not in active_features:
            continue
        if isinstance(value, list):
            tensor_value = torch.tensor(value).to(device)
            result_dict[key] = tensor_value
            logging.info(f"VOCAB: Converted to torch tensor and placed on {device}: {key} shape{tensor_value.shape}")
        elif isinstance(value, np.ndarray):
            tensor_value = torch.from_numpy(value).to(device)
            result_dict[key] = tensor_value
            logging.info(f"VOCAB: Converted to torch tensor and placed on {device}: {key} shape{tensor_value.shape}")
        else:
            result_dict[key] = value
            logging.info(f"VOCAB: Ignore as it is not a list or numpy array: {key} type({type(value)})")
    return result_dict

def get_var(var_name):
    return getattr(sys.modules[__name__], var_name, None)

def has_var(var_name):
    return var_name in globals()


def resume_process_text_data(name):
    process_text_data_step_any = get_var(name)
    conv_vocab_np = process_text_data_step_any['conv_vocab_np']
    src_session_vocab_np = process_text_data_step_any['src_session_vocab_np']
    item_vocab_np = process_text_data_step_any['item_vocab_np']
    token_map = process_text_data_step_any['token_map']
    token_ids_all = process_text_data_step_any['token_ids_all']
    return conv_vocab_np, src_session_vocab_np, item_vocab_np, token_map, token_ids_all



def process_text_data_en(args):
    global PAD_CATEGORY_TOKEN, PAD_CONV_REIDX, conv_vocab_np
    global src_session_vocab_np, item_vocab_np, user_vocab_np, active_features_dict
    global max_item_text_len, max_query_text_len, max_conv_text_len
    global llm_process_text_data
    global token_map, token_ids_all, token_ids_max_len,\
        PAD_TOKEN_ID, PAD_TOKEN
    
    PAD_TOKEN_ID = 0
    PAD_TOKEN = PAD_CATEGORY_TOKEN
    if not args.use_llm_token_emb_mode:
        llm_tokenizer = None
        token_map = TokenMapEn(pad_token=PAD_TOKEN,
                             pad_token_id=PAD_TOKEN_ID)
    else:
        from transformers import AutoTokenizer
        llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_path)
        token_map = TokenMapEn(pad_token=PAD_TOKEN,  # original
                                pad_token_id=PAD_TOKEN_ID,  # original
                                llm_tokenizer=llm_tokenizer)
        # update pad_token and pad_token_id
        PAD_TOKEN = llm_tokenizer.pad_token
        PAD_TOKEN_ID = llm_tokenizer.pad_token_id
    

    token_ids_all = []  
    if has_var("process_text_data_step1"):
        conv_vocab_np,\
        src_session_vocab_np,\
        item_vocab_np,\
        token_map,\
        token_ids_all = resume_process_text_data("process_text_data_step1")
    else:
        for feat_name in tqdm(conv_text_features,desc="collect-conv_text_features"):
            conv_vocab_np[feat_name] = \
                [token_map.collect(text) for text in tqdm(conv_vocab_np[feat_name], 
                                    desc=f"collect-conv_text_features-{feat_name}")]
        for feat_name in tqdm(['query'],desc="collect-src_session_text_features"):
            src_session_vocab_np[feat_name] = \
                [token_map.collect(text) for text in tqdm(src_session_vocab_np[feat_name],
                                    desc=f"collect-src_session_vocab_np-{feat_name}")]  
        for feat_name in tqdm(item_text_features,desc="collect-item_text_features"):
            item_vocab_np[feat_name] = \
                [token_map.collect(text) for text in tqdm(item_vocab_np[feat_name],
                                    desc=f"collect-item_vocab_np-{feat_name}")] 
        os.makedirs(os.path.join(load_path,f'process_text_data'), exist_ok=True)
        pickle.dump(dict(conv_vocab_np=conv_vocab_np,
                        token_map=token_map,
                        item_vocab_np=item_vocab_np,
                        src_session_vocab_np=src_session_vocab_np,
                        token_ids_all=token_ids_all), 
            open(os.path.join(load_path,f'process_text_data/process_text_data_step1_{args.llm_name}.pkl'), 'wb')) 
    
    # Stage 2 step 2: map 
    if has_var("process_text_data_step2"):
        conv_vocab_np,\
        src_session_vocab_np,\
        item_vocab_np,\
        token_map,\
        token_ids_all = resume_process_text_data("process_text_data_step2")
    else:
        for feat_name in tqdm(conv_text_features,desc="map-conv_text_features"):
            conv_vocab_np[feat_name] = \
                [token_map.map(text) for text in tqdm(conv_vocab_np[feat_name], 
                                    desc=f"map-conv_text_features-{feat_name}")]
        for feat_name in tqdm(['query'],desc="map-src_session_text_features"):
            src_session_vocab_np[feat_name] = \
                [token_map.map(text) for text in tqdm(src_session_vocab_np[feat_name],
                                    desc=f"map-src_session_vocab_np-{feat_name}")]  
        for feat_name in tqdm(item_text_features,desc="map-item_text_features"):
            item_vocab_np[feat_name] = \
                [token_map.map(text) for text in tqdm(item_vocab_np[feat_name],
                                    desc=f"map-item_vocab_np-{feat_name}")]  
        os.makedirs(os.path.join(load_path,f'process_text_data'), exist_ok=True)
        pickle.dump(dict(conv_vocab_np=conv_vocab_np,
                        token_map=token_map,
                        item_vocab_np=item_vocab_np,
                        src_session_vocab_np=src_session_vocab_np,
                        token_ids_all=token_ids_all),
            open(os.path.join(load_path,f'process_text_data/process_text_data_step2_{args.llm_name}.pkl'), 'wb')) 
        
    
    # Stage 2 step 3: pad and re-collect
    if has_var("process_text_data_step3"):
        conv_vocab_np,\
        src_session_vocab_np,\
        item_vocab_np,\
        token_map,\
        token_ids_all = resume_process_text_data("process_text_data_step3")
    else:
        for feat_name in tqdm(conv_text_features,desc="pad-conv_text_features"):
            conv_vocab_np[feat_name] = \
                [token_map.pad(token_ids,max_conv_text_len) for token_ids in tqdm(conv_vocab_np[feat_name], 
                                    desc=f"pad-conv_text_features-{feat_name}")]
            token_ids_all.extend(conv_vocab_np[feat_name])
        for feat_name in tqdm(['query'],desc="pad-src_session_text_features"):
            src_session_vocab_np[feat_name] = \
                [token_map.pad(token_ids,max_query_text_len) for token_ids in tqdm(src_session_vocab_np[feat_name],
                                    desc=f"pad-src_session_vocab_np-{feat_name}")]  
            token_ids_all.extend(src_session_vocab_np[feat_name])
        for feat_name in tqdm(item_text_features,desc="pad-item_text_features"):
            item_vocab_np[feat_name] = \
                [token_map.pad(token_ids,max_item_text_len) for token_ids in tqdm(item_vocab_np[feat_name],
                                    desc=f"pad-item_vocab_np-{feat_name}")]  
            token_ids_all.extend(item_vocab_np[feat_name])    
        token_ids_all = [tuple(token_ids) for token_ids in token_ids_all]
        token_ids_all = list(set(token_ids_all))
        token_ids_all = [list(token_ids) for token_ids in token_ids_all]
        os.makedirs(os.path.join(load_path,f'process_text_data'), exist_ok=True)
        pickle.dump(dict(conv_vocab_np=conv_vocab_np,
                        token_map=token_map,
                        item_vocab_np=item_vocab_np,
                        src_session_vocab_np=src_session_vocab_np,
                        token_ids_all=token_ids_all), 
            open(os.path.join(load_path,f'process_text_data/process_text_data_step3_{args.llm_name}.pkl'), 'wb')) 
        
    # Stage 3: 
    conv_vocab_np = list_or_numpy_to_torch_and_to_device(conv_vocab_np, args.device,
                                                         active_features_dict['conv'])
    src_session_vocab_np = list_or_numpy_to_torch_and_to_device(src_session_vocab_np, args.device)
    item_vocab_np = list_or_numpy_to_torch_and_to_device(item_vocab_np, args.device, 
                                                         active_features_dict['item'])
    user_vocab_np = list_or_numpy_to_torch_and_to_device(user_vocab_np, args.device, 
                                                         active_features_dict['user'])
    
    llm_process_text_data = None
    if args.use_extra_llm_emb:
        assert not args.use_llm_token_emb_mode
        llm_process_text_data = pd.read_pickle(os.path.join(load_path,
                             f'process_text_data/process_text_data_step3_{args.extra_llm_name}.pkl'))
        if args.dbg:
            pdb.set_trace()
        llm_process_text_data['conv_vocab_np'] = list_or_numpy_to_torch_and_to_device(
            llm_process_text_data['conv_vocab_np'] , args.device, active_features_dict['conv'])
        llm_process_text_data['src_session_vocab_np'] = list_or_numpy_to_torch_and_to_device(
            llm_process_text_data['src_session_vocab_np'], args.device)
        llm_process_text_data['item_vocab_np'] = list_or_numpy_to_torch_and_to_device(
            llm_process_text_data['item_vocab_np'], args.device, active_features_dict['item'])

    
 
    token_ids_max_len = token_map.max_token_list_len
    


class TokenMap:
    def __init__(self, pad_token_id, pad_token, llm_tokenizer=None):
        
        self.max_token_list_len = -1 # init
        
        # not llm mode
        if not llm_tokenizer:
            assert pad_token_id==0 # must 0
            self.llm_tokenizer = None
            self.all_words = []
            self.word2id = {}
            self.pad_token = pad_token
            self.pad_token_id = pad_token_id 
        else: # llm mode
            self.llm_tokenizer = llm_tokenizer 
            self.orig_pad_token = pad_token
            self.orig_pad_token_id = pad_token_id
            self.pad_token = llm_tokenizer.pad_token
            self.pad_token_id = llm_tokenizer.pad_token_id 
   
        
    def collect(self,text:Union[str,List[str]]):
        '''
        step 1 collect:
        '''
            
        if type(text) == str: 
            if not self.llm_tokenizer:
                self.all_words.append(text)
            else:
                text = text.replace(self.orig_pad_token,self.pad_token)
        elif type(text) == list: # precut str list
            if text == []:
                text = [self.pad_token]
            if not self.llm_tokenizer:
                self.all_words.extend(text)
            else:
                text = [w.replace(self.orig_pad_token,self.pad_token) \
                                if w!='' else self.pad_token for w in text]
        else:
            raise ValueError('text not str or list')
        return text
    
    def map(self,text:Union[str,List[str]]):
        '''
        step 2: text 2 token id
        '''
        try:
            if not self.llm_tokenizer:
                if not self.word2id:
                    self.all_words = [self.pad_token] + \
                            list(set(self.all_words) - set([self.pad_token]))
                    self.word2id = {w:i for i,w in enumerate(self.all_words)}
                
                if type(text) == str:
                    token_ids = [self.word2id[text]] 
                elif type(text) == list:
                    token_ids =  [self.word2id[w] for w in text] # List[int]
                else:
                    raise ValueError('text not str or list')
            else:
                if type(text) == str:
                    text = text.strip()
                    text = self.pad_token if text=='' else text
                    token_ids = self.llm_tokenizer(text).input_ids
                elif type(text) == list: # List[str]
                    text = [w.strip() for w in text if w.strip()!='']
                    if text == []:
                        token_ids = [self.pad_token_id] 
                    else:
                        token_ids = self.llm_tokenizer(text).input_ids
                        token_ids = [token_id for li in token_ids for token_id in li]     
                else:
                    raise ValueError('text not str or list')
        except Exception as e:
            logging.error(f'text:{text}')
            logging.error(e)
            pdb.set_trace()
        self.max_token_list_len = max(len(token_ids),self.max_token_list_len)
        return token_ids
            
    def pad(self,token_ids:List[int], max_len=None):
        '''
        step 3: use self.max_token_list_len and self.pad_token_id to pad
        '''
        if not max_len:
            max_len = self.max_token_list_len
        
        token_ids = token_ids[:max_len]
        token_ids += [self.pad_token_id]*(max_len-len(token_ids))
        
        return token_ids
    
    

class TokenMapEn(TokenMap):
    def __init__(self, pad_token_id, pad_token, llm_tokenizer=None):
        
        self.max_token_list_len = -1 # init
        
        # not llm mode
        if not llm_tokenizer:
            assert pad_token_id==0 # must 0
            self.llm_tokenizer = None
            self.all_words = []
            self.word2id = {}
            self.pad_token = pad_token
            self.pad_token_id = pad_token_id 
            self.stop_words = json.load(open('utils/all_stopwords.json', 'r'))
        else: # llm mode
            self.llm_tokenizer = llm_tokenizer 
            self.orig_pad_token = pad_token
            self.orig_pad_token_id = pad_token_id
            self.pad_token = llm_tokenizer.pad_token
            self.pad_token_id = llm_tokenizer.pad_token_id 
   
        
    def collect(self,text:Union[str,List[str]]):
        '''
        step 1 collect:
        '''
            
        if type(text) == str: 
            if not self.llm_tokenizer:
                text = text.lower().strip().split(' ')
                text = [w for w in text if w not in self.stop_words]
                self.all_words.extend(text)
            else:
                text = text.replace(self.orig_pad_token,self.pad_token)
        else:
            raise ValueError
        return text
    
    def map(self,text:Union[str,List[str]]):
        '''
        step 2: text 2 token id
        '''
        try:
            if not self.llm_tokenizer:
                if not self.word2id:
                    self.all_words = [self.pad_token] + \
                            list(set(self.all_words) - set([self.pad_token]))
                    self.word2id = {w:i for i,w in enumerate(self.all_words)}
                
                if type(text) == list:
                    token_ids =  [self.word2id[w] for w in text] # List[int]
                else:
                    raise ValueError
            else:
                if type(text) == str:
                    text = text.strip()
                    text = self.pad_token if text=='' else text
                    token_ids = self.llm_tokenizer(text).input_ids
   
                else:
                    raise ValueError
        except Exception as e:
            logging.error(f'text:{text}')
            logging.error(e)
            pdb.set_trace()
        self.max_token_list_len = max(len(token_ids),self.max_token_list_len)
        return token_ids
            
