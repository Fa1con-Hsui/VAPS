import argparse
import datetime
import logging
import os

from zmq import has

from models import *
from utils import *
from utils import const, utils
from transformers import AutoModel
# import setproctitle

# setproctitle.setproctitle("No Such Process")


def parse_global_args(parser: argparse.ArgumentParser):
    parser.add_argument('--gpu', type=str, default='4', help="'cpu' or gpu id str")
    # parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--random_seed', type=int, default=20230601)
    parser.add_argument('--suffix', type=str, default='none')
    parser.add_argument('--time', type=str, default='none')
    parser.add_argument('--train', type=int, default=1,help='the mode to run main.py: Test:0,Ttest:-1,Train:1')
    
    # parser.add_argument('--use_yaml_config', action='store_true') # 
    parser.add_argument('--yaml_config_path', type=str, default='')
    
    parser.add_argument(
        '--load_ckpt_path',
        type=str,
        default='',    )
    
    parser.add_argument('--overwrite_load_data', type=int, default=0,
                    help='0: Copy from the source dataset path and overwrite data in load_path; 1: Do not copy if data already exists in load_path')
    parser.add_argument('--use_llm_token_emb_mode', type=int, default=1,
                        help='0: Do not use LLM-enhanced token embeddings, making llm_path useless; 1: Use LLM tokenization and embeddings')
    parser.add_argument('--llm_path', type=str, default='/data1/Author/pretrained_models/Qwen2-7B-Instruct')

    # sampler:
    parser.add_argument('--is_seq_pad_left', type=int, default=0,
                        help='0: Pad right; 1: Pad left')
    parser.add_argument('--user_get_all_his_unisar', type=int, default=0,
                        help='Follow the unisar process, using timestamp to sort various behavioral interactions; 0: Do not use; 1: Use')
    # only for Commercial
    parser.add_argument('--src_negs_from', type=str, 
                        choices=['all','train_exposed','train_val_test_exposed','val_test_exposed'],
                        default='all',
                        help='In search data, whether to construct training or inference data using exposed negatives, or neither; default is neither, only for Commercial')
    # only for PersonalWAB
    parser.add_argument('--preSampleNeg4Infer', type=int, default=0,
                        help='Whether to use pre-sampled negatives as candidate items for inference, only for PersonalWAB')

    parser.add_argument('--num_candidates', type=int, default=0,
                        help='If src_negs_from==all and during inference, if >0, take the top k preprocessed items as negative candidates; otherwise, select all')
    parser.add_argument('--active_features_dict_name', type=str, default='active_features_dict',
                        help='')

    parser.add_argument('--emb_act', type=str, default='try_tanh',
                        choices=['try_relu','try_tanh','qwen2mlp','try_sigmoid','try_silu','try_prelu','try_swiglu','try_gelu'])
    # object_feat_act
    parser.add_argument('--object_feat_act', type=str, default='none', help='If "none", align with "emb_act"')
    parser.add_argument('--text_feat_act', type=str, default='none', help='If "none", align with "emb_act"')

    parser.add_argument('--use_LLMTokenEncoder_mlp', type=int, default=1,
                        help='0: Do not use; 1: Use')
    parser.add_argument('--use_query_cut_mode', type=str, default='only_no_cut',
                        choices=['only_no_cut', 'only_cut', 'add'])
    # dbg
    parser.add_argument('--dbg', type=int, default=0, help='Debug mode')
    # data_dbg
    parser.add_argument('--data_dbg_str', type=str, default='', 
                        help='If "train", "val", or "test" is in args.data_dbg_str, then only load the top 10 rows')
    # force_save_epoch
    parser.add_argument('--force_save_epochs', type=str, default="",
                        help='Disable early stopping and the mechanism of saving the best model based on the validation set, and force training to save the model at specific epochs. ' \
                        'If "45,46,47": Save results at these epochs and consider training finished at the largest epoch; ' \
                        'elif "": Do not force training to end and save the model at specific epochs, follow the validation-early stopping-saving mechanism')

    # freeze_text_mapping_epoch
    parser.add_argument('--freeze_text_mapping_epoch', type=int, default=0,
                        help='Number of epochs to freeze text-related parameters; if x, freeze from epoch 0 to x-1; if 0, do not freeze')

    # NOTE keyword2item CL
    parser.add_argument('--InfoNCE_kw2item_alpha', type=float, default=0.0,
                        help='The proportion of keyword-to-item contrastive learning in the total loss, ranging from 0 to 1; 0 means not using this contrastive learning')
    parser.add_argument('--kw_items_file_name', type=str, default='kw_items_original.pkl', 
                        choices=['kw_items_original.pkl','kw_items_onlyChinese.pkl', # for Commercial
                                'src_inv_index_v2.pkl','rec_inv_index_v2.pkl', # for PersonalWAB
                                'all_inv_index_v2.pkl','src_inv_index_v3.pkl', # for PersonalWAB
                                'rec_inv_index.pkl','src_inv_index.pkl','src_inv_index_v4.pkl','src_inv_index_v5.pkl',
                                'src_inv_index_v6.pkl'],
                        help='')
    # InfoNCE_kw2item_l2
    parser.add_argument('--InfoNCE_kw2item_l2', type=int, default=0,
                        help='Whether to apply L2 regularization to embeddings')
    parser.add_argument('--InfoNCE_kw2item_batchsize', type=int, default=512)
    parser.add_argument('--InfoNCE_kw2item_temp', type=float, default=0.1)
    parser.add_argument('--loss_kw2item_lambda', type=float, default=0.5)
    parser.add_argument('--loss_item2kw_lambda', type=float, default=0.5, help="No need to provide, it is 1-args.loss_kw2item_lambda")
    parser.add_argument('--kw2item_neg_num', type=int, default=512)
    parser.add_argument('--kw2item_neg_sample_mode', type=str, default='in-batch',
                        choices=['in-batch','item,kw:random','item:random','kw:random'],
                        help="""
    If 'in-batch': 'args.InfoNCE_kw2item_batchsize' activates,
            else: 'args.kw2item_neg_num' activates.
    """ 
    )

    # multi_align_alpha
    parser.add_argument('--multi_align_alpha', type=float, default=0.0,
                        help='The proportion of multi-alignment in the total loss, ranging from 0 to 1; 0 means not using this contrastive learning')
    parser.add_argument('--multi_align_emb_l2', type=int, default=0,
                        help='Whether to apply L2 regularization to embeddings')
    parser.add_argument('--multi_align_temperature', type=float, default=0.07,
                        help='Alignment temperature')

    # saved_epochs_num
    parser.add_argument('--saved_epochs_num', type=int, default=2,
                        help='Number of epochs to save models, default is 5, meaning the last 5 epochs are saved')

    # aug_src_his
    parser.add_argument('--aug_his', type=str, default='',
                        help='Prepared interaction sequence augmentation data (CoPPS). Can include "src,rec,conv,src_session"') 

    # use_preprocess_conv_his_of_src_session
    parser.add_argument('--use_preprocessed_conv_his_of_src_session', type=int, default=1,
                        help='Whether to use preprocessed conv_his of src_session; 0: Do not use; 1: Use. Default is to use')

    # MOE pooling
    # moev1_top_k
    parser.add_argument('--use_moev1', type=int, default=0,
                        help='use_moev1; 0: Do not use; 1: Use')
    # moev1_top_k must <= moev1_num_experts
    parser.add_argument('--moev1_top_k', type=int, default=3,
                        help='')
    parser.add_argument('--moev1_num_experts', type=int, default=5,
                        help='')
    parser.add_argument('--switch2MoEv2', type=int, default=0,
                        help='0 means not using; if used, args.use_moev1 must be 1')
    parser.add_argument('--switch2MoEv3', type=int, default=0,
                        help='0 means not using; if used, args.use_moev1 must be 1')
    parser.add_argument('--use_item_moe', type=int, default=0,
                        help='0 means not using; if used, args.use_moev1 must be 1')

    # use_extra_llm_emb
    parser.add_argument('--use_extra_llm_emb', type=int, default=0,
                        help='0 means not using; if used, args.use_llm_token_emb_mode must be 0')
    # llm_text_features
    parser.add_argument('--llm_text_features', type=str, default='item_name',
                        help='Will be split into a list using commas as separators')

    
    parser.add_argument('--seq_len', type=int, default=30,
                        help='')
    
    return parser


if __name__ == '__main__':
    

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    global_start_time = datetime.datetime.now()

    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

    init_parser = argparse.ArgumentParser(description='Model')
    init_parser.add_argument('--model', type=str, default='DIN')
    init_parser.add_argument('--runner', type=str, default="RecRunner")

    init_args, init_extras = init_parser.parse_known_args()
    
    for flag, value in sorted(init_args.__dict__.items(), key=lambda x: x[0]):
        logging.info('{}: {}'.format(flag, value))

    model_name: BaseModel = eval('{0}.{0}'.format(init_args.model))
    runner_name: BaseRunner = eval(init_args.runner)

    parser = argparse.ArgumentParser(description='')
    
    # parser.add_argument...
    parser = parse_global_args(parser)
    parser = model_name.parse_model_args(parser)
    parser = runner_name.parse_runner_args(parser)
    args, extras = parser.parse_known_args()
    if args.yaml_config_path:
        args = utils.load_hyperparam(args, args.yaml_config_path) # Author
        
    args.model = init_args.model
    if args.gpu == 'cpu':
        args.device = torch.device('cpu')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        args.device = torch.device('cuda')
        
    if args.force_save_epochs:
        args.force_save_epochs = [int(x) for x in args.force_save_epochs.split(',')]
        args.force_save_epochs.sort()
    else:
        args.force_save_epochs = []
    if args.text_feat_act == 'none':
        args.text_feat_act = args.emb_act
    if args.object_feat_act == 'none':
        args.object_feat_act = args.emb_act
    
    assert args.loss_kw2item_lambda >= 0 and args.loss_kw2item_lambda <= 1
    args.loss_item2kw_lambda = 1 - args.loss_kw2item_lambda   
    args.llm_text_features = args.llm_text_features.split(',') 
    
    assert args.InfoNCE_kw2item_alpha + args.multi_align_alpha < 1
    

    utils.setup_seed(args.random_seed)
    args.llm_name = args.llm_path.split('/')[-1] if args.use_llm_token_emb_mode else ''
    args.extra_llm_name = args.llm_path.split('/')[-1] 
    data_process_info = const.init_setting(args) 
    if data_process_info:
        for key, value in data_process_info.items():
            setattr(const, key, value)
            logging.info(f'SetConst -> setting const.{key}')
            
    if args.seq_len > 0:
        if hasattr(const,'max_conv_his_len'):
            setattr(const, 'max_conv_his_len', args.seq_len)
        if hasattr(const,'max_src_his_len'):
            setattr(const, 'max_src_his_len', args.seq_len)
        if hasattr(const,'max_src_session_his_len'):
            setattr(const, 'max_src_session_his_len', args.seq_len)
        if hasattr(const,'max_rec_his_len'):
            setattr(const, 'max_rec_his_len', args.seq_len)
    
        
    # Process and prepare all textual data to obtain the vocabulary tensor on the device.
    if 'PersonalWAB' in args.data:
        const.process_text_data_en(args)
    elif "Commercial" in args.data:
        const.process_text_data(args)   
    else:
        raise ValueError
    
    
    llm = AutoModel.from_pretrained(args.llm_path)# .to(args.device)
    setattr(const, 'llm', llm)
    logging.info(f'SetConst -> setting const.llm from {args.llm_path}')
    
    
    if args.time == 'none':
        cur_time = datetime.datetime.now()
        args.time = cur_time.strftime(r"%Y%m%d-%H%M%S")
    
    args.model_path = f"output/{args.data}/{init_args.runner}/{init_args.model}/{args.suffix}"
    os.makedirs(args.model_path,exist_ok=True)



    for flag, value in sorted(args.__dict__.items(), key=lambda x: x[0]):
        logging.info('{}: {}'.format(flag, value))

    model: BaseModel = model_name(args)
    
    runner: BaseRunner = runner_name(model, args) # build dataset and dataloader

    num_parameters = model.count_variables()
    logging.info("num model parameters:{}".format(num_parameters))

    if args.train == 1:

        logging.info(f"{runner.train_loader.dataset.sampler.train_mode}")
        logging.info(f"{runner.val_loader.dataset.sampler.train_mode}")
        logging.info(f"{runner.test_loader.dataset.sampler.train_mode}")
    
        
            
        if args.load_ckpt_path:
            assert 'finetune' in args.suffix
            model.load_model(ckpt_path=args.load_ckpt_path)
            logging.info(f'load ckpt: {args.load_ckpt_path}')
            runner.train(model)
        else:
            runner.train(model)
    
    elif args.train == -1:
        t_test_result = []
        print()
        logging.info("Ttest")
        logging.info(const.rec_test)
        logging.info(const.src_test)
        dir_list = sorted(
            os.listdir(
                "output/KuaiSAR_v2_1119/{}_{}_Ttest/checkpoints/".format(
                    init_args.model, init_args.runner)))
        for dir in dir_list:
            path = "output/KuaiSAR_v2_1119/{}_{}_Ttest/checkpoints/{}/best.pt".format(
                init_args.model, init_args.runner, dir)

            model.load_model(ckpt_path=path)

            test_result = runner.test(model, 'test', tqdm_desc=f'TEST')
            logging.info("Test Result:")
            # logging.info(utils.format_metric(test_result))
            logging.info(test_result)
            print()

            t_test_result.append(test_result)

        t_test_df = pd.DataFrame(t_test_result)
        t_test_df.to_pickle(
            "output/KuaiSAR_v2_1119/{}_{}_Ttest/t_test.pkl".format(
                init_args.model, init_args.runner))


    elif args.train == 0:
        model.load_model(ckpt_path=args.load_ckpt_path)

        test_result = runner.test(model, 'test', tqdm_desc=f'TEST')
        logging.info("Test Result:")
        logging.info(utils.format_metric(test_result))

        # val_result, _ = runner.evaluate(model, 'val')
        # logging.info("Val Result:")
        # logging.info(val_result)

    else: 
        raise ValueError('args.train error')
        # train == 1
        # "self.train_loader" 
       

    global_end_time = datetime.datetime.now()
    print("runnning used time:{}".format(global_end_time - global_start_time))
