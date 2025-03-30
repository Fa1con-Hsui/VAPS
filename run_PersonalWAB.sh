ex_id='0-qwen2.5Emb-lr_scheduler'
model_name='TEM' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr 1e-4  \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 1   \
    --lr_scheduler 1   \
    \
    --gpu 2 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &


ex_id='1-noLlmEmb-lr_scheduler'
model_name='TEM' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr 1e-4  \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --gpu 1 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &





ex_id='0-qwen2.5Emb-lr_scheduler'
model_name='AEM' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr 1e-4  \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 1   \
    --lr_scheduler 1   \
    \
    --gpu 0 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &

ex_id='1-noLlmEmb-lr_scheduler'
model_name='AEM' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr 1e-4  \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --gpu 1 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &





ex_id='0-qwen2.5Emb-lr_scheduler'
model_name='HEM' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr 1e-4  \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 1   \
    --lr_scheduler 1   \
    \
    --gpu 2 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &

ex_id='1-noLlmEmb-lr_scheduler'
model_name='HEM' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr 1e-4  \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --gpu 3 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &



ex_id='0-qwen2.5Emb-lr_scheduler'
model_name='QEM' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr 1e-4  \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 1   \
    --lr_scheduler 1   \
    \
    --gpu 4 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &

ex_id='1-noLlmEmb-lr_scheduler'
model_name='QEM' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr 1e-4  \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --gpu 5 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &




ex_id='0-qwen2.5Emb-lr_scheduler'
model_name='ZAM' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr 1e-4  \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 1   \
    --lr_scheduler 1   \
    \
    --gpu 6 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &

ex_id='1-noLlmEmb-lr_scheduler'
model_name='ZAM' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr 1e-4  \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --gpu 7 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &






ex_id='0-qwen2.5Emb-lr_scheduler'
model_name='CoPPS' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr 1e-4  \
    \
    --batch_size 512   --eval_batch_size 128   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 1   \
    --lr_scheduler 1   \
    \
    --aug_his src \
    \
    --gpu 4 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &

ex_id='1-noLlmEmb-lr_scheduler'
model_name='CoPPS' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr 1e-4  \
    \
    --batch_size 512   --eval_batch_size 128   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --aug_his src \
    \
    --gpu 5 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &




############### TEM_CS_0118 ##################


# ex_id='0-qwen2.5Emb-lr_scheduler'
# model_name='TEM_CS_0118' 
# data='PersonalWAB'
# runner='SrcRunner'
# current_time=$(date +"%Y-%m-%d_%H:%M:%S")
# mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
# nohup python3 main.py --model ${model_name} --runner $runner   \
#     --suffix ${ex_id} --epoch 50 --data $data   \
#     --lr 1e-4  \
#     \
#     --batch_size 72   --eval_batch_size 1   \
#     --negs_num 10  --preSampleNeg4Infer 1   \
#     \
#     --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
#     --active_features_dict_name  active_features_dict \
#     --emb_act try_tanh \
#     --use_LLMTokenEncoder_mlp 0   \
#     --use_llm_token_emb_mode 1   \
#     --lr_scheduler 1   \
#     \
#     --use_preprocessed_conv_his_of_src_session 0 \
#     \
#     --gpu 2 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &


# ex_id='1-noLlmEmb-lr_scheduler'
# model_name='TEM_CS_0118' 
# data='PersonalWAB'
# runner='SrcRunner'
# current_time=$(date +"%Y-%m-%d_%H:%M:%S")
# mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
# nohup python3 main.py --model ${model_name} --runner $runner   \
#     --suffix ${ex_id} --epoch 50 --data $data   \
#     --lr 1e-4  \
#     \
#     --batch_size 72   --eval_batch_size 1   \
#     --negs_num 10  --preSampleNeg4Infer 1   \
#     \
#     --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
#     --active_features_dict_name  active_features_dict \
#     --emb_act try_tanh \
#     --use_LLMTokenEncoder_mlp 0   \
#     --use_llm_token_emb_mode 0   \
#     --lr_scheduler 1   \
#     \
#     --use_preprocessed_conv_his_of_src_session 0 \
#     \
#     --gpu 1 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &


# ex_id='2-qwen2.5Emb-lr_scheduler'
# model_name='TEM_CS_0118' 
# data='PersonalWAB'
# runner='SrcRunner'
# current_time=$(date +"%Y-%m-%d_%H:%M:%S")
# mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
# nohup python3 main.py --model ${model_name} --runner $runner   \
#     --suffix ${ex_id} --epoch 50 --data $data   \
#     --lr 1e-4  \
#     \
#     --batch_size 72   --eval_batch_size 1   \
#     --negs_num 10  --preSampleNeg4Infer 1   \
#     \
#     --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
#     --active_features_dict_name  active_features_dict \
#     --emb_act try_tanh \
#     --use_LLMTokenEncoder_mlp 0   \
#     --use_llm_token_emb_mode 1   \
#     --lr_scheduler 1   \
#     \
#     --conv_seq_pad_mode left \
#     --use_preprocessed_conv_his_of_src_session 0 \
#     \
#     --gpu 0 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &


# ex_id='3-noLlmEmb-lr_scheduler'
# model_name='TEM_CS_0118' 
# data='PersonalWAB'
# runner='SrcRunner'
# current_time=$(date +"%Y-%m-%d_%H:%M:%S")
# mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
# nohup python3 main.py --model ${model_name} --runner $runner   \
#     --suffix ${ex_id} --epoch 50 --data $data   \
#     --lr 1e-4  \
#     \
#     --batch_size 72   --eval_batch_size 1   \
#     --negs_num 10  --preSampleNeg4Infer 1   \
#     \
#     --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
#     --active_features_dict_name  active_features_dict \
#     --emb_act try_tanh \
#     --use_LLMTokenEncoder_mlp 0   \
#     --use_llm_token_emb_mode 0   \
#     --lr_scheduler 1   \
#     \
#     --conv_seq_pad_mode left \
#     --use_preprocessed_conv_his_of_src_session 0 \
#     \
#     --gpu 1 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &



# ex_id='4-qwen2.5Emb-lr_scheduler'
# model_name='TEM_CS_0118' 
# data='PersonalWAB'
# runner='SrcRunner'
# current_time=$(date +"%Y-%m-%d_%H:%M:%S")
# mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
# nohup python3 main.py --model ${model_name} --runner $runner   \
#     --suffix ${ex_id} --epoch 50 --data $data   \
#     --lr 1e-4  \
#     \
#     --batch_size 72   --eval_batch_size 1   \
#     --negs_num 10  --preSampleNeg4Infer 1   \
#     \
#     --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
#     --active_features_dict_name  active_features_dict \
#     --emb_act try_tanh \
#     --use_LLMTokenEncoder_mlp 0   \
#     --use_llm_token_emb_mode 1   \
#     --lr_scheduler 1   \
#     \
#     --InfoNCE_kw2item_alpha 0.2 \
#     --kw_items_file_name 'src_inv_index_v2.pkl' \
#     --InfoNCE_kw2item_batchsize 512 \
#     --InfoNCE_kw2item_temp 0.1 \
#     --loss_kw2item_lambda 1.0 \
#     --loss_item2kw_lambda 0.0 \
#     --kw2item_neg_num 512 \
#     --kw2item_neg_sample_mode 'in-batch' \
#     \
#     --use_preprocessed_conv_his_of_src_session 0 \
#     \
#     --gpu 2 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &


# ex_id='5-noLlmEmb-lr_scheduler'
# model_name='TEM_CS_0118' 
# data='PersonalWAB'
# runner='SrcRunner'
# current_time=$(date +"%Y-%m-%d_%H:%M:%S")
# mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
# nohup python3 main.py --model ${model_name} --runner $runner   \
#     --suffix ${ex_id} --epoch 50 --data $data   \
#     --lr 1e-4  \
#     \
#     --batch_size 72   --eval_batch_size 1   \
#     --negs_num 10  --preSampleNeg4Infer 1   \
#     \
#     --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
#     --active_features_dict_name  active_features_dict \
#     --emb_act try_tanh \
#     --use_LLMTokenEncoder_mlp 0   \
#     --use_llm_token_emb_mode 0   \
#     --lr_scheduler 1   \
#     \
#     --InfoNCE_kw2item_alpha 0.2 \
#     --kw_items_file_name 'src_inv_index_v2.pkl' \
#     --InfoNCE_kw2item_batchsize 512 \
#     --InfoNCE_kw2item_temp 0.1 \
#     --loss_kw2item_lambda 1.0 \
#     --loss_item2kw_lambda 0.0 \
#     --kw2item_neg_num 512 \
#     --kw2item_neg_sample_mode 'in-batch' \
#     \
#     --use_preprocessed_conv_his_of_src_session 0 \
#     \
#     --gpu 3 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &

# --overwrite_load_data 1

ex_id='0-bugFix-qwen2.5Emb-lr_scheduler'
model_name='TEM_CS_0118' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr 1e-4  \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 1   \
    --lr_scheduler 1   \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    \
    --gpu 2 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &


ex_id='1-bugFix-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0118' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr 1e-4  \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    \
    --gpu 3 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &


ex_id='2-bugFix-qwen2.5Emb-lr_scheduler'
model_name='TEM_CS_0118' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr 1e-4  \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 1   \
    --lr_scheduler 1   \
    \
    --conv_seq_pad_mode left \
    --use_preprocessed_conv_his_of_src_session 0 \
    \
    --gpu 2 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &


ex_id='3-bugFix-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0118' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr 1e-4  \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --conv_seq_pad_mode left \
    --use_preprocessed_conv_his_of_src_session 0 \
    \
    --gpu 3 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &



ex_id='4-bugFix-qwen2.5Emb-lr_scheduler'
model_name='TEM_CS_0118' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr 1e-4  \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 1   \
    --lr_scheduler 1   \
    \
    --InfoNCE_kw2item_alpha 0.2 \
    --kw_items_file_name 'src_inv_index_v2.pkl' \
    --InfoNCE_kw2item_batchsize 512 \
    --InfoNCE_kw2item_temp 0.1 \
    --loss_kw2item_lambda 1.0 \
    --loss_item2kw_lambda 0.0 \
    --kw2item_neg_num 512 \
    --kw2item_neg_sample_mode 'in-batch' \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    \
    --gpu 4 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &


ex_id='5-bugFix-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0118' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr 1e-4  \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --InfoNCE_kw2item_alpha 0.2 \
    --kw_items_file_name 'src_inv_index_v2.pkl' \
    --InfoNCE_kw2item_batchsize 512 \
    --InfoNCE_kw2item_temp 0.1 \
    --loss_kw2item_lambda 1.0 \
    --loss_item2kw_lambda 0.0 \
    --kw2item_neg_num 512 \
    --kw2item_neg_sample_mode 'in-batch' \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    \
    --gpu 5 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &





############### TEM_CS_0204 ##################


ex_id='0-qwen2.5Emb-lr_scheduler'
model_name='TEM_CS_0204' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr 1e-4  \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 1   \
    --lr_scheduler 1   \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    \
    --gpu 0 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &


ex_id='1-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0204' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr 1e-4  \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    \
    --gpu 1 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &



ex_id='1.cp-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0204' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr 1e-4  \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    \
    --gpu 7 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &



ex_id='1.cp5e4-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0204' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr 5e-4  \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    \
    --gpu 7 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &


ex_id='1.cp5e4-info-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0204' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr 5e-4  \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --InfoNCE_kw2item_alpha 0.2 \
    --kw_items_file_name 'src_inv_index_v2.pkl' \
    --InfoNCE_kw2item_batchsize 512 \
    --InfoNCE_kw2item_temp 0.1 \
    --loss_kw2item_lambda 1.0 \
    --loss_item2kw_lambda 0.0 \
    --kw2item_neg_num 512 \
    --kw2item_neg_sample_mode 'in-batch' \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    \
    --gpu 5 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &

############## TEM_CS_0204_QuerySum ################



ex_id='0-qwen2.5Emb-lr_scheduler'
model_name='TEM_CS_0204_QuerySum' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr  1e-4   \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 1   \
    --lr_scheduler 1   \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    \
    \
    --wa_learable 0 \
    --wa_initial_weights '1,1,1' \
    \
    --gpu 0 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &


ex_id='1-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0204_QuerySum' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr  1e-4   \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    \
    \
    --wa_learable 0 --wa_initial_weights '1,1,1' \
    \
    --gpu 1 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &



ex_id='2-qwen2.5Emb-lr_scheduler'
model_name='TEM_CS_0204_QuerySum' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr  1e-4  \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 1   \
    --lr_scheduler 1   \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    \
    \
    --wa_learable 1 --wa_initial_weights '1,1,1' \
    \
    --gpu 2 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &


ex_id='3-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0204_QuerySum' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr  1e-4   \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    \
    --wa_learable 1 --wa_initial_weights '1,1,1' \
    \
    --gpu 3 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &




ex_id='4-waL118-5e4-info-qwen2.5Emb-lr_scheduler'
model_name='TEM_CS_0204_QuerySum' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr 5e-4 --early_stop 15 \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 1   \
    --lr_scheduler 1   \
    \
    --InfoNCE_kw2item_alpha 0.2 \
    --kw_items_file_name 'src_inv_index_v2.pkl' \
    --InfoNCE_kw2item_batchsize 512 \
    --InfoNCE_kw2item_temp 0.1 \
    --loss_kw2item_lambda 1.0 \
    --loss_item2kw_lambda 0.0 \
    --kw2item_neg_num 512 \
    --kw2item_neg_sample_mode 'in-batch' \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    \
    --wa_learable 1 --wa_initial_weights '1,1,8' \
    \
    --gpu 7 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &



ex_id='5-waL118-5e4-info-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0204_QuerySum' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr 5e-4 --early_stop 15 \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --InfoNCE_kw2item_alpha 0.2 \
    --kw_items_file_name 'src_inv_index_v2.pkl' \
    --InfoNCE_kw2item_batchsize 512 \
    --InfoNCE_kw2item_temp 0.1 \
    --loss_kw2item_lambda 1.0 \
    --loss_item2kw_lambda 0.0 \
    --kw2item_neg_num 512 \
    --kw2item_neg_sample_mode 'in-batch' \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    \
    --wa_learable 1 --wa_initial_weights '1,1,8' \
    \
    --gpu 6 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &




############## TEM_CS_0206_QuerySum ################


ex_id='0-qwen2.5Emb-lr_scheduler'
model_name='TEM_CS_0206_QuerySum' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr  1e-4   \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 1   \
    --lr_scheduler 1   \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    \
    \
    --wa_learable 0 --wa_initial_weights '1,1,5' \
    \
    --gpu 1 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &


ex_id='1-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0206_QuerySum' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr  1e-4   \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    \
    \
    --wa_learable 0 --wa_initial_weights '1,1,5' \
    \
    --gpu 0 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &




ex_id='2-wa118-5e4-info-qwen2.5Emb-lr_scheduler'
model_name='TEM_CS_0206_QuerySum' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr 5e-4 --early_stop 15 \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 1   \
    --lr_scheduler 1   \
    \
    --InfoNCE_kw2item_alpha 0.2 \
    --kw_items_file_name 'src_inv_index_v2.pkl' \
    --InfoNCE_kw2item_batchsize 512 \
    --InfoNCE_kw2item_temp 0.1 \
    --loss_kw2item_lambda 1.0 \
    --loss_item2kw_lambda 0.0 \
    --kw2item_neg_num 512 \
    --kw2item_neg_sample_mode 'in-batch' \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    \
    --wa_learable 0 --wa_initial_weights '1,1,8' \
    \
    --gpu 0 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &



ex_id='3-wa118-5e4-info-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0206_QuerySum' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr 5e-4 --early_stop 15 \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --InfoNCE_kw2item_alpha 0.2 \
    --kw_items_file_name 'src_inv_index_v2.pkl' \
    --InfoNCE_kw2item_batchsize 512 \
    --InfoNCE_kw2item_temp 0.1 \
    --loss_kw2item_lambda 1.0 \
    --loss_item2kw_lambda 0.0 \
    --kw2item_neg_num 512 \
    --kw2item_neg_sample_mode 'in-batch' \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    \
    --wa_learable 0 --wa_initial_weights '1,1,8' \
    \
    --gpu 1 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &


ex_id='3.cp-wa118-5e4-info-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0206_QuerySum' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr 5e-4 --early_stop 15 \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --InfoNCE_kw2item_alpha 0.2 \
    --kw_items_file_name 'src_inv_index_v2.pkl' \
    --InfoNCE_kw2item_batchsize 512 \
    --InfoNCE_kw2item_temp 0.1 \
    --loss_kw2item_lambda 1.0 \
    --loss_item2kw_lambda 0.0 \
    --kw2item_neg_num 512 \
    --kw2item_neg_sample_mode 'in-batch' \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    \
    --wa_learable 0 --wa_initial_weights '1,1,8' \
    \
    --gpu 1 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &



ex_id='3.iname-wa118-5e4-info-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0206_QuerySum' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr 5e-4 --early_stop 15 \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict_iname \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --InfoNCE_kw2item_alpha 0.2 \
    --kw_items_file_name 'src_inv_index_v2.pkl' \
    --InfoNCE_kw2item_batchsize 512 \
    --InfoNCE_kw2item_temp 0.1 \
    --loss_kw2item_lambda 1.0 \
    --loss_item2kw_lambda 0.0 \
    --kw2item_neg_num 512 \
    --kw2item_neg_sample_mode 'in-batch' \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    \
    --wa_learable 0 --wa_initial_weights '1,1,8' \
    \
    --gpu 0 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &



# 
ex_id='3.iname-exllm-wa118-5e4-info-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0206_QuerySum' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr 5e-4 --early_stop 15 \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict_iname \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --InfoNCE_kw2item_alpha 0.2 \
    --kw_items_file_name 'src_inv_index_v2.pkl' \
    --InfoNCE_kw2item_batchsize 512 \
    --InfoNCE_kw2item_temp 0.1 \
    --loss_kw2item_lambda 1.0 \
    --loss_item2kw_lambda 0.0 \
    --kw2item_neg_num 512 \
    --kw2item_neg_sample_mode 'in-batch' \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    \
    --wa_learable 0 --wa_initial_weights '1,1,8' \
    \
    --llm_text_features item_name --use_extra_llm_emb 1 \
    \
    --gpu 2 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &




ex_id='3.iname-multiAlign-exllm-wa118-5e4-info-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0206_QuerySum' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr 5e-4 --early_stop 15 \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict_iname \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --InfoNCE_kw2item_alpha 0.2 \
    --kw_items_file_name 'src_inv_index_v2.pkl' \
    --InfoNCE_kw2item_batchsize 512 \
    --InfoNCE_kw2item_temp 0.1 \
    --loss_kw2item_lambda 1.0 \
    --loss_item2kw_lambda 0.0 \
    --kw2item_neg_num 512 \
    --kw2item_neg_sample_mode 'in-batch' \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    \
    --wa_learable 0 --wa_initial_weights '1,1,8' \
    \
    --llm_text_features item_name --use_extra_llm_emb 1 \
    \
    --multi_align_alpha 0.05 --multi_align_emb_l2 0 --multi_align_temperature 0.07 \
    \
    --gpu 1 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &








ex_id='3.iname-wa118-5e4-inforec-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0206_QuerySum' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr 5e-4 --early_stop 15 \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict_iname \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --InfoNCE_kw2item_alpha 0.2 \
    --kw_items_file_name 'rec_inv_index_v2.pkl' \
    --InfoNCE_kw2item_batchsize 512 \
    --InfoNCE_kw2item_temp 0.1 \
    --loss_kw2item_lambda 1.0 \
    --loss_item2kw_lambda 0.0 \
    --kw2item_neg_num 512 \
    --kw2item_neg_sample_mode 'in-batch' \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    \
    --wa_learable 0 --wa_initial_weights '1,1,8' \
    \
    --gpu 0 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &



ex_id='3.iname-exllm-wa118-5e4-inforec-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0206_QuerySum' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr 5e-4 --early_stop 15 \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict_iname \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --InfoNCE_kw2item_alpha 0.2 \
    --kw_items_file_name 'rec_inv_index_v2.pkl' \
    --InfoNCE_kw2item_batchsize 512 \
    --InfoNCE_kw2item_temp 0.1 \
    --loss_kw2item_lambda 1.0 \
    --loss_item2kw_lambda 0.0 \
    --kw2item_neg_num 512 \
    --kw2item_neg_sample_mode 'in-batch' \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    \
    --wa_learable 0 --wa_initial_weights '1,1,8' \
    \
    --llm_text_features item_name --use_extra_llm_emb 1 \
    \
    --gpu 1 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &




ex_id='3.iname-wa118-5e4-infoall-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0206_QuerySum' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr 5e-4 --early_stop 15 \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict_iname \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --InfoNCE_kw2item_alpha 0.2 \
    --kw_items_file_name 'all_inv_index_v2.pkl' \
    --InfoNCE_kw2item_batchsize 512 \
    --InfoNCE_kw2item_temp 0.1 \
    --loss_kw2item_lambda 1.0 \
    --loss_item2kw_lambda 0.0 \
    --kw2item_neg_num 512 \
    --kw2item_neg_sample_mode 'in-batch' \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    \
    --wa_learable 0 --wa_initial_weights '1,1,8' \
    \
    --gpu 0 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &



ex_id='3.iname-exllm-wa118-5e4-infoall-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0206_QuerySum' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr 5e-4 --early_stop 15 \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict_iname \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --InfoNCE_kw2item_alpha 0.2 \
    --kw_items_file_name 'all_inv_index_v2.pkl' \
    --InfoNCE_kw2item_batchsize 512 \
    --InfoNCE_kw2item_temp 0.1 \
    --loss_kw2item_lambda 1.0 \
    --loss_item2kw_lambda 0.0 \
    --kw2item_neg_num 512 \
    --kw2item_neg_sample_mode 'in-batch' \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    \
    --wa_learable 0 --wa_initial_weights '1,1,8' \
    \
    --llm_text_features item_name --use_extra_llm_emb 1 \
    \
    --gpu 1 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &



ex_id='3.iname-exllm-wa118-5e4-infosrcv3-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0206_QuerySum' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr 5e-4 --early_stop 15 \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict_iname \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --InfoNCE_kw2item_alpha 0.2 \
    --kw_items_file_name 'src_inv_index_v3.pkl' \
    --InfoNCE_kw2item_batchsize 512 \
    --InfoNCE_kw2item_temp 0.1 \
    --loss_kw2item_lambda 1.0 \
    --loss_item2kw_lambda 0.0 \
    --kw2item_neg_num 512 \
    --kw2item_neg_sample_mode 'in-batch' \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    \
    --wa_learable 0 --wa_initial_weights '1,1,8' \
    \
    --llm_text_features item_name --use_extra_llm_emb 1 \
    \
    --gpu 1 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &


ex_id='3.iname-exllm-wa118-1e3-infosrcv3-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0206_QuerySum' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr 1e-3 --early_stop 15 \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict_iname \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --InfoNCE_kw2item_alpha 0.2 \
    --kw_items_file_name 'src_inv_index_v3.pkl' \
    --InfoNCE_kw2item_batchsize 512 \
    --InfoNCE_kw2item_temp 0.1 \
    --loss_kw2item_lambda 1.0 \
    --loss_item2kw_lambda 0.0 \
    --kw2item_neg_num 512 \
    --kw2item_neg_sample_mode 'in-batch' \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    \
    --wa_learable 0 --wa_initial_weights '1,1,8' \
    \
    --llm_text_features item_name --use_extra_llm_emb 1 \
    \
    --gpu 0 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &





ex_id='4-waL118-5e4-info-qwen2.5Emb-lr_scheduler'
model_name='TEM_CS_0206_QuerySum' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr 5e-4 --early_stop 15 \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 1   \
    --lr_scheduler 1   \
    \
    --InfoNCE_kw2item_alpha 0.2 \
    --kw_items_file_name 'src_inv_index_v2.pkl' \
    --InfoNCE_kw2item_batchsize 512 \
    --InfoNCE_kw2item_temp 0.1 \
    --loss_kw2item_lambda 1.0 \
    --loss_item2kw_lambda 0.0 \
    --kw2item_neg_num 512 \
    --kw2item_neg_sample_mode 'in-batch' \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    \
    --wa_learable 1 --wa_initial_weights '1,1,8' \
    \
    --gpu 2 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &



ex_id='5-waL118-5e4-info-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0206_QuerySum' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr 5e-4 --early_stop 15 \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --InfoNCE_kw2item_alpha 0.2 \
    --kw_items_file_name 'src_inv_index_v2.pkl' \
    --InfoNCE_kw2item_batchsize 512 \
    --InfoNCE_kw2item_temp 0.1 \
    --loss_kw2item_lambda 1.0 \
    --loss_item2kw_lambda 0.0 \
    --kw2item_neg_num 512 \
    --kw2item_neg_sample_mode 'in-batch' \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    \
    --wa_learable 1 --wa_initial_weights '1,1,8' \
    \
    --gpu 3 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &


# CUDA out of memory error
ex_id='6-moev1-waL118-5e4-info-qwen2.5Emb-lr_scheduler'
model_name='TEM_CS_0206_QuerySum' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr 5e-4 --early_stop 15 \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 1   \
    --lr_scheduler 1   \
    \
    --InfoNCE_kw2item_alpha 0.2 \
    --kw_items_file_name 'src_inv_index_v2.pkl' \
    --InfoNCE_kw2item_batchsize 512 \
    --InfoNCE_kw2item_temp 0.1 \
    --loss_kw2item_lambda 1.0 \
    --loss_item2kw_lambda 0.0 \
    --kw2item_neg_num 512 \
    --kw2item_neg_sample_mode 'in-batch' \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    \
    --wa_learable 1 --wa_initial_weights '1,1,8' \
    \
    --use_moev1 1 --moev1_top_k 3 --moev1_num_experts 5 \
    \
    --gpu 7 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &


ex_id='7-moev1-waL118-5e4-info-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0206_QuerySum' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr 5e-4 --early_stop 15 \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --InfoNCE_kw2item_alpha 0.2 \
    --kw_items_file_name 'src_inv_index_v2.pkl' \
    --InfoNCE_kw2item_batchsize 512 \
    --InfoNCE_kw2item_temp 0.1 \
    --loss_kw2item_lambda 1.0 \
    --loss_item2kw_lambda 0.0 \
    --kw2item_neg_num 512 \
    --kw2item_neg_sample_mode 'in-batch' \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    \
    --wa_learable 1 --wa_initial_weights '1,1,8' \
    \
    --use_moev1 1 --moev1_top_k 3 --moev1_num_experts 5 \
    \
    --gpu 0 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &




ex_id='7.iname-moev1-waL118-5e4-info-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0206_QuerySum' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr 5e-4 --early_stop 15 \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict_iname \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --InfoNCE_kw2item_alpha 0.2 \
    --kw_items_file_name 'src_inv_index_v2.pkl' \
    --InfoNCE_kw2item_batchsize 512 \
    --InfoNCE_kw2item_temp 0.1 \
    --loss_kw2item_lambda 1.0 \
    --loss_item2kw_lambda 0.0 \
    --kw2item_neg_num 512 \
    --kw2item_neg_sample_mode 'in-batch' \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    \
    --wa_learable 1 --wa_initial_weights '1,1,8' \
    \
    --use_moev1 1 --moev1_top_k 3 --moev1_num_experts 5 \
    \
    --gpu 0 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &



ex_id='8.iname-moev1-waL118-5e4-infol2-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0206_QuerySum' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr 5e-4 --early_stop 15 \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict_iname \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --InfoNCE_kw2item_alpha 0.2 \
    --kw_items_file_name 'src_inv_index_v2.pkl' --InfoNCE_kw2item_l2 1 \
    --InfoNCE_kw2item_batchsize 512 \
    --InfoNCE_kw2item_temp 0.1 \
    --loss_kw2item_lambda 1.0 \
    --loss_item2kw_lambda 0.0 \
    --kw2item_neg_num 512 \
    --kw2item_neg_sample_mode 'in-batch' \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    \
    --wa_learable 1 --wa_initial_weights '1,1,8' \
    \
    --use_moev1 1 --moev1_top_k 3 --moev1_num_experts 5 \
    \
    --gpu 1 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &


ex_id='7.iname-moev1-exllm-waL118-5e4-info-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0206_QuerySum' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr 5e-4 --early_stop 15 \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict_iname \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --InfoNCE_kw2item_alpha 0.2 \
    --kw_items_file_name 'src_inv_index_v2.pkl' \
    --InfoNCE_kw2item_batchsize 512 \
    --InfoNCE_kw2item_temp 0.1 \
    --loss_kw2item_lambda 1.0 \
    --loss_item2kw_lambda 0.0 \
    --kw2item_neg_num 512 \
    --kw2item_neg_sample_mode 'in-batch' \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    --wa_learable 1 --wa_initial_weights '1,1,8' \
    --llm_text_features item_name --use_extra_llm_emb 1 \
    --use_moev1 1 --moev1_top_k 3 --moev1_num_experts 5 \
    \
    --gpu 1 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &



ex_id='8.iname-moev1-exllm-waL118-5e4-infol2-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0206_QuerySum' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr 5e-4 --early_stop 15 \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict_iname \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --InfoNCE_kw2item_alpha 0.2 \
    --kw_items_file_name 'src_inv_index_v2.pkl' --InfoNCE_kw2item_l2 1 \
    --InfoNCE_kw2item_batchsize 512 \
    --InfoNCE_kw2item_temp 0.1 \
    --loss_kw2item_lambda 1.0 \
    --loss_item2kw_lambda 0.0 \
    --kw2item_neg_num 512 \
    --kw2item_neg_sample_mode 'in-batch' \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    --wa_learable 1 --wa_initial_weights '1,1,8' \
    --llm_text_features item_name --use_extra_llm_emb 1 \
    --use_moev1 1 --moev1_top_k 3 --moev1_num_experts 5 \
    \
    --gpu 2 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &



ex_id='7.iname-AddUser2-moev1-exllm-waL118-5e4-info-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0206_QuerySum' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr 5e-4 --early_stop 15 \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict_iname \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --InfoNCE_kw2item_alpha 0.2 \
    --kw_items_file_name 'src_inv_index_v2.pkl' \
    --InfoNCE_kw2item_batchsize 512 \
    --InfoNCE_kw2item_temp 0.1 \
    --loss_kw2item_lambda 1.0 \
    --loss_item2kw_lambda 0.0 \
    --kw2item_neg_num 512 \
    --kw2item_neg_sample_mode 'in-batch' \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    --wa_learable 1 --wa_initial_weights '1,1,8' \
    --llm_text_features item_name --use_extra_llm_emb 1 \
    --use_moev1 1 --moev1_top_k 3 --moev1_num_experts 5 \
    --add_user_emb_mode 'uni_query_emb'\
    \
    --gpu 7 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &


ex_id='7.iname-AddUser3-moev1-exllm-waL118-5e4-info-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0206_QuerySum' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr 5e-4 --early_stop 15 \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict_iname \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --InfoNCE_kw2item_alpha 0.2 \
    --kw_items_file_name 'src_inv_index_v2.pkl' \
    --InfoNCE_kw2item_batchsize 512 \
    --InfoNCE_kw2item_temp 0.1 \
    --loss_kw2item_lambda 1.0 \
    --loss_item2kw_lambda 0.0 \
    --kw2item_neg_num 512 \
    --kw2item_neg_sample_mode 'in-batch' \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    --wa_learable 1 --wa_initial_weights '1,1,8' \
    --llm_text_features item_name --use_extra_llm_emb 1 \
    --use_moev1 1 --moev1_top_k 3 --moev1_num_experts 5 \
    --add_user_emb_mode 'seq_output'\
    \
    --gpu 5 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &



ex_id='7.cp.iname-AddUser3-moev1-exllm-waL118-5e4-info-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0206_QuerySum' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 100 --data $data   \
    --lr 5e-4 --early_stop 15 \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict_iname \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --InfoNCE_kw2item_alpha 0.2 \
    --kw_items_file_name 'src_inv_index_v2.pkl' \
    --InfoNCE_kw2item_batchsize 512 \
    --InfoNCE_kw2item_temp 0.1 \
    --loss_kw2item_lambda 1.0 \
    --loss_item2kw_lambda 0.0 \
    --kw2item_neg_num 512 \
    --kw2item_neg_sample_mode 'in-batch' \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    --wa_learable 1 --wa_initial_weights '1,1,8' \
    --llm_text_features item_name --use_extra_llm_emb 1 \
    --use_moev1 1 --moev1_top_k 3 --moev1_num_experts 5 \
    --add_user_emb_mode 'seq_output'\
    \
    --gpu 0 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &





# ex_id='9.iname-multiAlign-moev1-waL118-5e4-info-noLlmEmb-lr_scheduler'
# model_name='TEM_CS_0206_QuerySum' 
# data='PersonalWAB'
# runner='SrcRunner'
# current_time=$(date +"%Y-%m-%d_%H:%M:%S")
# mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
# nohup python3 main.py --model ${model_name} --runner $runner   \
#     --suffix ${ex_id} --epoch 50 --data $data   \
#     --lr 5e-4 --early_stop 15 \
#     \
#     --batch_size 72   --eval_batch_size 1   \
#     --negs_num 10  --preSampleNeg4Infer 1   \
#     \
#     --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
#     --active_features_dict_name  active_features_dict_iname \
#     --emb_act try_tanh \
#     --use_LLMTokenEncoder_mlp 0   \
#     --use_llm_token_emb_mode 0   \
#     --lr_scheduler 1   \
#     \
#     --InfoNCE_kw2item_alpha 0.2 \
#     --kw_items_file_name 'src_inv_index_v2.pkl' \
#     --InfoNCE_kw2item_batchsize 512 \
#     --InfoNCE_kw2item_temp 0.1 \
#     --loss_kw2item_lambda 1.0 \
#     --loss_item2kw_lambda 0.0 \
#     --kw2item_neg_num 512 \
#     --kw2item_neg_sample_mode 'in-batch' \
#     \
#     --use_preprocessed_conv_his_of_src_session 0 \
#     \
#     --wa_learable 1 --wa_initial_weights '1,1,8' \
#     \
#     --use_moev1 1 --moev1_top_k 3 --moev1_num_experts 5 \
#     \
#     --multi_align_alpha 0.05 --multi_align_emb_l2 0 --multi_align_temperature 0.07 \
#     \
#     --gpu 1 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &



ex_id='9.iname-multiAlign-moev1-exllm-waL118-5e4-info-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0206_QuerySum' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr 5e-4 --early_stop 15 \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict_iname \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --InfoNCE_kw2item_alpha 0.2 \
    --kw_items_file_name 'src_inv_index_v2.pkl' \
    --InfoNCE_kw2item_batchsize 512 \
    --InfoNCE_kw2item_temp 0.1 \
    --loss_kw2item_lambda 1.0 \
    --loss_item2kw_lambda 0.0 \
    --kw2item_neg_num 512 \
    --kw2item_neg_sample_mode 'in-batch' \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    --wa_learable 1 --wa_initial_weights '1,1,8' \
    --use_moev1 1 --moev1_top_k 3 --moev1_num_experts 5 \
    --llm_text_features item_name --use_extra_llm_emb 1 \
    --multi_align_alpha 0.05 --multi_align_emb_l2 0 --multi_align_temperature 0.07 \
    \
    --gpu 0 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &



ex_id='9.iname-multiAlign-AddUser2-moev1-exllm-waL118-5e4-info-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0206_QuerySum' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr 5e-4 --early_stop 15 \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict_iname \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --InfoNCE_kw2item_alpha 0.2 \
    --kw_items_file_name 'src_inv_index_v2.pkl' \
    --InfoNCE_kw2item_batchsize 512 \
    --InfoNCE_kw2item_temp 0.1 \
    --loss_kw2item_lambda 1.0 \
    --loss_item2kw_lambda 0.0 \
    --kw2item_neg_num 512 \
    --kw2item_neg_sample_mode 'in-batch' \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    --wa_learable 1 --wa_initial_weights '1,1,8' \
    --use_moev1 1 --moev1_top_k 3 --moev1_num_experts 5 \
    --llm_text_features item_name --use_extra_llm_emb 1 \
    --add_user_emb_mode 'uni_query_emb'\
    --multi_align_alpha 0.05 --multi_align_emb_l2 0 --multi_align_temperature 0.07 \
    \
    --gpu 2 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &




ex_id='9.iname-multiAlign-AddUser3-moev1-exllm-waL118-5e4-info-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0206_QuerySum' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 120 --data $data   \
    --lr 5e-4 --early_stop 20 \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict_iname \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --InfoNCE_kw2item_alpha 0.2 \
    --kw_items_file_name 'src_inv_index_v2.pkl' \
    --InfoNCE_kw2item_batchsize 512 \
    --InfoNCE_kw2item_temp 0.1 \
    --loss_kw2item_lambda 1.0 \
    --loss_item2kw_lambda 0.0 \
    --kw2item_neg_num 512 \
    --kw2item_neg_sample_mode 'in-batch' \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    --wa_learable 1 --wa_initial_weights '1,1,8' \
    --use_moev1 1 --moev1_top_k 3 --moev1_num_experts 5 \
    --llm_text_features item_name --use_extra_llm_emb 1 \
    --add_user_emb_mode 'seq_output' \
    --multi_align_alpha 0.05 --multi_align_emb_l2 0 --multi_align_temperature 0.07 \
    \
    --gpu 3 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &



ex_id='9.cp.iname-multiAlign-AddUser3-moev1-exllm-waL118-5e4-info-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0206_QuerySum' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 120 --data $data   \
    --lr 5e-4 --early_stop 15 \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict_iname \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --InfoNCE_kw2item_alpha 0.2 \
    --kw_items_file_name 'src_inv_index_v2.pkl' \
    --InfoNCE_kw2item_batchsize 512 \
    --InfoNCE_kw2item_temp 0.1 \
    --loss_kw2item_lambda 1.0 \
    --loss_item2kw_lambda 0.0 \
    --kw2item_neg_num 512 \
    --kw2item_neg_sample_mode 'in-batch' \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    --wa_learable 1 --wa_initial_weights '1,1,8' \
    --use_moev1 1 --moev1_top_k 3 --moev1_num_experts 5 \
    --llm_text_features item_name --use_extra_llm_emb 1 \
    --add_user_emb_mode 'seq_output' \
    --multi_align_alpha 0.02 --multi_align_emb_l2 0 --multi_align_temperature 0.07 \
    \
    --gpu 1 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &



ex_id='9.cp2.iname-multiAlign-AddUser3-moev1-exllm-waL118-5e4-info-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0206_QuerySum' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 120 --data $data   \
    --lr 5e-4 --early_stop 15 \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict_iname \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --InfoNCE_kw2item_alpha 0.2 \
    --kw_items_file_name 'src_inv_index_v2.pkl' \
    --InfoNCE_kw2item_batchsize 512 \
    --InfoNCE_kw2item_temp 0.1 \
    --loss_kw2item_lambda 1.0 \
    --loss_item2kw_lambda 0.0 \
    --kw2item_neg_num 512 \
    --kw2item_neg_sample_mode 'in-batch' \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    --wa_learable 1 --wa_initial_weights '1,1,8' \
    --use_moev1 1 --moev1_top_k 3 --moev1_num_experts 5 \
    --llm_text_features item_name --use_extra_llm_emb 1 \
    --add_user_emb_mode 'seq_output' \
    --multi_align_alpha 0.002 --multi_align_emb_l2 0 --multi_align_temperature 0.07 \
    \
    --gpu 1 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &



ex_id='10.cp2.iname-multiAlign-AddUser3-moev2-exllm-waL118-5e4-info-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0206_QuerySum' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 120 --data $data   \
    --lr 5e-4 --early_stop 15 \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict_iname \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --InfoNCE_kw2item_alpha 0.2 \
    --kw_items_file_name 'src_inv_index_v2.pkl' \
    --InfoNCE_kw2item_batchsize 512 \
    --InfoNCE_kw2item_temp 0.1 \
    --loss_kw2item_lambda 1.0 \
    --loss_item2kw_lambda 0.0 \
    --kw2item_neg_num 512 \
    --kw2item_neg_sample_mode 'in-batch' \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    --wa_learable 1 --wa_initial_weights '1,1,8' \
    --use_moev1 1 --moev1_top_k 4 --moev1_num_experts 6 --switch2MoEv2 1 \
    --llm_text_features item_name --use_extra_llm_emb 1 \
    --add_user_emb_mode 'seq_output' \
    --multi_align_alpha 0.002 --multi_align_emb_l2 0 --multi_align_temperature 0.07 \
    \
    --gpu 0 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &









###########################  TEM_CS_0208_QuerySum_PredMlp #################







ex_id='3.iname-exllm-wa118-5e4-info-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0208_QuerySum_PredMlp' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr 5e-4 --early_stop 15 \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict_iname \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --InfoNCE_kw2item_alpha 0.2 \
    --kw_items_file_name 'src_inv_index_v2.pkl' \
    --InfoNCE_kw2item_batchsize 512 \
    --InfoNCE_kw2item_temp 0.1 \
    --loss_kw2item_lambda 1.0 \
    --loss_item2kw_lambda 0.0 \
    --kw2item_neg_num 512 \
    --kw2item_neg_sample_mode 'in-batch' \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    \
    --wa_learable 0 --wa_initial_weights '1,1,8' \
    \
    --llm_text_features item_name --use_extra_llm_emb 1 \
    \
    --pred_mlp 1 --pred_hid_units '64,1' --pred_mlp_act 'relu' \
    \
    --gpu 2 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &


ex_id='4.iname-exllm-wa118-5e4-info-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0208_QuerySum_PredMlp' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr 5e-4 --early_stop 15 \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict_iname \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --InfoNCE_kw2item_alpha 0.2 \
    --kw_items_file_name 'src_inv_index_v2.pkl' \
    --InfoNCE_kw2item_batchsize 512 \
    --InfoNCE_kw2item_temp 0.1 \
    --loss_kw2item_lambda 1.0 \
    --loss_item2kw_lambda 0.0 \
    --kw2item_neg_num 512 \
    --kw2item_neg_sample_mode 'in-batch' \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    \
    --wa_learable 0 --wa_initial_weights '1,1,8' \
    \
    --llm_text_features item_name --use_extra_llm_emb 1 \
    \
    --pred_mlp 1 --pred_hid_units '64,1' --pred_mlp_act 'prelu' \
    \
    --gpu 3 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &



#############  TEM_CS_0206_QuerySum_UserEmb



ex_id='1.iname-AddUser1-exllm-wa118-5e4-info-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0206_QuerySum_UserEmb' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr 5e-4 --early_stop 15 \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict_iname \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --InfoNCE_kw2item_alpha 0.2 \
    --kw_items_file_name 'src_inv_index_v2.pkl' \
    --InfoNCE_kw2item_batchsize 512 \
    --InfoNCE_kw2item_temp 0.1 \
    --loss_kw2item_lambda 1.0 \
    --loss_item2kw_lambda 0.0 \
    --kw2item_neg_num 512 \
    --kw2item_neg_sample_mode 'in-batch' \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    \
    --wa_learable 0 --wa_initial_weights '1,1,8' \
    \
    --llm_text_features item_name --use_extra_llm_emb 1 \
    \
    --add_user_emb_mode 'first'\
    \
    --gpu 5 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &





ex_id='2.iname-AddUser2-exllm-wa118-5e4-info-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0206_QuerySum_UserEmb' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr 5e-4 --early_stop 15 \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict_iname \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0  \
    --use_llm_token_emb_mode 0  \
    --lr_scheduler 1  \
    \
    --InfoNCE_kw2item_alpha 0.2 \
    --kw_items_file_name 'src_inv_index_v2.pkl' \
    --InfoNCE_kw2item_batchsize 512 \
    --InfoNCE_kw2item_temp 0.1 \
    --loss_kw2item_lambda 1.0 \
    --loss_item2kw_lambda 0.0 \
    --kw2item_neg_num 512 \
    --kw2item_neg_sample_mode 'in-batch' \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    \
    --wa_learable 0 --wa_initial_weights '1,1,8' \
    \
    --llm_text_features item_name --use_extra_llm_emb 1 \
    \
    --add_user_emb_mode 'uni_query_emb'\
    \
    --gpu 6 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &



ex_id='3.iname-AddUser3-exllm-wa118-5e4-info-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0206_QuerySum_UserEmb' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 50 --data $data   \
    --lr 5e-4 --early_stop 15 \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict_iname \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --InfoNCE_kw2item_alpha 0.2 \
    --kw_items_file_name 'src_inv_index_v2.pkl' \
    --InfoNCE_kw2item_batchsize 512 \
    --InfoNCE_kw2item_temp 0.1 \
    --loss_kw2item_lambda 1.0 \
    --loss_item2kw_lambda 0.0 \
    --kw2item_neg_num 512 \
    --kw2item_neg_sample_mode 'in-batch' \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    \
    --wa_learable 0 --wa_initial_weights '1,1,8' \
    \
    --llm_text_features item_name --use_extra_llm_emb 1 \
    \
    --add_user_emb_mode 'seq_output'\
    \
    --gpu 7 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &





ex_id='9(src_inv_index_v1).iname-multiAlign-AddUser3-moev1-exllm-waL118-5e4-info-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0206_QuerySum' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 120 --data $data   \
    --lr 5e-4 --early_stop 20 \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict_iname \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --InfoNCE_kw2item_alpha 0.2 \
    --kw_items_file_name 'src_inv_index.pkl' \
    --InfoNCE_kw2item_batchsize 512 \
    --InfoNCE_kw2item_temp 0.1 \
    --loss_kw2item_lambda 1.0 \
    --loss_item2kw_lambda 0.0 \
    --kw2item_neg_num 512 \
    --kw2item_neg_sample_mode 'in-batch' \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    --wa_learable 1 --wa_initial_weights '1,1,8' \
    --use_moev1 1 --moev1_top_k 3 --moev1_num_experts 5 \
    --llm_text_features item_name --use_extra_llm_emb 1 \
    --add_user_emb_mode 'seq_output' \
    --multi_align_alpha 0.05 --multi_align_emb_l2 0 --multi_align_temperature 0.07 \
    \
    --gpu 3 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &



ex_id='9(rec_inv_index_v1).iname-multiAlign-AddUser3-moev1-exllm-waL118-5e4-info-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0206_QuerySum' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 120 --data $data   \
    --lr 5e-4 --early_stop 20 \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict_iname \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --InfoNCE_kw2item_alpha 0.2 \
    --kw_items_file_name 'rec_inv_index.pkl' \
    --InfoNCE_kw2item_batchsize 512 \
    --InfoNCE_kw2item_temp 0.1 \
    --loss_kw2item_lambda 1.0 \
    --loss_item2kw_lambda 0.0 \
    --kw2item_neg_num 512 \
    --kw2item_neg_sample_mode 'in-batch' \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    --wa_learable 1 --wa_initial_weights '1,1,8' \
    --use_moev1 1 --moev1_top_k 3 --moev1_num_experts 5 \
    --llm_text_features item_name --use_extra_llm_emb 1 \
    --add_user_emb_mode 'seq_output' \
    --multi_align_alpha 0.05 --multi_align_emb_l2 0 --multi_align_temperature 0.07 \
    \
    --gpu 2 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &




ex_id='9(src_inv_index_v4).iname-multiAlign-AddUser3-moev1-exllm-waL118-5e4-info-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0206_QuerySum' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 120 --data $data   \
    --lr 5e-4 --early_stop 20 \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict_iname \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --InfoNCE_kw2item_alpha 0.2 \
    --kw_items_file_name 'src_inv_index_v4.pkl' \
    --InfoNCE_kw2item_batchsize 512 \
    --InfoNCE_kw2item_temp 0.1 \
    --loss_kw2item_lambda 1.0 \
    --loss_item2kw_lambda 0.0 \
    --kw2item_neg_num 512 \
    --kw2item_neg_sample_mode 'in-batch' \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    --wa_learable 1 --wa_initial_weights '1,1,8' \
    --use_moev1 1 --moev1_top_k 3 --moev1_num_experts 5 \
    --llm_text_features item_name --use_extra_llm_emb 1 \
    --add_user_emb_mode 'seq_output' \
    --multi_align_alpha 0.05 --multi_align_emb_l2 0 --multi_align_temperature 0.07 \
    \
    --gpu 1 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &




ex_id='9(src_inv_index_v5).iname-multiAlign-AddUser3-moev1-exllm-waL118-5e4-info-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0206_QuerySum' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 120 --data $data   \
    --lr 5e-4 --early_stop 20 \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict_iname \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --InfoNCE_kw2item_alpha 0.2 \
    --kw_items_file_name 'src_inv_index_v5.pkl' \
    --InfoNCE_kw2item_batchsize 512 \
    --InfoNCE_kw2item_temp 0.1 \
    --loss_kw2item_lambda 1.0 \
    --loss_item2kw_lambda 0.0 \
    --kw2item_neg_num 512 \
    --kw2item_neg_sample_mode 'in-batch' \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    --wa_learable 1 --wa_initial_weights '1,1,8' \
    --use_moev1 1 --moev1_top_k 3 --moev1_num_experts 5 \
    --llm_text_features item_name --use_extra_llm_emb 1 \
    --add_user_emb_mode 'seq_output' \
    --multi_align_alpha 0.05 --multi_align_emb_l2 0 --multi_align_temperature 0.07 \
    \
    --gpu 6 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &






ex_id='9(src_inv_index_v6).iname-multiAlign-AddUser3-moev1-exllm-waL118-5e4-info-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0206_QuerySum' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 120 --data $data   \
    --lr 5e-4 --early_stop 20 \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict_iname \
    --emb_act try_tanh \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --InfoNCE_kw2item_alpha 0.2 \
    --kw_items_file_name 'src_inv_index_v6.pkl' \
    --InfoNCE_kw2item_batchsize 512 \
    --InfoNCE_kw2item_temp 0.1 \
    --loss_kw2item_lambda 1.0 \
    --loss_item2kw_lambda 0.0 \
    --kw2item_neg_num 512 \
    --kw2item_neg_sample_mode 'in-batch' \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    --wa_learable 1 --wa_initial_weights '1,1,8' \
    --use_moev1 1 --moev1_top_k 3 --moev1_num_experts 5 \
    --llm_text_features item_name --use_extra_llm_emb 1 \
    --add_user_emb_mode 'seq_output' \
    --multi_align_alpha 0.05 --multi_align_emb_l2 0 --multi_align_temperature 0.07 \
    \
    --gpu 7 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &





# ex_id='9.(relu)iname-multiAlign-AddUser3-moev1-exllm-waL118-5e4-info-noLlmEmb-lr_scheduler'
# model_name='TEM_CS_0206_QuerySum' 
# data='PersonalWAB'
# runner='SrcRunner'
# current_time=$(date +"%Y-%m-%d_%H:%M:%S")
# mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
# nohup python3 main.py --model ${model_name} --runner $runner   \
#     --suffix ${ex_id} --epoch 120 --data $data   \
#     --lr 5e-4 --early_stop 20 \
#     \
#     --batch_size 72   --eval_batch_size 1   \
#     --negs_num 10  --preSampleNeg4Infer 1   \
#     \
#     --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
#     --active_features_dict_name  active_features_dict_iname \
#     --emb_act try_relu \
#     --use_LLMTokenEncoder_mlp 0   \
#     --use_llm_token_emb_mode 0   \
#     --lr_scheduler 1   \
#     \
#     --InfoNCE_kw2item_alpha 0.2 \
#     --kw_items_file_name 'src_inv_index_v2.pkl' \
#     --InfoNCE_kw2item_batchsize 512 \
#     --InfoNCE_kw2item_temp 0.1 \
#     --loss_kw2item_lambda 1.0 \
#     --loss_item2kw_lambda 0.0 \
#     --kw2item_neg_num 512 \
#     --kw2item_neg_sample_mode 'in-batch' \
#     \
#     --use_preprocessed_conv_his_of_src_session 0 \
#     --wa_learable 1 --wa_initial_weights '1,1,8' \
#     --use_moev1 1 --moev1_top_k 3 --moev1_num_experts 5 \
#     --llm_text_features item_name --use_extra_llm_emb 1 \
#     --add_user_emb_mode 'seq_output' \
#     --multi_align_alpha 0.05 --multi_align_emb_l2 0 --multi_align_temperature 0.07 \
#     \
#     --gpu 3 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &






ex_id='9.(relu)iname-multiAlign-AddUser3-moev1-exllm-waL118-5e4-info-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0206_QuerySum' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 120 --data $data   \
    --lr 5e-4 --early_stop 20 \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict_iname \
    --emb_act try_relu \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --InfoNCE_kw2item_alpha 0.2 \
    --kw_items_file_name 'src_inv_index_v2.pkl' \
    --InfoNCE_kw2item_batchsize 512 \
    --InfoNCE_kw2item_temp 0.1 \
    --loss_kw2item_lambda 1.0 \
    --loss_item2kw_lambda 0.0 \
    --kw2item_neg_num 512 \
    --kw2item_neg_sample_mode 'in-batch' \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    --wa_learable 1 --wa_initial_weights '1,1,8' \
    --use_moev1 1 --moev1_top_k 3 --moev1_num_experts 5 \
    --llm_text_features item_name --use_extra_llm_emb 1 \
    --add_user_emb_mode 'seq_output' \
    --multi_align_alpha 0.05 --multi_align_emb_l2 0 --multi_align_temperature 0.07 \
    \
    --gpu 0 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &





ex_id='9.(relu_text)iname-multiAlign-AddUser3-moev1-exllm-waL118-5e4-info-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0206_QuerySum' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 120 --data $data   \
    --lr 5e-4 --early_stop 20 \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict_iname \
    --emb_act try_tanh --text_emb_act try_relu \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --InfoNCE_kw2item_alpha 0.2 \
    --kw_items_file_name 'src_inv_index_v2.pkl' \
    --InfoNCE_kw2item_batchsize 512 \
    --InfoNCE_kw2item_temp 0.1 \
    --loss_kw2item_lambda 1.0 \
    --loss_item2kw_lambda 0.0 \
    --kw2item_neg_num 512 \
    --kw2item_neg_sample_mode 'in-batch' \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    --wa_learable 1 --wa_initial_weights '1,1,8' \
    --use_moev1 1 --moev1_top_k 3 --moev1_num_experts 5 \
    --llm_text_features item_name --use_extra_llm_emb 1 \
    --add_user_emb_mode 'seq_output' \
    --multi_align_alpha 0.05 --multi_align_emb_l2 0 --multi_align_temperature 0.07 \
    \
    --gpu 0 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &





ex_id='9.(relu_text)iname-multiAlign-AddUser3-moev1-exllm-waL118-5e4-info-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0206_QuerySum' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 120 --data $data   \
    --lr 5e-4 --early_stop 20 \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name active_features_dict_iname \
    --emb_act try_tanh --text_feat_act try_relu \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --InfoNCE_kw2item_alpha 0.2 \
    --kw_items_file_name 'src_inv_index_v2.pkl' \
    --InfoNCE_kw2item_batchsize 512 \
    --InfoNCE_kw2item_temp 0.1 \
    --loss_kw2item_lambda 1.0 \
    --loss_item2kw_lambda 0.0 \
    --kw2item_neg_num 512 \
    --kw2item_neg_sample_mode 'in-batch' \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    --wa_learable 1 --wa_initial_weights '1,1,8' \
    --use_moev1 1 --moev1_top_k 3 --moev1_num_experts 5 \
    --llm_text_features item_name --use_extra_llm_emb 1 \
    --add_user_emb_mode 'seq_output' \
    --multi_align_alpha 0.05 --multi_align_emb_l2 0 --multi_align_temperature 0.07 \
    \
    --gpu 7 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &




ex_id='9.(relu_object)iname-multiAlign-AddUser3-moev1-exllm-waL118-5e4-info-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0206_QuerySum' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 120 --data $data   \
    --lr 5e-4 --early_stop 20 \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name active_features_dict_iname \
    --emb_act try_tanh --object_feat_act try_relu \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --InfoNCE_kw2item_alpha 0.2 \
    --kw_items_file_name 'src_inv_index_v2.pkl' \
    --InfoNCE_kw2item_batchsize 512 \
    --InfoNCE_kw2item_temp 0.1 \
    --loss_kw2item_lambda 1.0 \
    --loss_item2kw_lambda 0.0 \
    --kw2item_neg_num 512 \
    --kw2item_neg_sample_mode 'in-batch' \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    --wa_learable 1 --wa_initial_weights '1,1,8' \
    --use_moev1 1 --moev1_top_k 3 --moev1_num_experts 5 \
    --llm_text_features item_name --use_extra_llm_emb 1 \
    --add_user_emb_mode 'seq_output' \
    --multi_align_alpha 0.05 --multi_align_emb_l2 0 --multi_align_temperature 0.07 \
    \
    --gpu 0 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &








ex_id='9.(prelu)iname-multiAlign-AddUser3-moev1-exllm-waL118-5e4-info-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0206_QuerySum' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 120 --data $data   \
    --lr 5e-4 --early_stop 20 \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict_iname \
    --emb_act try_prelu \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --InfoNCE_kw2item_alpha 0.2 \
    --kw_items_file_name 'src_inv_index_v2.pkl' \
    --InfoNCE_kw2item_batchsize 512 \
    --InfoNCE_kw2item_temp 0.1 \
    --loss_kw2item_lambda 1.0 \
    --loss_item2kw_lambda 0.0 \
    --kw2item_neg_num 512 \
    --kw2item_neg_sample_mode 'in-batch' \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    --wa_learable 1 --wa_initial_weights '1,1,8' \
    --use_moev1 1 --moev1_top_k 3 --moev1_num_experts 5 \
    --llm_text_features item_name --use_extra_llm_emb 1 \
    --add_user_emb_mode 'seq_output' \
    --multi_align_alpha 0.05 --multi_align_emb_l2 0 --multi_align_temperature 0.07 \
    \
    --gpu 1 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &




ex_id='9.(gelu)iname-multiAlign-AddUser3-moev1-exllm-waL118-5e4-info-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0206_QuerySum' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 120 --data $data   \
    --lr 5e-4 --early_stop 20 \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict_iname \
    --emb_act try_gelu \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --InfoNCE_kw2item_alpha 0.2 \
    --kw_items_file_name 'src_inv_index_v2.pkl' \
    --InfoNCE_kw2item_batchsize 512 \
    --InfoNCE_kw2item_temp 0.1 \
    --loss_kw2item_lambda 1.0 \
    --loss_item2kw_lambda 0.0 \
    --kw2item_neg_num 512 \
    --kw2item_neg_sample_mode 'in-batch' \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    --wa_learable 1 --wa_initial_weights '1,1,8' \
    --use_moev1 1 --moev1_top_k 3 --moev1_num_experts 5 \
    --llm_text_features item_name --use_extra_llm_emb 1 \
    --add_user_emb_mode 'seq_output' \
    --multi_align_alpha 0.05 --multi_align_emb_l2 0 --multi_align_temperature 0.07 \
    \
    --gpu 3 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &



ex_id='9.(silu)iname-multiAlign-AddUser3-moev1-exllm-waL118-5e4-info-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0206_QuerySum' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 120 --data $data   \
    --lr 5e-4 --early_stop 20 \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict_iname \
    --emb_act try_silu \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --InfoNCE_kw2item_alpha 0.2 \
    --kw_items_file_name 'src_inv_index_v2.pkl' \
    --InfoNCE_kw2item_batchsize 512 \
    --InfoNCE_kw2item_temp 0.1 \
    --loss_kw2item_lambda 1.0 \
    --loss_item2kw_lambda 0.0 \
    --kw2item_neg_num 512 \
    --kw2item_neg_sample_mode 'in-batch' \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    --wa_learable 1 --wa_initial_weights '1,1,8' \
    --use_moev1 1 --moev1_top_k 3 --moev1_num_experts 5 \
    --llm_text_features item_name --use_extra_llm_emb 1 \
    --add_user_emb_mode 'seq_output' \
    --multi_align_alpha 0.05 --multi_align_emb_l2 0 --multi_align_temperature 0.07 \
    \
    --gpu 4 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &



ex_id='9.(swiglu)iname-multiAlign-AddUser3-moev1-exllm-waL118-5e4-info-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0206_QuerySum' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 120 --data $data   \
    --lr 5e-4 --early_stop 20 \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict_iname \
    --emb_act try_swiglu \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --InfoNCE_kw2item_alpha 0.2 \
    --kw_items_file_name 'src_inv_index_v2.pkl' \
    --InfoNCE_kw2item_batchsize 512 \
    --InfoNCE_kw2item_temp 0.1 \
    --loss_kw2item_lambda 1.0 \
    --loss_item2kw_lambda 0.0 \
    --kw2item_neg_num 512 \
    --kw2item_neg_sample_mode 'in-batch' \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    --wa_learable 1 --wa_initial_weights '1,1,8' \
    --use_moev1 1 --moev1_top_k 3 --moev1_num_experts 5 \
    --llm_text_features item_name --use_extra_llm_emb 1 \
    --add_user_emb_mode 'seq_output' \
    --multi_align_alpha 0.05 --multi_align_emb_l2 0 --multi_align_temperature 0.07 \
    \
    --gpu 7 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &



ex_id='9.(sigmoid)iname-multiAlign-AddUser3-moev1-exllm-waL118-5e4-info-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0206_QuerySum' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 120 --data $data   \
    --lr 5e-4 --early_stop 20 \
    \
    --batch_size 72   --eval_batch_size 1   \
    --negs_num 10  --preSampleNeg4Infer 1   \
    \
    --llm_path '/data1/qinwc1/pretrained_models/Qwen2.5-7B-Instruct'  \
    --active_features_dict_name  active_features_dict_iname \
    --emb_act try_sigmoid \
    --use_LLMTokenEncoder_mlp 0   \
    --use_llm_token_emb_mode 0   \
    --lr_scheduler 1   \
    \
    --InfoNCE_kw2item_alpha 0.2 \
    --kw_items_file_name 'src_inv_index_v2.pkl' \
    --InfoNCE_kw2item_batchsize 512 \
    --InfoNCE_kw2item_temp 0.1 \
    --loss_kw2item_lambda 1.0 \
    --loss_item2kw_lambda 0.0 \
    --kw2item_neg_num 512 \
    --kw2item_neg_sample_mode 'in-batch' \
    \
    --use_preprocessed_conv_his_of_src_session 0 \
    --wa_learable 1 --wa_initial_weights '1,1,8' \
    --use_moev1 1 --moev1_top_k 3 --moev1_num_experts 5 \
    --llm_text_features item_name --use_extra_llm_emb 1 \
    --add_user_emb_mode 'seq_output' \
    --multi_align_alpha 0.05 --multi_align_emb_l2 0 --multi_align_temperature 0.07 \
    \
    --gpu 7 > "output/${data}/${runner}/${model_name}/${ex_id}/${current_time}.log" 2>&1 &

