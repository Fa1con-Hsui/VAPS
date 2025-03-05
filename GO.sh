# EXAMPLE:

ex_id='9.(prelu)iname-multiAlign-AddUser3-moev1-exllm-waL118-5e4-info-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0206_Final' 
data='PersonalWAB'
runner='SrcRunner'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
mkdir -p output/${data}/${runner}/${model_name}/${ex_id}
nohup python3 main.py --model ${model_name} --runner $runner   \
    --suffix ${ex_id} --epoch 100 --data $data   \
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




ex_id='7.cp.iname-AddUser3-moev1-exllm-waL118-5e4-info-noLlmEmb-lr_scheduler'
model_name='TEM_CS_0206_Final' 
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


