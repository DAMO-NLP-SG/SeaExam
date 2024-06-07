model=SeaLLMs/SeaLLM-7B-v2.5

data_names=m3exam
selected_langs=english,chinese

output_dir=outputs

CUDA_VISIBLE_DEVICES=0 python scripts/main.py \
--model $model \
--model_type default \
--data_names $data_names \
--selected_langs $selected_langs \
--num_samples all \
--max_tokens 8 \
--output_dir $output_dir \
--run_inference \
--run_evaluation \
--dynamic_template \
--seed 1