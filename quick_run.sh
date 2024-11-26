model=SeaLLMs/SeaLLMs-v3-7B-Chat 

data_names=m3exam
selected_langs=sea

output_dir=outputs

CUDA_VISIBLE_DEVICES=0 python scripts/main.py \
--model $model \
--model_type default \
--data_names $data_names \
--selected_langs $selected_langs \
--num_samples all \
--max_tokens 4 \
--overwrite 1 \
--output_dir $output_dir \
--run_inference 1 \
--run_evaluation 1 \
--dynamic_template 1 \
--seed 1