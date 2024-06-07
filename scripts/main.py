from utils import *
from utils_model import *
from utils_data import *
import os, csv
import json
import argparse
from tqdm import tqdm
import concurrent.futures

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

dic_api_functions = {
    "openai": parallel_query_chatgpt_model,
    "claude": parallel_query_claude_model,
    "gemini": parallel_query_gemini_model,
    "together": parallel_query_together_model,
    "azure": parallel_query_chatgpt_model_azure
}

def process_lang(args, data_name, lang, subject, method, api_key):

    output_folder = f"{args.output_dir}/{data_name}/model_{args.model.split('/')[-1]}_{args.generation_mode}/{lang}/{subject}/{method}/"

    if args.overwrite and os.path.exists(f"{output_folder}/{lang}-pred.json"):
        print(f"Prediction exit, will skip: {lang} {subject} {method}...")
        return
    
    # get the questions to test
    test_questions = generate_questions(data_name, lang, args.setting, args.dynamic_template, args.add_space, random_seed=args.seed, repo_path=args.dataset_path)
    if subject != 'all':
        test_questions = [q for q in test_questions if q['metadata']['subject'] == subject]
        
    if len(test_questions) == 0:
        print(f"Skip: {lang} {subject} {method}...")
        return

    if args.num_samples != 'all':
        num_samples = int(args.num_samples)
        test_questions = test_questions[:num_samples]

    os.makedirs(output_folder, exist_ok=True)

    # generate prompts
    all_prompts = [question['prompt'] for question in test_questions]
    
    # inference in batch
    prompt_args = [(api_key, p, args.model, args.max_tokens) for p in all_prompts]

    if args.model_type in dic_api_functions:
        parallel_call = dic_api_functions[args.model_type]
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            predictions = list(tqdm(executor.map(parallel_call, prompt_args), total=len(prompt_args), desc=f"Conducting inference"))
    elif args.model_type == 'vllm':
        if args.generation_mode == 'chat':
            all_prompts = [prompt_to_chatprompt(p, args.tokenizer) for p in all_prompts]
        predictions = get_vllm_completion(args.llm, all_prompts, args.sampling_params)
    elif args.model_type == 'hf':
        if args.generation_mode == 'chat':
            all_prompts = [prompt_to_chatprompt(p, args.tokenizer) for p in all_prompts]
        predictions = get_hf_completion(args.llm, args.tokenizer, all_prompts, max_new_tokens=args.max_tokens, batch_size=args.batch_size)
    else:
        raise ValueError(f"Model type {args.model_type} is not supported.")

    # save the predictions
    for idx, question in enumerate(test_questions):
        question['pred'] = predictions[idx]    # save the pred
        question['prompt'] = all_prompts[idx]    # also save the prompt
    
    with open(f"{output_folder}/{lang}-pred.json", "w") as f:
        json.dump(test_questions, f, indent=2, ensure_ascii=False)
    
    print(f"Done: {len(test_questions)} {lang} questions!")


def run_evaluate(args, data_name, subject, method, lang, generation_mode):

    acc_dict = defaultdict()
    output_folder = f"{args.output_dir}/{data_name}/model_{args.model.split('/')[-1]}_{generation_mode}/{lang}/{subject}/{method}/"

    print('='*50)
    print(f"Run eval on {lang}...")
    pred_file_path = output_folder + f"{lang}-pred.json"
    if os.path.exists(pred_file_path):
        with open(pred_file_path, "r") as f:
            preds = json.load(f)

        acc_scores, errors, illformats, checks = compute_acc_score(preds, args.model)

        acc_dict[lang] = acc_scores
        error_file_path = output_folder + f"{lang}-error.json"
        illformat_file_path = output_folder + f"{lang}-illformat.json"

        with open(error_file_path, 'w') as f:
            json.dump(errors, f, indent=4, ensure_ascii=False)

        with open(illformat_file_path, 'w') as f:
            json.dump(illformats, f, indent=4, ensure_ascii=False)
        
        with open(pred_file_path, 'w') as f:
            json.dump(checks, f, indent=4, ensure_ascii=False)

        #write the result to csv
        acc = acc_scores[1]/acc_scores[0] if acc_scores[0] > 0 else 0
        for csv_file in [f"{output_folder}result.csv", f"{args.output_dir}/eval_result.csv"]:
            with open(csv_file, 'a', newline='') as f:
                results = [data_name, args.model, generation_mode, subject, method, lang, acc_scores[0], acc_scores[1], acc]
                writer = csv.writer(f)
                writer.writerow(results)
        print(results)
    else:
        print(f"Cannot find prediction file: {pred_file_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="SeaLLMs/SeaLLM-7B-v2.5", help="the model to be tested")
    parser.add_argument("--model_type", type=str, default="default", choices="[default, openai, claude, gemini, together, vllm, hf, azure]", help="model type")
    parser.add_argument("--dataset_path", type=str, default="SeaLLMs/SeaExam", help="SeaExam benchmark name or local path")
    parser.add_argument("--data_names", type=str, default="m3exam,mmlu", help="comma seperated names of the data, e.g m3exam,mmlu")
    parser.add_argument("--selected_langs", type=str, default='all', help="list of string of languages")
    parser.add_argument("--selected_subjects", type=str, default='all', help="list of string of subjects")
    parser.add_argument('--api_key', type=str, default=None, help="API key for the model")
    parser.add_argument("--num_samples", type=str, default='all', help="Number of samples to test, set to 'all' to test all samples")
    parser.add_argument("--num_workers", type=int, default=5, help="Number of workers to use, only for API based model")
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1, help="Number of GPUs to use for vllm")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--max_tokens", type=int, default=64, help="max tokens for the model")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size for the model if using hf")
    parser.add_argument("--generation_mode", type=str, default="complete", choices="[chat, complete]", help="generation mode for the model, only for hf and vllm")
    parser.add_argument("--setting", type=str, default="few-shot", help="[few-shot, zero-shot]")
    parser.add_argument("--methods", type=str, default="en-instruct", choices=["en-instruct"],help="the prompting strategies")
    parser.add_argument("--overwrite", action='store_true', help="overwrite the existing results or not")
    parser.add_argument("--run_inference", action='store_true', help="Run inference or not")
    parser.add_argument("--run_evaluation", action='store_true', help="run evaluation or not")
    parser.add_argument("--output_dir", type=str, default="outputs/", help="path for writing the result")
    parser.add_argument("--dynamic_template", action='store_true', help="whether to use dynamic template")
    parser.add_argument("--add_space", type=int, default=1, help="Whether add a space between prompt and completion, 0 for false and 1 for true, it is only effective when dynamic_template is false")
    return parser.parse_args()


def main(args):
    all_langs = ['english', 'chinese', 'indonesian', 'thai', 'vietnamese']
    selected_langs = args.selected_langs.split(',') if args.selected_langs != 'all' else all_langs
    selected_subjects = args.selected_subjects.split(',')
    selected_methods = args.methods.split(',') if args.methods else ['en-instruct']

    dic_api_keys = {
        "openai": os.getenv("OPENAI_API_KEY"),
        "claude": os.getenv("CLAUDE_API_KEY"),
        "gemini": os.getenv("GEMINI_API_KEY"),
        "together": os.getenv("TOGETHER_API_KEY"),
        "azure": os.getenv("AZURE_OPENAI_KEY"),
        "vllm": "vllm",
        "hf": "hf"
    }

    if args.model_type == "default":
        if "gpt-3.5" in args.model or "gpt-4" in args.model:
            args.model_type = "openai"
        elif "gemini" in args.model:
            args.model_type = "gemini"
        elif "claude" in args.model:
            args.model_type = "claude"
        elif args.model in ["aisingapore/sea-lion-7b", "aisingapore/sea-lion-7b-instruct"]:
            args.model_type = "hf"
        else:
            args.model_type = "vllm"

    api_key = dic_api_keys[args.model_type] if args.api_key is None else args.api_key

    if args.model_type == 'vllm' and args.run_inference:
        args.llm, args.sampling_params, args.tokenizer = prepare_vllm(args.model, max_tokens=args.max_tokens, tensor_parallel_size=args.vllm_tensor_parallel_size)
    if args.model_type == 'hf' and args.run_inference:
        args.llm, args.tokenizer = prepare_hf_model(args.model)

    for data_name in args.data_names.split(','):
        for lang in selected_langs:
            for subject in selected_subjects:
                for method in selected_methods:
                    print(f"Processing {args.model} {data_name} {lang} {subject} {method}...")
                    if args.run_inference:
                        process_lang(args, data_name, lang, subject, method, api_key)
                    if args.run_evaluation:
                        run_evaluate(args, data_name, subject, method, lang, args.generation_mode)

    # post process the result
    df = load_df(args.output_dir)
    for data_name in df['data_name'].unique():
        print(f"Summarize the result for {data_name}...")
        print(f"Output to {args.output_dir}/{data_name}/eval_{data_name}_pivot.csv")
        print('-'*50)
        df_new = post_process(df, data_name, args.output_dir)
        print(df_new)
        print('-'*50)


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    main(args)