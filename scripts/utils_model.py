import os, torch
from openai import OpenAI
from openai import AzureOpenAI
import requests
from tqdm import tqdm
import numpy as np
from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
    wait_random
)
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import anthropic
from transformers import AutoTokenizer, AutoModelForCausalLM
import google.generativeai as genai


def prepare_vllm(model, temperature=0, max_tokens=64, tensor_parallel_size=1):
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    llm = LLM(model=model, tensor_parallel_size=tensor_parallel_size, trust_remote_code=True, gpu_memory_utilization=0.8)
    return llm, sampling_params, tokenizer


def get_vllm_completion(llm, prompts, sampling_params):
    outputs = llm.generate(prompts, sampling_params)
    responses = [output.outputs[0].text for output in outputs]
    return responses


def prepare_hf_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id,trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype="auto").to("cuda")
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token == None else tokenizer.pad_token
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id == None else tokenizer.pad_token_id
    model.config.pad_token_id = model.config.eos_token_id if model.config.pad_token_id == None else model.config.pad_token_id
    model.eval()
    return model, tokenizer


def get_hf_completion(model, tokenizer, prompts, max_new_tokens=20, batch_size=1):
    all_responses = []
    
    # sort the prompts based on length and batch them accordingly, after completion sort them back to original order
    length_sorted_idx = np.argsort([len(sen) for sen in prompts])[::-1]
    prompts_sorted = [prompts[idx] for idx in length_sorted_idx]
    for i in tqdm(range(0, len(prompts_sorted), batch_size)):
        batch_prompts = prompts_sorted[i:i+batch_size]

        if model.config.architectures[0] == 'MPTForCausalLM':
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=model.config.max_seq_len-max_new_tokens).to(model.device)
        else:
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            output_sequences = model.generate(inputs["input_ids"], max_new_tokens=max_new_tokens)

        responses = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
        outputs = [response.replace(p, '') for response, p in zip(responses, batch_prompts)]
        all_responses.extend(outputs)

    all_responses = [all_responses[idx] for idx in np.argsort(length_sorted_idx)]
    
    return all_responses


def prompt_to_messages(role, prompt, messages=[]):
    message = {"role": role, "content": prompt}
    messages.append(message)
    return messages


def messages_to_prompt(messages, tokenizer):
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = ' '.join([f"{m['content']}" for m in messages])
    return prompt


def prompt_to_chatprompt(prompt, tokenizer):
    messages = prompt_to_messages('user', prompt, messages=[])
    prompt = messages_to_prompt(messages, tokenizer)
    return prompt


def before_retry_fn(retry_state):
    if retry_state.attempt_number > 1:
        print(f"Retrying API call. Attempt #{retry_state.attempt_number}, f{retry_state}")


def parallel_query_chatgpt_model(args):
    return query_chatgpt_model(*args)


def parallel_query_chatgpt_model_azure(args):
    return query_chatgpt_model_azure(*args)


def parallel_query_claude_model(args):
    return query_claude_model(*args)


def parallel_query_gemini_model(args):
    return query_gemini_model(*args)


def parallel_query_together_model(args):
    return query_together_model(*args)


def parallel_query_bloom_model(args):
    return query_bloom_model(*args)


@retry(wait=wait_fixed(5)+ wait_random(0, 5),stop=stop_after_attempt(6), before=before_retry_fn)
def query_chatgpt_model(api_key, prompt, model="gpt-3.5-turbo-0125", max_tokens=64, temperature=0):
    # openai.api_key = api_key
    client = OpenAI(api_key=api_key)
    try:
        completions = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            n=1,
            stop=None,
            temperature=temperature,
        )
        output = completions.choices[0].message.content.strip()

    except Exception as e:
        print(e)
        output = '[CAUTION] Problematic Output' # still save an output

    return output


@retry(wait=wait_fixed(5)+ wait_random(0, 5),stop=stop_after_attempt(3), before=before_retry_fn)
def query_chatgpt_model_azure(api_key, prompt, model="gpt4-1106", max_tokens=64, temperature=0):
    client = AzureOpenAI(
        # replace with your own azure endpoint
        azure_endpoint = "https://xxx.openai.azure.com/",
        api_key=api_key,
        api_version="2024-02-15-preview"
    )
    try:
        completions = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            n=1,
            stop=None,
            temperature=temperature,
        )
        output = completions.choices[0].message.content.strip()

    except Exception as e:
        print(e)
        output = '[CAUTION] Problematic Output' # still save an output
        
    return output


@retry(wait=wait_fixed(5)+ wait_random(0, 5),stop=stop_after_attempt(3), before=before_retry_fn)
def query_gemini_model(api_key, prompt, model="gemini-1.0-pro", max_tokens=64, temperature=0):
    genai.configure(api_key=api_key)
    generation_config = {
        "temperature": temperature,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": max_tokens,
    }
    safety_level = "BLOCK_NONE"
    # safety_level = "BLOCK_MEDIUM_AND_ABOVE"
    safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": safety_level
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": safety_level
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": safety_level
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": safety_level
    },
    ]

    model = genai.GenerativeModel(model_name=model,
                                  generation_config=generation_config,
                                  safety_settings=safety_settings,)
    # convo = model.start_chat(history=[
    # ])
    # convo.send_message(prompt)
    # return convo.last.text
    

    response = model.generate_content(prompt)
    try:
        # output = convo.last.text
        output = response.text
    except Exception as e:
        # print(f"Error: {e}")
        output = "No output from the model due to safety settings."
    return output


@retry(wait=wait_fixed(5)+ wait_random(0, 5),stop=stop_after_attempt(3), before=before_retry_fn)
def query_claude_model(api_key, prompt, model="claude-instant-1.2", max_tokens=64, temperature=0):
    # models: claude-3-opus-20240229, claude-3-sonnet-20240229, claude-2.0, claude-2.1, claude-instant-1.2
    client = anthropic.Anthropic(api_key=api_key, max_retries=3)

    try:
        completions = client.messages.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        output = completions.content[0].text.strip()

    except anthropic.APIConnectionError as e:
        print("The server could not be reached")
        print(e.__cause__)  # an underlying Exception, likely raised within httpx.
    except anthropic.RateLimitError as e:
        print("A 429 status code was received; we should back off a bit.")
    except anthropic.APIStatusError as e:
        print("Another non-200-range status code was received")
        print(e.status_code)
        print(e.response)

    return output


@retry(wait=wait_fixed(5)+ wait_random(0, 5),stop=stop_after_attempt(5), before=before_retry_fn)
def query_together_model(api_key, prompt, model="Qwen/Qwen1.5-72B-Chat", max_tokens=64, temperature=0):
    endpoint = 'https://api.together.xyz/v1/chat/completions'
    headers = {"Authorization": f"Bearer {api_key}",}
    try:
        res = requests.post(endpoint, json={
            "model": model,
            "max_tokens": max_tokens,
            "request_type": "language-model-inference",
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}]
        }, headers=headers)
        output = res.json()['choices'][0]['message']['content']
    except Exception as e:
        res_json = res.json()
        if 'error' in res_json and res_json['error']['message'].startswith('Input validation error: `inputs`'):
            output = "the question is too long"
        else:
            raise e
    return output


@retry(wait=wait_fixed(10), stop=stop_after_attempt(6), before=before_retry_fn)
def query_bloom_model(api_key, prompt, model_id="bigscience/bloom"):
    model_url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "inputs": f"{prompt}",
        "temperature": 0.0
    }
    try:
        response = requests.post(model_url, headers=headers, json=payload)
        pred = response.json()[0]['generated_text'].strip()
    except Exception as e:
        response_json = response.json()
        # if the error is due to max context length, save such an error
        if "error" in response_json and response_json['error'].startswith('Input validation error: `inputs`'):
            pred = "the question is too long"
        else:
            raise e
    return pred


# test chatgpt 
def test_chatgpt():
    api_key = os.environ["OPENAI_API_KEY"]
    output = query_chatgpt_model(api_key, "What is the capital of France?")
    print(f"ChatGPT: {output}")


def test_gemini():
    api_key = os.environ["GEMINI_API_KEY"]
    model="gemini-1.0-pro"
    prompt = "The following is a multiple-choice question. Please only give the correct option, without any other details or explanations.\n\nWhich statement is an opinion about World War II?\nA. Dropping atomic bombs was not necessary to end the war.\nB. Italy and Germany were members of the Axis powers.\nC. Many families suffered the loss of loved ones during the war.\nD. The economies of many countries were damaged by the war.\nAnswer:"
    # prompt = "What is the capital of China?"
    output = query_gemini_model(api_key, prompt, model=model)
    print(f"Gemini: {output}")


def test_claude():
    api_key = os.environ["ANTHROPIC_API_KEY"]
    output = query_claude_model(api_key, "What is the capital of France?")
    print(f"Claude: {output}")


def test_together():
    api_key = os.environ["TOGETHER_API_KEY"]
    output = query_together_model(api_key, "What is the capital of France?")
    print(f"Together: {output}")


def test_bloom():
    api_key = os.environ["HF_API_KEY"]
    output = query_bloom_model(api_key, "What is the capital of France?")
    print(f"Bloom: {output}")


def test_vllm():
    llm, sampling_params, tokenizer = prepare_vllm()
    prompt = "What is the capital of France?"
    messages = prompt_to_messages('user', prompt, messages=[])
    prompt = messages_to_prompt(messages, tokenizer)
    responses = get_vllm_completion(llm, prompt, sampling_params)
    print(f"VLLM: {responses}")


def test_hf():
    model_id = "aisingapore/sea-lion-7b"
    tokenizer, model = prepare_hf_model(model_id)
    sentences = [
         "Hello, my dog is a little",
          "Today, I",
          "Apple is",
          "How to make a cake?"
        ]
    outputs = get_hf_completion(model, tokenizer, sentences, max_new_tokens=20)
    print(f"HF: {outputs}")


def test_azure():
    api_key = os.environ["AZURE_OPENAI_KEY"]
    output = query_chatgpt_model_azure(api_key, "What is the capital of France?")
    print(f"ChatGPT: {output}")


if __name__ == "__main__":
    # test_chatgpt()
    # test_together()
    # test_bloom()
    test_vllm()
    # test_hf()
    # test_claude()
    # test_gemini()
    # test_azure()
