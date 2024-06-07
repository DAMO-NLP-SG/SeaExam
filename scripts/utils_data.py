from utils_model import *
from collections import defaultdict
import random
import json
from datasets import load_dataset

all_langs = ['english', 'chinese', 'thai', 'vietnamese', 'indonesian']

dic_subjects = {
               'm3exam': ['language', 'math', 'social-science', 'natural-science'],
               'mmlu': ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']
               }

def shuffle_options(options, answer):
    new_options = options.copy()
    random.shuffle(new_options)
    try:
        new_answer = new_options.index(options[answer])
    except:
        print(f"Warning: answer {answer} not found in {new_options}.")
        raise ValueError
    return new_options, new_answer


def generate_one_example(question, fill_answer=False, add_space=True):
    option_indexs = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    choices = '\n'.join([f"{option_indexs[i]}. {question['choices'][i]}" for i in range(len(question['choices']))])
    prompt =  question['question'] + '\n' + choices + f'\nAnswer:'
    
    if fill_answer:
        if add_space:
            prompt += ' ' + option_indexs[question['answer']] # template_mmlu
        else:
            prompt += option_indexs[question['answer']] # template_m3exam
    
    return prompt


def generate_dev_examples(dev_questions, add_space=True):
    # save the dev examples into a dict, according to their subject categories
    dev_example_dict = defaultdict(list)
    for q in dev_questions:
        if 'subject' in q['metadata']:
            cate = q['metadata']['subject']
        else:
            cate = 'all'
        dev_string = generate_one_example(q, fill_answer=True, add_space=add_space)
        dev_example_dict[cate].append(dev_string)
    
    return dev_example_dict


def generate_prompt(setting, test_question, dev_question, dynamic_template=False, add_space=True):
    if 'subject' in test_question['metadata']:
        subject = test_question['metadata']['subject']
        list_description = [f"The following are multiple choice questions (with answers) about {subject.replace('_',' ')}.",
                            f"The following is a multiple choice question about {subject.replace('_',' ')}.", 
                            f"The following are multiple choice questions (with answers).",
                            f"The following is a multiple choice question.",""]
        description = random.choice(list_description) if dynamic_template else list_description[0]
    else:
        subject = 'all'
        list_description = [f"The following are multiple choice questions (with answers).",
                            f"The following is a multiple choice question.",""]
        description = random.choice(list_description) if dynamic_template else list_description[0]

    if setting == 'zero-shot':
        description = "The following is a multiple-choice question. Please only give the correct option, without any other details or explanations."
        prompt = generate_one_example(test_question, add_space=add_space)
    elif setting == 'few-shot':
        dev_questions_list = dev_question[subject][:3]
        if len(dev_questions_list) < 3:
            print(f"Warning: only {len(dev_questions_list)} dev questions found for {subject}.")
        prompt = '\n\n'.join(dev_questions_list) + '\n\n' + generate_one_example(test_question, add_space=add_space)
    else:
        raise NotImplementedError
    prompt = description + '\n\n' + prompt if len(description) > 0 else prompt

    return prompt


def generate_questions(data_name, lang, setting, dynamic_template=False, add_space=True, random_seed=1, repo_path = "SeaLLMs/SeaExam"):
    # dynamic_template: If dynamic_template is True, then the prompt description will be generated randomly and the choices will be shuffled.
    # add_space: If add_space is True, then the answer will be filled with a space, otherwise not. It only works when dynamic_template is False.

    random.seed(random_seed)

    dataset = load_dataset(repo_path, data_name + '-' + lang)
    
    if len(dataset['test']) > 0:
        test_questions = dataset['test'].to_list()
        
        for question in test_questions:
            if dynamic_template:
                add_space = random.choice([True, False])
            
            # if conduct few-shot settings
            if setting == 'few-shot':
                if len(dataset['dev']) > 0:
                    dev_questions = dataset['dev'].to_list()
                    if dynamic_template:
                        for question_dev in dev_questions:
                            question_dev['choices'], question_dev['answer'] = shuffle_options(question_dev['choices'], question_dev['answer'])
                    dev_examples = generate_dev_examples(dev_questions, add_space=add_space)
                else:
                    raise FileNotFoundError
            else:
                dev_examples = {} 

            if dynamic_template:
                question['choices'], question['answer'] = shuffle_options(question['choices'], question['answer'])
            prompt = generate_prompt(setting, question, dev_examples, dynamic_template, add_space)
            question['prompt'] = prompt 
    else: 
        # raise FileNotFoundError
        test_questions = []
    return test_questions


def _test_generate_questions():
    data_name = 'mmlu'

    lang = 'indonesian'
    setting = 'few-shot'
    dynamic_template = True
    add_space = True
    # choose a random seed
    seed = random.randint(0, 1000)
    
    test_questions = generate_questions(data_name, lang, setting, dynamic_template, add_space, random_seed=seed)
    print(len(test_questions))
    print(test_questions[0]['prompt'])


if __name__ == "__main__":
    _test_generate_questions()

