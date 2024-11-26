import re
import pandas as pd

def compute_acc_score(preds, model):
    option_indexs = ['A', 'B', 'C', 'D', 'E']
    """ Compute acc scores for a particular json file """
    match, total = 0, 0
    errors, illformats, checks = [], [], []
    for question in preds:
        total += 1
        answer = option_indexs[question['answer']]
        pred = question['pred'].strip()
        
        # prediction of bloom also include the input prompt
        if model == 'bloom':
            pred = pred.replace(question['prompt'], "").strip()
            question['bloom_pred_strip'] = pred

        # write regex to extrat the first lettr ['A', 'B', 'C', 'D'] from the pred
        pred = re.findall(r'[A-E]', pred)
        if len(pred) > 0:
            pred = pred[0]
        else:
            pred = ''
            illformats.append(question)
        
        question['pred_extract'] = pred

        if answer == pred:
            match += 1
            question['match'] = 1
        else:
            question['match'] = 0
            errors.append(question)
        checks.append(question)

    return (total, match), errors, illformats, checks


def load_df(output_dir):
    df = pd.read_csv(f'{output_dir}/eval_exam.csv',header=None)
    df.columns = ['data_name', 'model', 'generation_mode', 'subject', 'method', 'lang', 'num_total','num_correct', 'acc']
    df = df.drop_duplicates(subset=['data_name', 'model', 'generation_mode', 'subject', 'method', 'lang', 'num_total'],keep='last')
    df = df.sort_values(by=['data_name', 'model', 'subject', 'method', 'lang','generation_mode',]).reset_index(drop=True)
    return df


def post_process(df, data_name, output_dir):
    df_new = df[df['data_name'] == data_name]
    # list_langs = ['english', 'chinese', 'indonesian','thai','vietnamese']
    list_langs = ['english', 'chinese', 'afrikaans', 'hindi', 'spanish', 'french', 'arabic', 'bengali', 'russian', 'portuguese', 'indonesian', 'urdu', 'german', 'japanese', 'swahili', 'marathi', 'telugu', 'tamil', 'turkish', 'korean', 'thai',  'vietnamese', 'javanese', 'italian', 'hausa', 'gujarati']
    langs = list(df_new['lang'].unique())
    langs = [lang for lang in list_langs if lang in langs]
    print("all langs: ", langs)
    df_new = df_new.groupby(['data_name','model','generation_mode','method','lang']).agg({'num_total': 'sum', 'num_correct': 'sum'}).reset_index()
    df_new['acc'] = df_new['num_correct'] / df_new['num_total']
    df_new = df_new.drop(columns=['num_total', 'num_correct'])
    df_new = df_new.pivot_table(index=['data_name','model','generation_mode','method'],columns='lang',values='acc')
    df_new = df_new.reset_index()
    df_new['avg'] = df_new[langs].mean(axis=1)
    df_new = df_new.sort_values(by=['model']).reset_index(drop=True)
    df_new['model'] = df_new['model'].apply(lambda x: '/'.join(x.split('/')[-2:]))
    df_new = df_new[['data_name','model','generation_mode','method'] + langs + ['avg']]
    df_new = df_new.drop(columns=["data_name"])
    df_new.to_csv(f'{output_dir}/eval_{data_name}_pivot.csv', index=False)
    # df_new = df_new.drop(columns=["generation_mode","data_name"])
    return df_new