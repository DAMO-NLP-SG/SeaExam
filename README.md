# SeaExam: Benchmarking LLMs for Southeast Aisa languages with Human Exam Questions

<p align="center">
<a href="https://huggingface.co/datasets/SeaLLMs/SeaExam" target="_blank" rel="noopener"> 🤗 Dataset</a>
&nbsp;&nbsp;
<a href="https://huggingface.co/spaces/SeaLLMs/LLM_Leaderboard_for_SEA" target="_blank" rel="noopener"> 🤗 Leaderboard</a>
&nbsp;&nbsp;
<a href="https://aclanthology.org/2025.findings-naacl.341/" target="_blank" rel="noopener"> Paper</a>
</p>

This repo contains code for SeaExam, a toolkit for evaluating large language models (LLM) for Southeast Asian (SEA) languages including Chinese, English, Indonesian, Thai, and Vietnamese.

The evaluation dataset consists of [M3Exam](https://github.com/DAMO-NLP-SG/M3Exam) and translated [MMLU](https://github.com/hendrycks/test) datsets. For more information, refer to the huggingface [dataset page](https://huggingface.co/datasets/SeaLLMs/SeaExam).

Please also check SeaBench dataset [here](https://github.com/DAMO-NLP-SG/SeaBench) for more evaluation tasks on SEA languages.

## Setup enironment
```
git clone https://github.com/DAMO-NLP-SG/SeaExam.git
cd SeaExam
conda create -n SeaExam python=3.9
conda activate SeaExam
pip install -r requirement.txt
```

## Evaluate models

To quickly evaluate your model on SeaExam, just run 
```
python scripts/main.py --model $model_name_or_path
```

For example: 
```
python scripts/main.py --model SeaLLMs/SeaLLMs-v3-7B-Chat
```
Or 
```
bash quick_run.sh
```

## Under the hood

Our goal is to ensure a fair and consistent comparison across different LLMs while mitigating the risk of data contamination. 

To ensure a fair comparison and reduce LLMs' dependence on specific prompt templates, we have designed several templates. If `dynamic_template` is set as True (which is the default setting), a template will be randomly selected for each question. Additionally, users have the option to change the seed value to generate a different set of questions for evaluation purposes.

## Citation
```bibtex
@inproceedings{liu-etal-2025-seaexam,
    title = "{S}ea{E}xam and {S}ea{B}ench: Benchmarking {LLM}s with Local Multilingual Questions in {S}outheast {A}sia",
    author = "Liu, Chaoqun  and
      Zhang, Wenxuan  and
      Ying, Jiahao  and
      Aljunied, Mahani  and
      Luu, Anh Tuan  and
      Bing, Lidong",
    editor = "Chiruzzo, Luis  and
      Ritter, Alan  and
      Wang, Lu",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2025",
    month = apr,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-naacl.341/",
    pages = "6119--6136",
    ISBN = "979-8-89176-195-7"
}
```
