# OverPrompt: Enhancing ChatGPT through Efficient In-Context Learning
Jiazheng Li*, Runcong Zhao*, Yongxin Yang, Yulan He and Lin Gui

## Abstract
The remarkable performance of pre-trained large language models has revolutionised various natural language processing applications. Due to huge parameter sizes and extensive running costs, companies or organisations tend to transfer the models to the target task by zero-shot prompting techniques. However, the prohibitive costs of tokens and time have hindered their adoption in applications. We propose OverPrompt, leveraging the in-context learning capability of LLMs to handle multiple task inputs, thereby reducing token and time costs. This approach could potentially improve task performance during API queries due to better conditional distribution mapping. Evaluated across diverse classification datasets, our experiments show that OverPrompt can achieve cost-efficient zero-shot classification without causing significant detriment to task performance, and in some cases, even improving it. An ablation study conducted on various LLMs, along with an investigation into the robustness of our prompting strategy to different input ordering, offers valuable insights into the broader applicability of our method across diverse tasks. These findings also suggest a more seamless integration of our method with LLMs through an API.

## Getting Started

### Requirements

```bash
pip install -r requirements.txt
```

### Basic Usage
`python run.py` with the following commands:

`-d, --dataset (str)`: Required. Dataset name.   
`-p, --prompt (int)`: Zero-shot (0) or OverPrompt (>0). Default is 0.   
`-m, --model (str)`: Large Language Model name (e.g., "gpt-3.5-turbo"). Default is "gpt-3.5-turbo".   
`-e, --evaluation (bool)`: Perform evaluation. Default is False.   
`-o, --output (str)`: Pre-defined output path, for reruns. Optional.   
`-q, --portion (float)`: Portion of dataset to use. Default is 1.0.   
`-r, --random_seed (int)`: Set a random seed. Optional.   
`-a, --analysis (bool)`: Enable efficiency analysis. Default is False.   
`-t, --template (str)`: Prompt template ("plain" or "json"). Default is 'plain'.   
`-c, --cutoff (int)`: Cut off too long dataset. Default is 1000.   
`-l, --log (bool)`: Enable logging. Default is True.   
`-k, --key (str)`: OpenAI API key. Optional.   
`--permutation (str)`: Enable permutation. Optional.
