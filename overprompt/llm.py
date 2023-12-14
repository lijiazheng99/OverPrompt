import openai
import timeout_decorator
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from transformers import AutoTokenizer, LlamaForCausalLM, AutoModelForCausalLM, T5Tokenizer, T5ForConditionalGeneration
import torch

def set_openai_key(key):
    """Sets OpenAI key."""
    openai.api_key = key


class LLM:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.model_name = model_name
        self.local = False
        if model_name not in ["gpt-3.5-turbo", "gpt-3.5", "gpt-3"]:
            self.local = True
            if not torch.cuda.is_available():
                raise ValueError("CUDA not avaliable")
            else:
                if "flan-t5" in model_name:
                    self.tokenizer = T5Tokenizer.from_pretrained(model_name)
                    self.model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto")
                elif "llama" in model_name:
                    self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-hf")
                    self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
                else:
                    raise ValueError(f"Local model not found {model_name}")
            print("Local models successfully initialized")
    
    def gen_response_local(self, content:str=""):
        input_ids = self.tokenizer(content, return_tensors="pt").input_ids.to("cuda")
        outputs = self.model.generate(input_ids, max_new_tokens = 300)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].replace(content,'')

    def gen_response(self, content:str=""):
        if content == "":
            raise ValueError("Content cannot be empty")
        
        if self.local:
            return self.gen_response_local(content)
        else:
            return self.gen_response_openai(content, self.model_name)

    @retry(wait=wait_random_exponential(min=30, max=90), stop=stop_after_attempt(3))
    @timeout_decorator.timeout(30)
    def gen_response_openai(self, content:str=""):
        if content == "":
            raise ValueError("Content cannot be empty")

        completion = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
            {"role": "user", "content": content},
            ],
            temperature=0.0,
        )
        return completion.choices[0].message["content"]


@retry(wait=wait_random_exponential(min=30, max=90), stop=stop_after_attempt(3))
@timeout_decorator.timeout(30)
def gen_response(content:str="", model:str="gpt-3.5-turbo"):
    if content == "":
        raise ValueError("Content cannot be empty")
    
    completion = openai.ChatCompletion.create(
        model=model,
        messages=[
        {"role": "user", "content": content},
        ],
        temperature=0.0,
    )
    return completion.choices[0].message["content"]