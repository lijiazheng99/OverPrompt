import pandas as pd
import json
import jsonlines
import os
from sklearn.metrics import matthews_corrcoef,accuracy_score, f1_score
import numpy as np

def data_loader(dataset_name:str="", dataset_path:str="./dataset", prompt_mode:int=1, random_seed:int=None, prompt_template:str=''):
    if dataset_name == "":
        raise ValueError("Please specify a dataset.")
    elif dataset_name == "sst2":
         return SST2(dataset_path=dataset_path, random_seed=random_seed, prompt_mode=prompt_mode, prompt_template=prompt_template)
    elif dataset_name == "multi_nli":
        return MultiNLI(dataset_path=dataset_path, random_seed=random_seed, prompt_mode=prompt_mode, prompt_template=prompt_template)
    elif dataset_name == "fever":
        return FEVER(dataset_path=dataset_path, random_seed=random_seed, prompt_mode=prompt_mode, prompt_template=prompt_template)
    elif dataset_name == "vitaminc":
        return VITAMINC(dataset_path=dataset_path, random_seed=random_seed, prompt_mode=prompt_mode, prompt_template=prompt_template)
    elif dataset_name == "hover":
        return HOVER(dataset_path=dataset_path, random_seed=random_seed, prompt_mode=prompt_mode, prompt_template=prompt_template)
    elif dataset_name == "rte":
        pass
    elif dataset_name == "mpqa":
        return MPQA(dataset_path=dataset_path, random_seed=random_seed, prompt_mode=prompt_mode, prompt_template=prompt_template)
    else:
        raise ValueError(f"Dataset name not implemented: {dataset_name}")

class DataLoader:
    data = []  # List of json objects

    def __init__(self, random_seed:int=None, prompt_mode:int=0, prompt_template:str='') -> None:
        self.random_seed = random_seed
        self.prompt_mode = prompt_mode
        self.prompt_template = prompt_template

        # Prompt mode
        if int(self.prompt_mode) == 1:
            self.single = True
        elif int(self.prompt_mode) > 1:
            self.single = False

        self.file_path = f"" # File path of the dataset
        self.labels = [] # List of labels in string
        # self.data = []  # List of json objects

        # Prompt templates
        self.single_begginning = ""
        self.single_ending = ""
        self.multiple_beginning = ""
        self.multiple_ending = ""
    
    def random_portion_processor(self, data, portion, random_seed):
        dataframe = pd.DataFrame(data)
        if random_seed is not None:
            dataframe = dataframe.sample(frac=portion, random_state=random_seed).reset_index(drop=True)
        elif portion < 1.0:
            dataframe = dataframe.sample(frac=portion).reset_index(drop=True)
        if len(dataframe) == 0:
            raise ValueError(f"Portion too small: {portion}")
        data = dataframe.to_json(orient='records', lines=True).splitlines()
        data = [json.loads(each) for each in data]
        return data
    
    def jsonl_loader(self, file_name: str) -> list:
        if os.path.exists(file_name):
            data = []
            with open(file_name, 'r') as file:
                for each in file:
                    json_line = json.loads(each)
                    data.append(json_line)
            return data
        else:
            raise ValueError(f"File not found: {file_name}")
    
    def hf_datasets_loader(self, dataset_name:str, split:str="validation") -> list:
        from datasets import load_dataset
        if type(dataset_name) == list:
            dataset = load_dataset(dataset_name[0], dataset_name[1])[split]
        else:
            dataset = load_dataset(dataset_name)[split]
        data = []
        for item in dataset:
            json_line = json.dumps(item, ensure_ascii=False)
            json_line = json.loads(json_line)
            data.append(json_line)
        return data
    
    def check_data_type(self, data):
        if type(data) != list:
            raise ValueError(f"Invalid dataset format: {type(data)}")
        if type(data[0]) != dict:
            raise ValueError(f"Invalid dataset row format: {type(data[0])}")
    
    def single_sentence_prompt_generator(self, sentences):
        prompt = ""
        if self.single:
            prompt += self.single_begginning 
            prompt += f"\"{sentences}\"\n"
            prompt += self.single_ending
            return prompt
        else:
            prompt += self.multiple_beginning
            for idx, sentence in enumerate(zip(sentences)):
                prompt += f"{idx}: \"{str(sentence[0])}\"\n"
            prompt += self.multiple_ending
            return prompt
    
    def duo_sentence_prompt_generator(self, sentences1, sentences2):
        prompt = ""
        if self.single:
            prompt += self.single_begginning 
            prompt += f"[sentence1]: \"{sentences1}\"\n[sentence2]: \"{sentences2}\"\n"
            prompt += self.single_ending
            return prompt
        else:
            prompt += self.multiple_beginning
            for idx, (sentence1, sentence2) in enumerate(zip(sentences1, sentences2)):
                prompt += f"Pair {idx}: [sentence1]: \"{sentence1}\" [sentence2]: \"{sentence2}\"\n"
            prompt += self.multiple_ending
            return prompt
    
    def save_generated(self, file_name:str, data:list):
        if data is None:
            data = self.data
        if os.path.exists(file_name):
            raise ValueError(f"File already exists: {file_name}")
        else:
            with jsonlines.open(file_name, 'w') as writer:
                writer.write_all(data)
    
    def permutation_evaluation(self, n):
        batched = [self.data[i:i+n] for i in range(0, len(self.data), n)]
        all_accuracy = []
        for batch in batched:
            predicted_labels = []
            true_labels = []
            for each in batch:
                predicted_labels.append(each[model_name])
                true_labels.append(each[self.colum_info['target']])
            all_accuracy += [accuracy_score(predicted_labels, true_labels)]
        print(np.mean(all_accuracy),max(all_accuracy), min(all_accuracy))

class MultiNLI(DataLoader):
    def __init__(self, dataset_path:str="./dataset", random_seed:int=42, prompt_mode:int=0, prompt_template:str='') -> None:
        super().__init__(random_seed, prompt_mode, prompt_template)

        # Dataset specific descriptions
        self.colum_info = {"target":"gold_label", "input":["sentence1", "sentence2"]}
        if dataset_path is None:
            self.file_name = f"./dataset/multinli_1.0/multinli_1.0_dev_matched.jsonl"
        else:
            self.file_name = f"{dataset_path}/output.jsonl"
        self.labels = ['contradiction', 'entailment', 'neutral']
        # Single pair sentences
        self.single_begginning = "Please read through this pair of sentences \n"
        self.single_ending = "and determine whether the setences are \"entailment\", \"neutral\" or \"contradiction\" to each other. Give me the label only: "
        # Multiple pairs of sentences
        self.multiple_beginning = "Please read through these pairs of sentences\n"
        if self.prompt_template == 'plain':
            self.multiple_ending = "and determine whether these sentences are \"entailment\", \"neutral\" or \"contradiction\" to each other. Give me the labels only: "
        elif self.prompt_template == 'json':
            self.multiple_ending = "and determine whether these sentences are \"entailment\", \"neutral\" or \"contradiction\" to each other. Return in JSON format, such as: {\"1\": \"c_1\", \"2\":\"c_2\"}: "
    
    def load_dataset(self, portion:float=1.0, cutoff:int=1000):
        data = self.jsonl_loader(self.file_name)
        # housekeeping
        data = [each for each in data if each[self.colum_info['target']] in self.labels]
        data = self.random_portion_processor(data, portion, self.random_seed)
        data = data[:cutoff]
        self.data = data
        self.check_data_type(data)
        return data
    
    def prompt_generator(self, sentence1, sentence2):
        if type(sentence1) == str and type(sentence2) == str and self.single:
            pass
        elif type(sentence1) == list and type(sentence2) == list and not self.single:
            pass
            if len(sentence1) != len(sentence2):
                raise ValueError(f"Invalid input lengths: sentence1 got {len(sentence1)} and sentence2 got {len(sentence2)}")
        else:
            raise ValueError(f"Invalid input types: sentence1 got {type(sentence1)} and sentence2 got {type(sentence2)}")
        return self.duo_sentence_prompt_generator(sentence1, sentence2)

    def case_conventor(self, text):
        if text not in self.labels and len(text) > 0:
            if 'entailment'in text or 'Entailment' in text:
                text = 'entailment'
            elif 'neutral' in text or 'Neutral' in text:
                text = 'neutral'
            elif 'contradiction' in text or 'Contradiction' in text:
                text = 'contradiction'
        return text
    
    def evaluation(self, model_name:str="gpt-3.5-turbo"):
        predicted_labels = []
        true_labels = []
        for each in self.data:
            predicted_labels.append(each[model_name])
            true_labels.append(each[self.colum_info['target']])
        return_dict = {}
        return_dict['Accuracy'] = accuracy_score(predicted_labels, true_labels)
        return return_dict

class FEVER(DataLoader):
    def __init__(self, dataset_path:str="./dataset", random_seed:int=42, prompt_mode:int=0, prompt_template:str='') -> None:
        super().__init__(random_seed, prompt_mode, prompt_template)

        # Dataset specific descriptions
        self.colum_info = {"target":"classification", "input":["query", "evidences"]}
        if dataset_path is None:
            self.file_name = f"./dataset/fever/val.jsonl"
        else:
            self.file_name = f"{dataset_path}/output.jsonl"
        self.labels = ["REFUTES", "SUPPORTS"]
        # Single pair sentences
        self.single_begginning = "Please read through this pair of claim and evidence\n"
        self.single_ending = "and determine whether the evidence \"support\" or \"refute\" the claim. \nGive me the label only: "
        # Multiple pairs of sentences
        self.multiple_beginning = "Please read through these pairs of claim and evidence\n"
        if self.prompt_template == 'plain':
            self.multiple_ending = "and determine whether the evidence \"support\" or \"refute\" the claim. \nGive me the labels only: "
        elif self.prompt_template == 'json':
            self.multiple_ending = "and determine whether the evidence \"support\" or \"refute\" the claim. \nReturn in JSON format, such as: {\"1\": \"c_1\", \"2\":\"c_2\"}: "
    
    def load_dataset(self, portion:float=1.0, cutoff:int=1000):
        data = self.jsonl_loader(self.file_name)
        # housekeeping
        data = [each for each in data if each[self.colum_info['target']] in self.labels]
        data = self.random_portion_processor(data, portion, self.random_seed)
        data = data[:cutoff]
        self.data = data
        self.check_data_type(data)
        return data
    
    def duo_sentence_prompt_generator(self, sentences1, sentences2):
        prompt = ""
        if self.single:
            prompt += self.single_begginning 
            prompt += f"[Claim]: \"{sentences1}\"\n[Evidence]: \"{sentences2[0][0]['text']}\"\n"
            prompt += self.single_ending
            return prompt
        else:
            prompt += self.multiple_beginning
            for idx, (sentence1, sentence2) in enumerate(zip(sentences1, sentences2)):
                prompt += f"Pair {idx}: [Claim]: \"{sentence1}\" [Evidence]: \"{sentence2[0][0]['text']}\"\n"
            prompt += self.multiple_ending
            return prompt
    
    def prompt_generator(self, sentence1, sentence2):
        if type(sentence1) == str and type(sentence2) == str and self.single:
            pass
        elif type(sentence1) == list and type(sentence2) == list and not self.single:
            pass
            if len(sentence1) != len(sentence2):
                raise ValueError(f"Invalid input lengths: sentence1 got {len(sentence1)} and sentence2 got {len(sentence2)}")
        else:
            raise ValueError(f"Invalid input types: sentence1 got {type(sentence1)} and sentence2 got {type(sentence2)}")
        return self.duo_sentence_prompt_generator(sentence1, sentence2)

    def case_conventor(self, text):
        if text not in ['SUPPORTS','REFUTES', 'NOT ENOUGH INFO'] and len(text) > 0:
            if 'refute' in text or 'Refute'in text or 'REFUTE' in text:
                text = 'REFUTES'
            elif 'support' in text or 'Support' in text or 'SUPPORT' in text:
                text = 'SUPPORTS'
            elif 'Not enough info' in text or 'Not Enough Info' in text or 'NOT ENOUGH INFO' in text or 'not enough info' in text:
                text = 'NOT ENOUGH INFO'
            else:
                print('not sure what output: ', text)
        return text
    
    def evaluation(self, model_name:str="gpt-3.5-turbo"):
        predicted_labels = []
        true_labels = []
        for each in self.data:
            predicted_labels.append(each[model_name])
            true_labels.append(each[self.colum_info['target']])
        return_dict = {}
        return_dict['Accuracy'] = accuracy_score(predicted_labels, true_labels)
        return return_dict

class SST2(DataLoader):
    def __init__(self, dataset_path:str="./dataset", random_seed:int=42, prompt_mode:int=0, prompt_template:str='') -> None:
        super().__init__(random_seed, prompt_mode, prompt_template)

        # Dataset specific descriptions
        self.colum_info = {"target":"label", "input":["sentence"]}
        if dataset_path is None:
            self.file_name = f"./dataset"
        else:
            self.file_name = f"{dataset_path}/output.jsonl"
        self.labels = ["positive", "negative"]
        # Single pair sentences
        self.single_begginning = "Please read through this sentence:\n"
        self.single_ending = "and determine the sentiment of the sentence is \"positive\" or \"negative\". Give me the label only: "
        # Multiple pairs of sentences
        self.multiple_beginning = "Please read through these sentences:\n"
        if self.prompt_template == 'plain':
            self.multiple_ending = "and determine the sentiment of sentences are \"positive\" or \"negative\". Give me the labels only: "
        elif self.prompt_template == 'json':
            self.multiple_ending = "and determine the sentiment of sentences are \"positive\" or \"negative\". Return in JSON format, such as: {\"1\": \"c_1\", \"2\":\"c_2\"}: "
    
    def load_dataset(self, portion:float=1.0, cutoff:int=1000):
        data = self.hf_datasets_loader("sst2", "validation")
        # housekeeping
        for each in data:
            if each[self.colum_info['target']] == 0:
                each[self.colum_info['target']] = 'negative'
            elif each[self.colum_info['target']] == 1:
                each[self.colum_info['target']] = 'positive'
            else:
                if each[self.colum_info['target']] not in self.labels:
                    raise ValueError(f"Invalid label: {each[self.colum_info['target']]}")         
        data = self.random_portion_processor(data, portion, self.random_seed)
        data = data[:cutoff]
        self.data = data
        self.check_data_type(data)
        return data
    
    def prompt_generator(self, sentence1):
        if type(sentence1) == str and self.single:
            pass
        elif type(sentence1) == list and not self.single:
            pass
        else:
            raise ValueError(f"Invalid input types: sentence1 got {type(sentence1)}")
        return self.single_sentence_prompt_generator(sentence1)

    def case_conventor(self, text):
        if len(text) > 0:
            if 'positive'in text or 'Positive' in text:
                text = 'positive'
            elif 'negative' in text or 'Negative' in text:
                text = 'negative'
            else:
                print('not sure what output: ', text)
                return text
        return text
    
    def evaluation(self, model_name:str="gpt-3.5-turbo"):
        predicted_labels = []
        true_labels = []
        for each in self.data:
            predicted_labels.append(each[model_name])
            true_labels.append(each[self.colum_info['target']])
        return_dict = {}
        return_dict['Accuracy'] = accuracy_score(predicted_labels, true_labels)
        return_dict['macro_f1'] = f1_score(predicted_labels, true_labels, average='macro')
        return return_dict
    
class HOVER(DataLoader):
    def __init__(self, dataset_path:str="./dataset", random_seed:int=42, prompt_mode:int=0, prompt_template:str='') -> None:
        super().__init__(random_seed, prompt_mode, prompt_template)

        # Dataset specific descriptions
        self.colum_info = {"target":"label", "input":["claim"]}
        if dataset_path is None:
            self.file_name = f"./dataset"
        else:
            self.file_name = f"{dataset_path}/output.jsonl"
        self.labels = ["support", "refute"]
        # Single pair sentences
        self.single_begginning = "Categories: \"support\" or \"refute\"\n"
        self.single_ending = "Please use your background knowledge to decide which category they fall into.\nGive me the label only: "
        # Multiple pairs of sentences
        self.multiple_beginning = "Categories: \"support\" or \"refute\"\n"
        if self.prompt_template == 'plain':
            self.multiple_ending = "Please use your background knowledge to decide which categories they fall into.\nGive me the labels only: "
        elif self.prompt_template == 'json':
            self.multiple_ending = "Please use your background knowledge to decide which categories they fall into.\nReturn in JSON format, such as: {\"1\": \"c_1\", \"2\":\"c_2\"}: "
    
    def load_dataset(self, portion:float=1.0, cutoff:int=1000):
        data = self.hf_datasets_loader("hover", "validation")
        # housekeeping
        for each in data:
            if each[self.colum_info['target']] == 0:
                each[self.colum_info['target']] = 'support'
            elif each[self.colum_info['target']] == 1:
                each[self.colum_info['target']] = 'refute'
            else:
                if each[self.colum_info['target']] not in self.labels:
                    raise ValueError(f"Invalid label: {each[self.colum_info['target']]}")
        data = self.random_portion_processor(data, portion, self.random_seed)
        data = data[:cutoff]
        self.data = data
        self.check_data_type(data)
        return data
    
    def prompt_generator(self, sentence1):
        if type(sentence1) == str and self.single:
            pass
        elif type(sentence1) == list and not self.single:
            pass
        else:
            raise ValueError(f"Invalid input types: sentence1 got {type(sentence1)}")
        return self.single_sentence_prompt_generator(sentence1)

    def case_conventor(self, text):
        #text = text.lower()
        if text not in ['support','refute','not enough info'] and len(text) > 0:
            if 'support' in text:
                return 'support'
            elif 'refute' in text:
                return 'refute'
            elif 'not enough info' in text:
                return 'not enough info'
            else:
                print('not sure what output: ', text)
                return 'not enough info'
        return text
    
    def evaluation(self, model_name:str="gpt-3.5-turbo"):
        predicted_labels = []
        true_labels = []
        for each in self.data:
            predicted_labels.append(each[model_name])
            true_labels.append(each[self.colum_info['target']])
        return_dict = {}
        return_dict['Accuracy'] = accuracy_score(predicted_labels, true_labels)
        return return_dict

class VITAMINC(DataLoader):
    def __init__(self, dataset_path:str="./dataset", random_seed:int=42, prompt_mode:int=0, prompt_template:str='') -> None:
        super().__init__(random_seed, prompt_mode, prompt_template)

        # Dataset specific descriptions
        self.colum_info = {"target":"label", "input":["claim", "evidence"]}
        if dataset_path is None:
            self.file_name = f"./dataset"
        else:
            self.file_name = f"{dataset_path}/output.jsonl"
        self.labels = ["REFUTES", "SUPPORTS", "NOT ENOUGH INFO"]
        # Single pair sentences
        self.single_begginning = "Please read through this pair of claim and evidence\n"
        self.single_ending = "and determine whether the evidence \"support\", \"refute\" the claim, or \"not enough info\" to decide which category it fall into.\nGive me the label only: "
        # Multiple pairs of sentences
        self.multiple_beginning = "Please read through these pairs of claim and evidence\n"
        if self.prompt_template == 'plain':
            self.multiple_ending = "and determine whether the evidence \"support\", \"refute\" the claim, or \"not enough info\" to decide which category it fall into.\nGive me the labels only: "
        elif self.prompt_template == 'json':
            self.multiple_ending = "and determine whether the evidence \"support\", \"refute\" the claim, or \"not enough info\" to decide which category it fall into.\nReturn in JSON format, such as: {\"1\": \"c_1\", \"2\":\"c_2\"}: "
    
    def load_dataset(self, portion:float=1.0, cutoff:int=1000):
        data = self.hf_datasets_loader("tals/vitaminc", "validation")
        # housekeeping
        data = self.random_portion_processor(data, portion, self.random_seed)
        data = data[:cutoff]
        self.data = data
        self.check_data_type(data)
        return data
    
    def duo_sentence_prompt_generator(self, sentences1, sentences2):
        prompt = ""
        if self.single:
            prompt += self.single_begginning 
            prompt += f"[Claim]: \"{sentences1}\"\n[Evidence]: \"{sentences2}\"\n"
            prompt += self.single_ending
            return prompt
        else:
            prompt += self.multiple_beginning
            for idx, (sentence1, sentence2) in enumerate(zip(sentences1, sentences2)):
                prompt += f"Pair {idx}: [Claim]: \"{sentence1}\" [Evidence]: \"{sentence2}\"\n"
            prompt += self.multiple_ending
            return prompt

    def prompt_generator(self, sentence1, sentence2):
        if type(sentence1) == str and type(sentence2) == str and self.single:
            pass
        elif type(sentence1) == list and type(sentence2) == list and not self.single:
            pass
            if len(sentence1) != len(sentence2):
                raise ValueError(f"Invalid input lengths: sentence1 got {len(sentence1)} and sentence2 got {len(sentence2)}")
        else:
            raise ValueError(f"Invalid input types: sentence1 got {type(sentence1)} and sentence2 got {type(sentence2)}")
        return self.duo_sentence_prompt_generator(sentence1, sentence2)
    
    def case_conventor(self, text):
        if text not in ['SUPPORTS','REFUTES', 'NOT ENOUGH INFO'] and len(text) > 0:
            if 'refute' in text or 'Refute'in text or 'REFUTE' in text:
                text = 'REFUTES'
            elif 'support' in text or 'Support' in text or 'SUPPORT' in text:
                text = 'SUPPORTS'
            elif 'Not enough info' in text or 'Not Enough Info' in text or 'NOT ENOUGH INFO' in text or 'not enough info' in text:
                text = 'NOT ENOUGH INFO'
            else:
                print('not sure what output: ', text)
        return text
    
    def evaluation(self, model_name:str="gpt-3.5-turbo"):
        predicted_labels = []
        true_labels = []
        for each in self.data:
            predicted_labels.append(each[model_name])
            true_labels.append(each[self.colum_info['target']])
        return_dict = {}
        return_dict['Accuracy'] = accuracy_score(predicted_labels, true_labels)
        return return_dict

class MPQA(DataLoader):
    def __init__(self, dataset_path:str="./dataset", random_seed:int=42, prompt_mode:int=0, prompt_template:str='') -> None:
        super().__init__(random_seed, prompt_mode, prompt_template)

        # Dataset specific descriptions
        self.colum_info = {"target":"polarity", "input":["sentence"]}
        if dataset_path is None:
            self.file_name = f"./dataset/MPQA_data/"
        else:
            self.file_name = f"{dataset_path}/output.jsonl"
        self.labels = ['positively','negatively']
        # Single pair sentences
        self.single_begginning = "Please read through the given sentence\n"
        self.single_ending = "and determine whether the sentence \"positively\" or \"negatively\" affects objects. Give me the label only: "
        # Multiple pairs of sentences
        self.multiple_beginning = "Please read through the given sentence\n"
        if self.prompt_template == 'plain':
            self.multiple_ending = "and determine whether the sentence \"positively\" or \"negatively\" affects objects. Give me the labels only: "
        elif self.prompt_template == 'json':
            self.multiple_ending = "and for each sentence, determine whether the sentence \"positively\" or \"negatively\" affects objects. Return in JSON format, such as: {\"1\": \"c_1\", \"2\":\"c_2\"}: "
    
    def load_dataset(self, portion:float=1.0, cutoff:int=1000):
        data_all = []
        data_dir = os.path.join(self.file_name)
        for file in os.listdir(data_dir):
            with open(os.path.join(data_dir, file), "r") as f:
                data = f.readlines()
                f.close()

            for line in data:
                line_type = line.split('\t')[3]
                if line_type == "gfbf":
                    polarity = line.split('"')[3]
                    sentence = line.split('"')[-2]
                    #print(line_type, polarity, sentence, data_all[-3:])'
                    if polarity == 'goodfor':
                        polarity = 'positively'
                    elif polarity == 'badfor':
                        polarity = 'negatively'
                    if polarity not in self.labels:
                        print(polarity)
                        pass
                    else:
                        data_all.append({"polarity": polarity, "sentence": sentence})
        # housekeeping
        data = self.random_portion_processor(data_all, portion, self.random_seed)
        data = data[:cutoff]
        self.data = data
        self.check_data_type(data)
        return data
    
    def prompt_generator(self, sentence1, sentence2):
        if type(sentence1) == str and type(sentence2) == str and self.single:
            pass
        elif type(sentence1) == list and type(sentence2) == list and not self.single:
            pass
            if len(sentence1) != len(sentence2):
                raise ValueError(f"Invalid input lengths: sentence1 got {len(sentence1)} and sentence2 got {len(sentence2)}")
        else:
            raise ValueError(f"Invalid input types: sentence1 got {type(sentence1)} and sentence2 got {type(sentence2)}")
        return self.duo_sentence_prompt_generator(sentence1, sentence2)

    def case_conventor(self, text):
        text = text.lower()
        if text not in ['positively','negatively'] and len(text) > 0:
            if 'positive' in text:
                return 'positively'
            elif 'negative' in text:
                return 'negatively'
            else:
                print('not sure what output: ', text)
                return 'na'
        return text
    
    def evaluation(self, model_name:str="gpt-3.5-turbo"):
        predicted_labels = []
        true_labels = []
        for each in self.data:
            predicted_labels.append(each[model_name])
            true_labels.append(each[self.colum_info['target']])
        return_dict = {}
        return_dict['Accuracy'] = accuracy_score(predicted_labels, true_labels)
        return return_dict

class RTE(DataLoader):
    def __init__(self, dataset_path:str="./dataset", random_seed:int=42, prompt_mode:int=0, prompt_template:str='') -> None:
        super().__init__(random_seed, prompt_mode, prompt_template)

        # Dataset specific descriptions
        self.colum_info = {"target":"label", "input":["sentence"]}
        if dataset_path is None:
            self.file_name = f"./dataset"
        else:
            self.file_name = f"{dataset_path}/output.jsonl"
        self.labels = ["entailment", "not_entailment"]
        # Single pair sentences
        self.single_begginning = "Please read through this pair of sentence:\n"
        self.single_ending = "and determine whether the sentences \"entailment\", \"not entailment\" to each other. Give me the label only: "
        # Multiple pairs of sentences
        self.multiple_beginning = "Please read through these pair of sentences:\n"
        if self.prompt_template == 'plain':
            self.multiple_ending = "and determine whether the sentences \"entailment\", \"not entailment\" to each other. Give me the labels only: "
        elif self.prompt_template == 'json':
            self.multiple_ending = "and determine whether the sentences \"entailment\", \"not entailment\" to each other. Return in JSON format, such as: {\"1\": \"c_1\", \"2\":\"c_2\"}: "
    
    def load_dataset(self, portion:float=1.0, cutoff:int=1000):
        data = self.hf_datasets_loader(['glue', 'rte'], "validation")
        # housekeeping
        for each in data:
            if each[self.colum_info['target']] == 0:
                each[self.colum_info['target']] = 'not_entailment'
            elif each[self.colum_info['target']] == 1:
                each[self.colum_info['target']] = 'entailment'
            else:
                if each[self.colum_info['target']] not in self.labels:
                    raise ValueError(f"Invalid label: {each[self.colum_info['target']]}")
        data = self.random_portion_processor(data, portion, self.random_seed)
        data = data[:cutoff]
        self.data = data
        self.check_data_type(data)
        return data
    
    def prompt_generator(self, sentence1):
        if type(sentence1) == str and self.single:
            pass
        elif type(sentence1) == list and not self.single:
            pass
        else:
            raise ValueError(f"Invalid input types: sentence1 got {type(sentence1)}")
        return self.single_sentence_prompt_generator(sentence1)

    def case_conventor(self, text):
        text = text.lower()
        if text not in ['entailment','not entailment'] and len(text) > 0:
            if 'not entailment' in text:
                return 'not entailment'
            elif 'entailment' in text:
                return 'entailment'
            else:
                print('not sure what output: ', text)
                return 'na'
        return text
    
    def evaluation(self, model_name:str="gpt-3.5-turbo"):
        predicted_labels = []
        true_labels = []
        for each in self.data:
            predicted_labels.append(each[model_name])
            true_labels.append(each[self.colum_info['target']])
        return_dict = {}
        return_dict['Accuracy'] = accuracy_score(predicted_labels, true_labels)
        return return_dict