import argparse
import json 
import logging
from tqdm import tqdm
from datetime import datetime
import os
import random
import overprompt

def timestamp():
    now = datetime.now()
    time = now.strftime("%H:%M:%S")
    return time

def main():
    parser = argparse.ArgumentParser(description="Perform basic arithmetic operations.")
    parser.add_argument('-d', '--dataset', required=True, type=str, help="Dataset name.")
    parser.add_argument('-p', '--prompt', type=int, default=0, help="Zero-shot or OverPrompt.")
    parser.add_argument('-m', '--model', type=str,  default="gpt-3.5-turbo", help="Large Language Model name, default \"gpt-3.5-turbo\".")
    parser.add_argument('-e', '--evaluation', type=bool, default=False, help="Perform evaluation.")
    parser.add_argument('-o', '--output', type=str, required=False, default=None, help="Pre-defined output path, generally use for rerun.")
    parser.add_argument('-q', '--portion', type=float, required=False, default=1.0, help="Portion of dataset to use, default 1.0.")
    parser.add_argument('-r', '--random_seed', type=int, default=None, help="Set random seed.")
    parser.add_argument('-a', '--analysis', type=bool, default=False, help="Set true to enable efficiency analysis.")
    parser.add_argument('-t', '--template', type=str, default='plain', help="Choose prompt template \"plain\" or \"json\".")
    parser.add_argument('-c', '--cutoff', type=int, default=1000, help="Cut off too long dataset, default \"1000\".")
    parser.add_argument('-l', '--log', type=bool, default=True, help="Set false to disable logging.")
    parser.add_argument('-k', '--key', type=str, required=False, help="OpenAI API key.")
    parser.add_argument('--permutation', type=str, required=False, help="permutation")

    args = parser.parse_args()

    # Check arguments
    if args.dataset == "":
        raise ValueError("Please specify a dataset.")
    if args.prompt < 0:
        raise ValueError(f"Invalid prompt mode {args.prompt}.Please specify a valid prompt mode.")
    if args.prompt < 0 and args.evaluation == False:
        raise ValueError("Please specify a prompt mode or evaluation mode.")
    
    # Check file structure
    output_dir = "./output"
    dataset_dir = "./dataset"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(dataset_dir):
        raise ValueError(f"Please download the dataset and put it in {dataset_dir} folder.")
    
    # Generate run name
    now = datetime.now()
    time = now.strftime("%m%d-%H%M")
    run_name = f'run-{time}'
    
    # Set logging
    if args.log:
        logging.basicConfig(filename=f'{output_dir}/run_logging.log', level=logging.INFO)
        logging.info("")
        logging.info(f" ------- Start running ------- ")
        logging.info(f"{run_name} arguments {args.__dict__}")

    # Initial Model
    OPENAI_API_KEY = ""
    if args.prompt == 0 and args.evaluation == True:
        # Evaluation only, no need to set API key
        pass
    elif args.prompt > 0 and args.model != "":
        if args.model in ["gpt-3.5-turbo", "gpt-3.5", "gpt-3"]:
            # Check if API key is set
            if args.key:
                OPENAI_API_KEY = args.key
            else:
                try:
                    with open('api_key.json', 'r') as file:
                        OPENAI_API_KEY = json.load(file)['api_key']
                except Exception as e:
                    raise e
            if OPENAI_API_KEY == "":
                if args.log:
                    logging.warning(f" {timestamp()} *** OpenAI API Key Not Set ***")
                raise ValueError("Please set your OpenAI API key in api_key.json or use -k option.")
            else:
                overprompt.set_openai_key(OPENAI_API_KEY)
                if args.log:
                    logging.info(f" OpenAI API key set SUCCESSFUL, key end with {OPENAI_API_KEY[-5:]}.")
        llm = overprompt.LLM(args.model)
        if args.template not in ['plain', 'json']:
            if args.log:
                logging.warning(f" {timestamp()} *** Invalid prompt template *** {args.template}")
            raise ValueError(f"Invalid prompt template {args.template}. Please specify a valid prompt template.")
        else:
            prompt_template = args.template

    # Set random seed
    random_seed = args.random_seed
    random.seed(args.random_seed)
    random_seed = None
    model = args.model

    # Load dataset and prompt templates
    dataset_name = args.dataset
    if args.output is not None:
        dataset_path = f"{output_dir}/{args.output}/output.jsonl"
        output_dir = f"{output_dir}/{args.output}/"
        output_path = dataset_path
        if not os.path.exists(output_path):
            if args.log:
                logging.warning(f" {timestamp()} *** Rerun Loading Path Not Found *** {output_path}")
            raise ValueError(f"Rerun loading path not found: {output_path}")
    else:
        output_dir = f"{output_dir}/{dataset_name}-{run_name}/"
        output_path = f"{output_dir}output.jsonl"
        dataset_path = None
    dataset_object = overprompt.data_loader(dataset_name=dataset_name, dataset_path=dataset_path, prompt_mode=args.prompt, random_seed=random_seed, prompt_template=prompt_template)
    if args.portion is not None:
        try:
            dataset_portion = float(args.portion)
        except Exception as e:
            raise ValueError(f"Invalid portion {args.portion}. Please specify a valid portion.")
        logging.info(f" {timestamp()} Dataset portion set {dataset_portion}.")
    else:
        dataset_portion = 1.0
    dataset_object.load_dataset(portion=dataset_portion, cutoff=args.cutoff)
    if args.log:
        logging.info(f" {timestamp()} Dataset successfully loaded, length {len(dataset_object.data)}.")
    
    if args.prompt > 0 and args.analysis:
        efficiency_analysis = overprompt.EffeciencyAnalysis(encoding_name=model)

    data_amount = len(dataset_object.data)
    # Run
    if args.prompt > 0:
        original_length = len(dataset_object.data)
        if args.prompt > 1:
            # OverPrompt
            n = int(args.prompt)
            dataset_object.data = [dataset_object.data[i:i+n] for i in range(0, len(dataset_object.data), n)]
            if args.permutation == "True":
                import itertools
                idx = 0
                dataset_object.data = [ list(each) for each in list(itertools.permutations(dataset_object.data[idx]))[:100]]
                print(len(dataset_object.data))
                print(dataset_object.data[0])

        generated = []
        if args.log:
            logging.info(f" {timestamp()} ------- Zero-shot Classification Started ------- ")
        for line in tqdm(dataset_object.data, total=len(dataset_object.data), desc="Processing lines"):
            # Prepare prompts
            empty = False
            if args.prompt == 1:
                if len(dataset_object.colum_info['input']) > 1:
                    sentence1 = line[dataset_object.colum_info['input'][0]]
                    sentence2 = line[dataset_object.colum_info['input'][1]]
                else:
                    sentence1 = line[dataset_object.colum_info['input'][0]]
                if model not in line.keys() or line[model] == '' or line[model] not in dataset_object.labels:
                    empty = True
                if len(dataset_object.colum_info['input']) > 1:
                    prompt = dataset_object.prompt_generator(sentence1, sentence2)
                else:
                    prompt = dataset_object.prompt_generator(sentence1)
            else:
                sentence1s = []
                sentence2s = []
                for row in line:
                    if len(dataset_object.colum_info['input']) > 1:
                        sentence1s += [row[dataset_object.colum_info['input'][0]]]
                        sentence2s += [row[dataset_object.colum_info['input'][1]]]
                    else:
                        sentence1s += [row[dataset_object.colum_info['input'][0]]]

                    if model not in row.keys() or line[model] == '' or line[model] not in dataset_object.labels:
                        empty = True 
                if len(dataset_object.colum_info['input']) > 1:
                    prompt = dataset_object.prompt_generator(sentence1s, sentence2s)
                else:
                    prompt = dataset_object.prompt_generator(sentence1s)

            # if any of the output is empty, generate response
            if empty:
                # print(prompt)
                # Use OpenAI API generate response
                if args.analysis:
                    start_time = datetime.now()
                try:
                    print(prompt)
                    gpt_gen = llm.gen_response(content=prompt)
                    print(gpt_gen)
                except Exception as e:
                    if args.log:
                        logging.warning(f" {timestamp()} *** OpenAI API Response Error *** ")
                        logging.warning(e)
                    print(e)
                    if args.prompt > 1:
                        gpt_gen = [""] * len(line)
                    else:
                        gpt_gen = ""
                if args.analysis:
                    end_time = datetime.now()
                    efficiency_analysis.add_checkpoint(start_time, end_time, prompt)

                # Convert cases
                if args.prompt > 1:
                    if prompt_template == 'json':
                        json_gpt_gen = json.loads(gpt_gen)
                        gpt_gen = [json_gpt_gen[each] for each in json_gpt_gen.keys()]
                    else:
                        if gpt_gen.count(',') > gpt_gen.count('\n'):
                            gpt_gen = gpt_gen.split(',')
                        else:
                            gpt_gen = gpt_gen.split('\n')
                    gpt_gen = [dataset_object.case_conventor(each) for each in gpt_gen]
                    for each in gpt_gen:
                        if each not in dataset_object.labels:
                            if args.log:
                                logging.warning(f" {timestamp()} *** Invalid Label *** {each}")
                                print(f"Invalid label: {each}")
                    if len(gpt_gen) != len(line):
                        if args.log:
                            logging.warning(f" {timestamp()} *** Unpaired Generated Length *** ")
                            logging.warning(f" {len(gpt_gen)} Generated:{gpt_gen}")
                        print(f"Unpaired generated length: {len(gpt_gen)}")
                    for each, result in zip(line, gpt_gen):
                        each[model] = result
                        generated += [each]
                else:
                    gpt_gen = dataset_object.case_conventor(gpt_gen)
                    line[model] = gpt_gen
                    generated += [line]
            else:
                if args.prompt > 1:
                    for each in line:
                        generated += [each]
                else:
                    generated += [line]
        
        # Efficiency analysis
        if args.analysis:
            total_time = efficiency_analysis.get_total_time_cost()
            average_time = efficiency_analysis.get_average_time_cost(original_length)
            average_token = efficiency_analysis.get_average_prompt_tokens(original_length)
            if args.log:
                logging.info(f" {timestamp()} ------- Efficiency Analysis ------- ")
                logging.info(f" Total time cost: {total_time} seconds")
                logging.info(f" Average time cost: {average_time} seconds/instance")
                logging.info(f" Average prompt tokens (tokens per line): {average_token} token/query")
            print(f" Total time cost: {total_time} seconds")
            print(f" Average time cost: {average_time} seconds/instance")
            print(f" Average prompt tokens (tokens per line): {average_token} token/query")

        if args.log:
            logging.info(f" {timestamp()} ------- Zero-shot Classification Finished ------- ")
            logging.info(f" Dataset length {len(generated)}.")
        if data_amount != len(generated):
            if args.log:
                logging.warning(f" {timestamp()} *** Unpaired Generated Length *** ")
                logging.warning(f" {data_amount} vs {len(generated)}")
            raise ValueError(f"Unpaired generated length: {data_amount} vs {len(generated)}")
        # Save generated data
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        dataset_object.save_generated(file_name=output_path, data=generated)
        dataset_object.data = generated
        if args.log:
            logging.info(f" {timestamp()} ------- Generated Data Saved ------- ")
            logging.info(f" Output path: {output_path}")

    # Evaluation
    if args.evaluation:
        if args.log:
            logging.info(f" {timestamp()} ------- Evaluation ------- ")

        evaluation_result = dataset_object.evaluation(model_name=model)
        for each in evaluation_result.keys():
            if args.log:
                logging.info(f" {each}: {evaluation_result[each]}")
            print(f" {each}: {evaluation_result[each]}")
        if args.permutation == "True":
            dataset_object.permutation_evaluation(n)

    if args.log:
        logging.info(f" {run_name} {timestamp()} ------- End of Run ------- ")

    
if __name__ == "__main__":
    main()
