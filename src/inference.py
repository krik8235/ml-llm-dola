import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'critical'

import json
import argparse
import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.custom_dataset import CustomDataset



if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_card', type=str, default='Qwen/Qwen3-0.6B')
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--label', type=str, default=None)
    parser.add_argument('--dola_layers', type=str, default=None) # 'high', 'low'
    parser.add_argument('--max_new_tokens', type=int, default=258)
    parser.add_argument('--decoding_method', type=str, default='greedy') # or 'sample' - generationconfig accepts greedy or sample
    args = parser.parse_args()


    model_card = args.model_card
    input_list = [args.input] if args.input else CustomDataset().get_all_questions()
    label_list = [args.label] if args.label else CustomDataset().get_all_labels_and_categories()[0]
    category_list = [1] if len(input_list) == 1 else CustomDataset().get_all_labels_and_categories()[1]
    dola_layers = args.dola_layers; assert dola_layers in ['low', 'high', None]
    max_new_tokens = args.max_new_tokens
    decoding_method = args.decoding_method; assert decoding_method in ['greedy', 'sample']
    top_p = 0.9 if decoding_method == 'sample' else None


    # torch device
    device_type = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    device = torch.device(device_type)


    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_card)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
   

    # model
    model = AutoModelForCausalLM.from_pretrained(model_card, dtype=torch.float16).to(device) # type: ignore
    if model.generation_config is not None and model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = tokenizer.eos_token_id
    
    
    # inference
    print(f'\n... {model_card} is performing inference ...')
    results = list()
    for i, input in enumerate(input_list):
        print(f'\n... question # {i + 1}: {input}')
        tokens = tokenizer(input, return_tensors='pt', max_length=256, padding='max_length', truncation=True).to(device)

        start_time = datetime.datetime.now()
        outputs = model.generate(**tokens, max_new_tokens=max_new_tokens, do_sample=False, top_p=top_p, repetition_penalty=1.2) \
            if not dola_layers \
                else model.generate(
                    **tokens,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    top_p=top_p,
                    dola_layers=dola_layers,
                    custom_generate='src',
                    trust_remote_code=True,
                    synced_gpus=None,
                    streamer=None,
                    repetition_penalty=1.2,
                )    
        end_time = datetime.datetime.now()    
        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        if generated_text.startswith(input): generated_text = generated_text[len(input):].strip()
        print(f'... answer: {generated_text}')

        overall_time_in_ms = (end_time - start_time).total_seconds()
        results.append(
            dict(question=input, label=label_list[i], category=category_list[i], answer=generated_text, time_in_sec=overall_time_in_ms)
        )


    # store in jsonl
    output_file_path = os.path.join('results_inference', model_card.replace('/', '_'), f'dola_{dola_layers}_{decoding_method}.jsonl')
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(json.dumps(results) + '\n')
