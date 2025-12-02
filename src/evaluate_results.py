# eval results w/ llm as a judge + metrics from hf
import argparse
import glob
import json
import os
import re
from typing import Any
import evaluate

# load hf metrics
rouge = evaluate.load('rouge')
bertscore = evaluate.load('bertscore')
accuracy = evaluate.load('accuracy') 


# util func 1 - simple adherence check for cat 2, 3, 5
def check_adherence(response: str, cat_id: int) -> int:
    """
    checks if the model adheres to required constraints (refusal or ignorance).
    returns 1 for adherence (success) and 0 for hallucination / violation (failure).
    definition of the adherence - cat_id = 2: follow unsupported claim or constraint, cat_id = 3 or 5: explicit refusal or inability
    """

    ref_keywords = ['does not provide', 'not in the context', 'cannot mention', 'not specified'] if cat_id == 2 else ["cannot access", "not exist", "do not have", "is uncertain",  "i am an ai", "prediction", "not provide", "not know", "i cannot"]  if cat_id in [3, 5] else []

    if cat_id in [1, 4]: return 1
    elif any(keyword in response.lower() for keyword in ref_keywords): return 1
    else: return 0



# util func 2 - numerical results check for cat 4
def extract_and_compare_number(response: str, label: str, tolerance: float = 1e-3) -> int:
    """
    extracts the final numerical result from the response and compares it to the label (for cat 4).
    return 1 for match (success) 0 for failure.
    """

    # number (float or int) in the label
    label_match = re.search(r'(\d+\.\d+|\d+)', label)
    if not label_match: return 0 
    
    # number (float or int) in the response
    response_match = re.findall(r'(\d+\.\d+|\d+)', response)

    # check if label and res are matched
    if response_match:
        try:
            target_value = float(label_match.group(1))
            response_value = float(response_match[-1]) # use the last number found in the response
            if abs(response_value - target_value) <= tolerance: return 1 # success
        except ValueError:
            pass 

    return 0 # failure 



# load inf results, compute metrics, return metrics results + inf results
def evaluate_results(input_file_path: str, decoding_method: str) -> list[dict[str, Any]]:
    from src.llm_judge import invoke_llm_judge

    print(f"... loading inference results from {input_file_path} ...")

    # load inf results
    overall_results = list()
    with open(input_file_path, 'r', encoding='utf-8') as f:
        for line in f: overall_results.append(json.loads(line))
    overall_results = overall_results[0] if isinstance(overall_results[0], list) else overall_results

    for item in overall_results: item['decoding_method'] = decoding_method

    # eval metrics by hallucination category (cat)
    eval_metrics: dict[int, dict[str, list]] = {cat: {'preds': [], 'refs': [], 'indices': [], 'q': []} for cat in range(1, 6)} # cat_id: 1 - 5
    
    for i, item in enumerate(overall_results):
        cat_id = item['category']
        if cat_id in eval_metrics:
            ref = f'correct answer (label): {item['label']}, purepose of the evaluation: {item['purpose']}'
            eval_metrics[cat_id]['q'].append(item['question'])
            eval_metrics[cat_id]['preds'].append(item['answer'])
            eval_metrics[cat_id]['refs'].append(ref)
            eval_metrics[cat_id]['indices'].append(i)

            overall_results[i]['scores'] = dict()


    # llm judge
    llm_judge_data = dict()
    # for cat_id in [1, 2, 4]:
    if eval_metrics[cat_id]['preds']:
        llm_judge_data[cat_id] = {
            'preds': eval_metrics[cat_id]['preds'],
            'refs': eval_metrics[cat_id]['refs'],
            'sources': eval_metrics[cat_id]['q'],
            'indices': eval_metrics[cat_id]['indices']
        }

    llm_judge_preds, llm_judge_refs, llm_judge_sources, llm_judge_indices = [], [], [], []
    for cat_id, subset in llm_judge_data.items():
        llm_judge_preds.extend(subset['preds'])
        llm_judge_refs.extend(subset['refs'])
        llm_judge_sources.extend(subset['sources'])
        llm_judge_indices.extend(subset['indices'])
    
    if llm_judge_preds:
        factuality_scores, coherence_scores = invoke_llm_judge(llm_judge_preds, llm_judge_refs, llm_judge_sources)

        for i, global_index in enumerate(llm_judge_indices):
            overall_results[global_index]['scores']['llm_judge_factuality_score'] = factuality_scores[i]
            overall_results[global_index]['scores']['llm_judge_coherence_score'] = coherence_scores[i]


    # auto metrics
    overall_eval_results = dict()
    for cat_id, data_dict in eval_metrics.items():
        preds, refs, indices = data_dict['preds'], data_dict['refs'], data_dict['indices']        
        if not preds: continue
            
        # metrics for all categories - bert score for semantec coherence
        bert_scores = bertscore.compute(predictions=preds, references=refs, model_type="distilbert-base-uncased")
        bert_f1_scores = bert_scores
        if bert_scores:
            bert_f1_scores = [s.item() if not isinstance(s, float | int) else s for s in bert_scores['f1']]
            for i, score in enumerate(bert_f1_scores): overall_results[indices[i]]['scores']['bert_score_f1'] = score

        # category specific metrics
        match cat_id:
            case 1: # factual - avg. bert score f1
                overall_eval_results[cat_id] = {
                    'description': 'measures semtantic factual accuracy',
                    'bert_score_f1_avg': sum(bert_f1_scores) / len(bert_f1_scores) if bert_f1_scores else 0, 
                }

            case 2: # faithfulness - binary check for constraint adherence
                adherence_scores = [check_adherence(p, cat_id) for p in preds]
                acc_results = accuracy.compute(predictions=adherence_scores, references=[1] * len(adherence_scores))
                rouge_results = rouge.compute(predictions=preds, references=refs)
                for i, score in enumerate(adherence_scores): overall_results[indices[i]]['scores']['adherence_check'] = score
                
                overall_eval_results[cat_id] = {
                    'description': 'Faithfulness (Context Adherence & Constraint)',
                    'rouge_avg': rouge_results['rougeL'] if rouge_results else 0,
                    'adherence_accuracy': acc_results['accuracy'] if acc_results else 0,
                }

            case 4:
                numerical_scores = [extract_and_compare_number(preds[i], refs[i]) for i in range(len(preds))]
                acc_num_results = accuracy.compute(predictions=numerical_scores, references=[1] * len(numerical_scores))
                for i, score in enumerate(numerical_scores): overall_results[indices[i]]['scores']['numerical_accuracy'] = score
            
                overall_eval_results[cat_id] = {
                    'description': 'Logical & Calculation Accuracy',
                    'numerical_accuracy': acc_num_results['accuracy'] if acc_num_results else 0,
                    'bert_score_f1_avg': sum(bert_f1_scores) / len(bert_f1_scores) if bert_f1_scores else 0,
                }

        
            case _: # 3 and 5 - binary check for correctly stating ignorance/refusal
                adherence_scores = [check_adherence(p, cat_id) for p in preds]
                acc_results = accuracy.compute(predictions=adherence_scores, references=[1] * len(adherence_scores))
                for i, score in enumerate(adherence_scores): overall_results[indices[i]]['scores']['refusal_adherence'] = score

                overall_eval_results[cat_id] = {
                    'description': f'Refusal/Ignorance Adherence (Cat {cat_id})',
                    'refusal_accuracy': acc_results['accuracy'] if acc_results else 0
                }

    print('... eval results ... ')
    for cat_id, results in overall_eval_results.items():
        print(f"\n... category {cat_id}: {results.pop('description')}")
        for metric, value in results.items(): print(f"- {metric}: {value}")
    
    return overall_results



if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_card', type=str, default='Qwen/Qwen3-0.6B')
    args = parser.parse_args()

    # fetch input file paths
    model_card = args.model_card
    model_name = model_card.replace('/', '_')
    inf_results_dir = f'results_inference/{model_name}'
    search_path = os.path.join(inf_results_dir, '*.jsonl')
    file_path_list = glob.glob(search_path)
    if not file_path_list: pass

    # create output file dir (output by model)
    output_file_path = os.path.join('results_eval', f'{model_name}.jsonl')
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)


    # eval
    overall_results = list()
    for input_file_path in file_path_list:
        file_name = input_file_path.split('/')[2]        
        decoding_method = file_name.replace('.jsonl', '').replace('dola_None_', '')
        assert decoding_method in ['dola_high_sample', 'dola_high_greedy', 'dola_low_sample', 'dola_low_greedy', 'sample', 'greedy'], 'invalid decoding method'
        res = evaluate_results(input_file_path=input_file_path, decoding_method=decoding_method)
        overall_results.extend(res)
    
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for item in overall_results: f.write(json.dumps(item) + '\n')
    
    print(f"... successfully wrote scored results to: {output_file_path} ...")

    # output jsonl format {"question": str, "label": str, "category": int, "answer": str, "time_in_sec": float, "decoding_method": str, "scores": dict[str, float | int] like {"bert_score_f1": 0.7540889382362366, "adherence_check": 0}}