import os
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def plot_scores(ax, data, metric_key, categories_to_plot, metric_name, y_limit, final_means, jitter_scale=0.1):
    decoding_methods = list(data.keys())
    x_pos_methods = np.arange(len(decoding_methods)) 
    colors = plt.cm.Dark2.colors # type: ignore
    legend_handles, legend_labels = list(), list()
    method_info = dict()

    for i, decoding_method in enumerate(decoding_methods): method_info[decoding_method] = {'color': colors[i % len(colors)]}

    for i, item in enumerate(categories_to_plot):
        cat_id, cat_label = item['id'], item['category']
            
        for m_idx, decoding_method in enumerate(decoding_methods):
            scores = data[decoding_method][cat_id].get(metric_key, [])
            mean_score = final_means[decoding_method][cat_id].get(metric_key)
            if not scores: continue

            # scatter
            x_base = x_pos_methods[m_idx]
            x_min_mean = x_base - jitter_scale * 1.5
            x_max_mean = x_base + jitter_scale * 1.5
            x_scatter = x_base + np.random.uniform(-jitter_scale, jitter_scale, len(scores))
            is_new_method = method not in legend_labels
            scatter = ax.scatter(
                x_scatter, scores,
                color=method_info[method]['color'],
                alpha=0.6,
                s=30,
                label=f'{method}' if is_new_method else None
            )
            if is_new_method:
                legend_handles.append(scatter)
                legend_labels.append(decoding_method)
            
            # horizontal mean (hlines) + text label
            if not np.isnan(mean_score): # type: ignore
                ax.hlines(
                    y=mean_score, 
                    xmin=x_min_mean,
                    xmax=x_max_mean,
                    color=method_info[decoding_method]['color'],
                    linestyle='-', 
                    linewidth=2.5, 
                    alpha=0.9,
                    zorder=3
                )
                ax.text(
                    x_base, 
                    mean_score + (y_limit[1] - y_limit[0]) * 0.015,
                    f'{mean_score:.3f}', 
                    color='black', 
                    fontsize=8, 
                    ha='center', 
                    va='bottom',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2')
                )
                 
        ax.set_title(f'{categories_to_plot[0]['category']}', fontsize=10)
        ax.xaxis.set_major_locator(ticker.FixedLocator(x_pos_methods)) # type: ignore
        ax.xaxis.set_major_formatter(ticker.FixedFormatter(decoding_methods))
        ax.set_xticklabels(decoding_methods, rotation=90, ha='center')

    ax.set_xticks(x_pos_methods)

    ax.set_ylabel(metric_name)
    ax.set_ylim(y_limit)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
   
    
    # mean_handle = plt.Line2D([0], [0], color='gray', linestyle='-', linewidth=2.5, label='Method Mean') # type: ignore
    # legend_handles.append(mean_handle)
    # legend_labels.append('Method Mean') # Renamed from 'Category Mean'
    
    # # Change legend position to avoid cluttering small subplots
    # ax.legend(legend_handles, legend_labels, loc='upper left', fontsize=7, fancybox=True, shadow=True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_card', type=str, default='Qwen/Qwen3-0.6B')
    args = parser.parse_args()

    model_name = args.model_card.replace('/', '_')
    eval_data_file_path = f'results_eval/{model_name}.jsonl'

    res = list()
    with open(eval_data_file_path, 'r', encoding='utf-8') as f:
        for line in f: res.append(json.loads(line))

    category_labels = list()
    with open('data/hallucination_categories.jsonl', 'r', encoding='utf-8') as f:
        for line in f: category_labels.append(json.loads(line))

    aggregated_scores: dict[str, dict[int, dict[str, list[float]]]] = dict()
    score_label_list = [
        {'label': 'bert_score_f1', 'name': 'BERTScore F1 (0.0 - 1.0)', 'y_limit': (0.55, 1.0),}, 
        {'label': 'llm_judge_factuality_score', 'name': 'Factuality Score by LLM Judge (1 - 5)', 'y_limit': (0.5, 5.5),},
        {'label': 'llm_judge_coherence_score', 'name': 'Coherence Score by LLM Judge (1 - 5)', 'y_limit': (0.5, 5.5),},
        {'label': 'adherence_check', 'name': 'Adherence Check (1 or 0)', 'y_limit': (0, 1.2),},
        {'label': 'refusal_adherence', 'name': 'Refusal Check (1 or 0)', 'y_limit': (0, 1.2),},
        {'label': 'numerical_accuracy', 'name': 'Numerical Accuracy Check (1 or 0)', 'y_limit': (0, 1.2),},
    ]

    for item in res:
        decoding_method = item.get('decoding_method')
        category = item.get('category')
        scores = item.get('scores', {})

        if decoding_method not in aggregated_scores:
            aggregated_scores[decoding_method] = {c['id']: {j['label']: [] for j in score_label_list} for c in category_labels}

        for item in score_label_list:
            score_label = item['label']
            if score_label in scores: aggregated_scores[decoding_method][category][score_label].append(scores[score_label])


    # cat average
    final_means: dict[str, dict[int, dict[str, float]]] = dict()
    for method, categories in aggregated_scores.items():
        final_means[method] = {}
        for cat_id, metrics in categories.items():
            final_means[method][cat_id] = {}
            for metric_name, score_list in metrics.items():
                if score_list:
                    final_means[method][cat_id][metric_name] = float(np.mean(score_list))
                else:
                    final_means[method][cat_id][metric_name] = np.nan # use nan when no scores are available

    fig, axes = plt.subplots(len(score_label_list), 5, figsize=(30, 40)) 
    
    for i, category_info in enumerate(category_labels):
        single_category_list = [category_info]
        for j, item in enumerate(score_label_list):
            label, name, y_limit = item['label'], item['name'], item['y_limit']
            plot_scores(
                ax=axes[j, i],
                data=aggregated_scores,
                metric_key=label,
                categories_to_plot=single_category_list,
                metric_name=name,
                y_limit=y_limit,
                final_means=final_means,
                jitter_scale=0.15, 
            )

    plt.suptitle(f'{model_name}', fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98]) # type: ignore
    
    fig_file_path = os.path.join('results_fig', f'{model_name}.jpg')
    os.makedirs(os.path.dirname(fig_file_path), exist_ok=True)
    plt.savefig(fig_file_path)
