# DoLa

## Quick Start 

1. To run all experments:

```bash
chmod +x scripts/main.sh
uv run scripts/main.sh
```


2. To run a model or decoding method specific experiment:

```bash
uv run inference.py \
    --model_card [MODEL_CARD] --input [FACTUAL_QUESTION] --label [LABEL] --dola_layer [DOLA_LAYER_CONFIG]\
    --max_new_tokens [MAX_NEW_TOKENS] --decoding_method [DECODING_METHOD]
```

Each variable can take a value of your choice:

- **MODEL_CARD**: A model card from HuggingFace (Small models are not recommended as incremental benefits of DoLa might be limited).
- **FACTUAL_QUESTION**: A factual question.
- **LABEL**: A correct answer to the given question.
- **DOLA_LAYER_CONFIG**: Either `high` (using the higher part of the model layers), `low` (using the lower part of the model layer), or `None` (disable DoLa).
- **MAX_NEW_TOKENS**: Max new tokens allowed. `default = 256`.
- **DECODING_METHOD**: A deterministic decoding method. Either `greedy` (greedy search) or `sample` (top-p with p=0.90).

<hr>

## Inference

- Selected models perform inference: `src/inference.py`.

- Inference results are stored in the `results_inference` directory by model and decoding method.


### Updating

- Edit the base script `src/inference.py`.
- Edit the sh script `scripts/main.sh` or add new scripts to the `scripts` directory.


### Default Q&As

- Default Q&A set for inference is defined in `data/sample_questions.jsonl` by **hallucination categories**:

```jsonl
{"id": 1, "category": "Factual/Entity Errors"}
{"id": 2, "category": "Context Coherence"}
{"id": 3, "category": "Temporal Information"}
{"id": 4, "category": "Complex Reasoning"}
{"id": 5, "category": "Admitting Ignorance"}
```

- You can add, remove, or edit items at `data/sample_questions.jsonl`.

<hr>

## Evaluation (Metrics + LLM-as-a-Judge)

- Inference results are evaluated using auto metrics from HuggingFace and an LLM-as-a-Judge ([gpt-5-mini](https://platform.openai.com/docs/models/gpt-5-mini))

- To evaluate all results:

```bash
chmod +x scripts/eval.sh
uv run scripts/eval.sh
```

- Evaluation scores are stored in the `results_eval` directory, separated by model card.

### Updating

- Edit `src/evaluation.py` to add, remove, edit any eval metrics.
- Edit `llm_judge.py` to update an LLM-as-a-judge.


### Visualizing

- The following command will save bar graphs (.jpg file) in the `results_fig` directory:

```bash
chmod +x scripts/visualize.sh
uv run scripts/visualize.sh
```

<hr>

## Results

### Qwen/Qwen3-0.6B

<img src='results_fig/Qwen_Qwen3-0.6B.jpg' alt='score results by qwen' width=420 />

### meta-llama/Llama-3.2-1B

<img src='results_fig/meta-llama_Llama-3.2-1B.jpg' alt='eval results by llama' width=420 />


<hr>

## Dependency Control

- Install all dependencies:

```bash
uv venv
source .venv/bin/activate
uv sync
```

- Add/remove dependenies:

```bash
uv add [PACKAGE]
uv remove [PACAKGE]
```


- Reset the vertual environment:

```bash
rm -rf uv.lock .venv
uv cache clean
uv sync
```
