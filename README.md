# Layer-Contrast Decoding - DoLa

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

<hr>

## Core Mechanism: What is Decoding by Contrasting Layers (DoLa)

**Decoding by Contrasting Layers (DoLa)** was first introduced by Chuang et al in 2023 in the paper `[2]` as an inference-time strategy that intervenes in the conditional probability step and enhance the model’s factual knowledge.

The below diagram illustrates how DoLa works:

<img 
    src='https://miro.medium.com/v2/resize:fit:4800/format:webp/1*oL76PIGvdLsVju0T5Ih9GA.png'
    alt='DoLa for a transformer-based LM (Created by [Kuriko IWAI)'
/>

**Figure A.** DoLa for a transformer-based LM (Created by [Kuriko IWAI](https://kuriko-iwai.com))

### Layer Contrast: Boosting Factual Knowledge in Transformer LMs

From a model interpretability perspective, **transformer-based language models (LMs)** encode lower-level information in the lower (earlier) layers and more semantic information in the higher (later) layers `[3]`, with its topmost layers containing the knowledge neurons that express factual knowledge they acquired in the pretraining process `[4]`.

* **Lower layers** contain low-level linguistic features, syntax, local context
    
* **Higher layers** contain high-level semantic features, abstract reasoning, factual knowledge
    

DoLa exploits this modular encoding to amplify the factual knowledge through a contrastive decoding approach where the conditional probability for a next word is generated based on the *difference* in logits (raw prediction scores) between a higher layer and a lower layer.

In **Figure A**, **greedy search** selects “*Ottawa”* because the last layer (the 32th layer) of the transformer block predicts the highest conditional probability (72%) for that token.

**DoLa**, on the other hand, selects *“Ottawa”* because the adjusted logits using a contrast score between the 32nd and 24th layers for the token are the highest.

This approach helps emphasizing the factual knowledge of higher layers and downplaying knowledge of lower layers, making the model more factual and reducing hallucinations.

### The Contrastive Decoding Methodology

Standard LLMs compute the conditional probability of the next token $x_t$ being a specific vocabulary item $v$ such that:

```math
P(x_t = v \mid x_{&lt;t}) = \text{softmax}(\phi(h_t^{(N)}))_v \quad \text{for all } v \in \mathcal{X} \quad (1)
```

where

* $v$ is a specific token from the vocabulary drawn from the vocabulary set $X$,
    
* $x_{<t}$ is context, the sequence of all preceding tokens ${x1, x2, \cdots, x_t−1}$,
    
* $N$: The final layer (**mature layer**) in the transformer,
    
* $h_t^{(N)}$ is the hidden state in the final layer of the transformer with $N$ stacked layers, and
    
* $ϕ(⋅)$ is the language head (size: X) from a final linear layer that projects the hidden state h into a vector of logits.
    

Instead of the standard **Eq. (1)**, DoLa takes two major steps to compute the next token probability.

First, the prediction distribution $q_j(x_t)$ is computed for each candidate layer $j$ using the early-exit mechanism:

$$q_j(x_t) = \text{softmax}(\phi(h_t^{(j)})) \quad j \in \mathcal{J} \quad (2)$$

where $J$ denotes a set of early/intermediate layers.

The **premature layer** $M$ is then selected as the layer whose distribution $q_M$ is most distant from the one of the mature layer $q_N$ such that:

$$M = \arg \max_{j \in \mathcal{J}} d(q_N(\cdot), q_j(\cdot)) \quad (3)$$

where $d(,)$ denotes the Jensen-Shannon Divergence, and $q (⋅)$'s are from **Eq. (2)**.

Because DoLa leverages the differences of logits between layers, it expects that the significant difference in logits of the layer $M$ from the logits of the mature layer $N$ signals the layer $M$ has crucial factual knowledge that the model should integrate.

After selecting the premature layer M, DoLa computes the final probability for the next token such that

$$\hat{P}(x_t = v \mid x_{&lt;t}) = \text{softmax}(\mathcal{F}(q_N(x_t), q_M(x_t)))_{v} \quad (4)$$

where $F( , )$ computes the log-domain difference of the two distributions q’s in **Eq. (2)** such that:

$$\mathcal{F}(q_N(x_t), q_M(x_t)) = \begin{cases} \log \frac{q_N(x_t)}{q_M(x_t)}, & \text{if } x_t \in \mathcal{V}_{\text{head}}(x_{<t}), \\ -\infty, & \text{otherwise}. \end{cases} \quad (5)$$

where the set of candidate tokens $V_{head}(x < t)$ is defined as whether the token has high enough probabilities from the mature layer N (the selection criterion) `[5]` such thats:

$$\mathcal{V}_{\text{head}}(x{<t}) = \{x_t \in \mathcal{X} : q_N(x_t) \geq \alpha \max_{w} q_N(w)\} \quad (6)$$

where

* $q_N(x_t)$ is probability of the token $x_t$ in the mature layer $N$ being selected,
    
* $α ∈ [0, 1]$ is a **confidence threshold** (hyperparameter) to define the lower bound of the probability that the candidate token can take, and
    
* $w$ is any token in the entire vocabulary set $X$.
    

In other word, **Eq. (6)** indicates that a token $x_t$ is included in the candidate set *if only* its probability $q_N(x_t)$ is at least $α$ times the maximum probability $max_{w} q_N(w)$ among all tokens in the vocabulary set $X$.

And by computing the log-difference as defined in **Eq. (5)**, the model attempts to weigh the tokens that the mature layer N predicts highly, but the less-informed layer M did not.

### Dynamic vs. Static Selection of the Premature Layer M

**Eq. (3)** represents the objective function of dynamically selecting a premature layer $M$.

On the other hand, **DoLa-static** runs experiments on all possible early layers using a validation set and picks the one with the best validation performance.

This approach is more intuitive than the dynamic selection, but has drawbacks of:

1. Requiring more hyperparameter search runs in layers and
    
2. Best layers are sensitive to data distribution, thus requiring in-distribution (ID) validation sets where samples are drawn from the same underlying probability distribution as the training data.
    

In common scenarios where perfectly ID validation sets are unavailable, DoLa-static selects different optimal layers when evaluated on different subsets randomly sampled from the original dataset.

Dynamic selection can mitigate these drawbacks by shrinking the search space of the premature layer and making the method more robust without heavily relying on ID validation sets `[2]`.


<hr>

## References

\[1\]. [Survey of Hallucination in Natural Language Generation](https://arxiv.org/abs/2202.03629) (Ji et al., arXiv: 2202.03629)

\[2\]. [DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models](https://arxiv.org/abs/2309.03883) (Chuang et al., arXiv: 2309.03883)

\[3\]. [BERT Rediscovers the Classical NLP Pipeline](https://aclanthology.org/P19-1452/) (Tenney et al., ACL 2019)

\[4\]. [Knowledge Neurons in Pretrained Transformers](https://aclanthology.org/2022.acl-long.581/) (Dai et al., ACL 2022)

\[5\]. [Contrastive Decoding: Open-ended Text Generation as Optimization](https://aclanthology.org/2023.acl-long.687/) (Li et al., ACL 2023)

\[6\]. [CTRL: A Conditional Transformer Language Model for Controllable Generation](https://arxiv.org/abs/1909.05858) (Keskur et al., arXiv 1909.05858)
