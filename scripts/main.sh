##### qwen
# dola high
CUDA_VISIBLE_DEVICES=0 python src/inference.py\
    --model_card Qwen/Qwen3-0.6B\
    --dola_layer high\
    --max_new_tokens 256\
    --decoding_method greedy

CUDA_VISIBLE_DEVICES=0 python src/inference.py\
    --model_card Qwen/Qwen3-0.6B\
    --dola_layer high\
    --max_new_tokens 256\
    --decoding_method sample


# dola low
CUDA_VISIBLE_DEVICES=0 python src/inference.py\
    --model_card Qwen/Qwen3-0.6B\
    --dola_layer low\
    --max_new_tokens 256\
    --decoding_method greedy


CUDA_VISIBLE_DEVICES=0 python src/inference.py\
    --model_card Qwen/Qwen3-0.6B\
    --dola_layer low\
    --max_new_tokens 256\
    --decoding_method sample


# sample p = 0.9
CUDA_VISIBLE_DEVICES=0 python src/inference.py\
    --model_card Qwen/Qwen3-0.6B\
    --max_new_tokens 256\
    --decoding_method sample


# greedy
CUDA_VISIBLE_DEVICES=0 python src/inference.py\
    --model_card Qwen/Qwen3-0.6B\
    --max_new_tokens 256\
    --decoding_method greedy



##### llama 3.2 1B
# dola high
CUDA_VISIBLE_DEVICES=0 python src/inference.py\
    --model_card meta-llama/Llama-3.2-1B\
    --dola_layer high\
    --max_new_tokens 256\
    --decoding_method greedy

CUDA_VISIBLE_DEVICES=0 python src/inference.py\
    --model_card meta-llama/Llama-3.2-1B\
    --dola_layer high\
    --max_new_tokens 256\
    --decoding_method sample


# dola low
CUDA_VISIBLE_DEVICES=0 python src/inference.py\
    --model_card meta-llama/Llama-3.2-1B\
    --dola_layer low\
    --max_new_tokens 256\
    --decoding_method greedy


CUDA_VISIBLE_DEVICES=0 python src/inference.py\
    --model_card meta-llama/Llama-3.2-1B\
    --dola_layer low\
    --max_new_tokens 256\
    --decoding_method sample


# sample p = 0.9
CUDA_VISIBLE_DEVICES=0 python src/inference.py\
    --model_card meta-llama/Llama-3.2-1B\
    --max_new_tokens 256\
    --decoding_method sample


# greedy
CUDA_VISIBLE_DEVICES=0 python src/inference.py\
    --model_card meta-llama/Llama-3.2-1B\
    --max_new_tokens 256\
    --decoding_method greedy