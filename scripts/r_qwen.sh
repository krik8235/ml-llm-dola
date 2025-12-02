CUDA_VISIBLE_DEVICES=0 python src/inference.py\
    --model_card Qwen/Qwen3-0.6B\
    --input "Who was the second person to walk on the moon, and what year did they do it?"\
    --dola_layer high\
    --max_new_tokens 128\
    --decoding_method greedy


# model card 
# https://huggingface.co/Qwen/Qwen3-0.6B