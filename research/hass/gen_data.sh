#This file is adapted from https://github.com/HArmonizedSS/HASS (arxiv: https://arxiv.org/abs/2408.15766)
#Which is a fork of the Eagle repository: https://github.com/SafeAILab/EAGLE (arxiv: https://arxiv.org/abs/2401.15077)

# to get the dataset, run: wget https://huggingface.co/datasets/Aeala/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V4.3_unfiltered_cleaned_split.json .

python -m ge_data.allocation \
--outdir dataDirectory/sharegpt \
--data_path ShareGPT_V4.3_unfiltered_cleaned_split.json \
--model_path meta-llama/Llama-3.1-8B-Instruct \
--chat_template llama \
--dataset sharegpt


# python -m ge_data.allocation \
# --outdir dataDirectory/sharegpt \
# --data_path ShareGPT_V4.3_unfiltered_cleaned_split.json \
# --model_path mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
# --chat_template mistral \
# --dataset sharegpt
