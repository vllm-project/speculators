from huggingface_hub import snapshot_download
model_id="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
local_path=snapshot_download(repo_id=model_id, cache_dir="eagle3")

