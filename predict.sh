cd /code/
CUDA_VISIBLE_DEVICES=0 python inference_vllm.py --model_path final --max_new_tokens 1024 --temperature 0.1 --output_filepath submission.csv