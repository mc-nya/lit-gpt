# --checkpoint_dir: The directory where the model checkpoints are saved
# --model_file: The name of the sepcific checkpoint e.g. lit_model_finetuned.pth, iter-000000-ckpt.pth
# --tokenizer_dir: The directory where the tokenizers are saved, usually the official model checkpoint directory
# --config_filepath: The config file that will be used for the model initialization
# --eval_tasks: The tasks that we want to evaluate the model on, e.g. hellaswag, lambada, etc.

# Evaluate the model on the Hellaswag dataset after training
python eval/tfpp_eval.py \
    --checkpoint_dir "out/hellaswag" \
    --model_file "lit_model_finetuned.pth" \
    --tokenizer_dir "/scratch/oymak_root/oymak0/shared_data/checkpoints/EleutherAI/pythia-160m" \
    --config_filepath "configs/pythia_160m.json" \
    --eval_tasks "[hellaswag]" \
    --precision "32-true" \
    --save_filepath "out/eval/EleutherAI_pythia-160m.json"

# Evaluation the initial model at ckpt 0
python eval/tfpp_eval.py \
    --checkpoint_dir "out/hellaswag" \
    --model_file "iter-000000-ckpt.pth" \
    --tokenizer_dir "/scratch/oymak_root/oymak0/shared_data/checkpoints/EleutherAI/pythia-160m" \
    --config_filepath "configs/pythia_160m.json" \
    --eval_tasks "[hellaswag]" \
    --precision "32-true" \
    --save_filepath "out/eval/EleutherAI_pythia-160m_ckpt0.json"