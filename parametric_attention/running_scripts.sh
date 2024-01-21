python finetune/full_instruct.py \
--checkpoint_dir /scratch/oymak_root/oymak0/shared_data/checkpoints/EleutherAI/pythia-160m \
--data_dir data/hellaswag \
--out_dir "out/hellaswag" \
--name "hellaswag-160-pythia" \
--config_file "configs/pythia_160m.json" 
