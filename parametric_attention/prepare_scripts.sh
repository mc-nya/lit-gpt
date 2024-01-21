# 1. prepare model
python scripts/download.py --repo_id EleutherAI/pythia-160m --checkpoint_dir /scratch/oymak_root/oymak0/shared_data/checkpoints
python scripts/convert_hf_checkpoint.py --checkpoint_dir /scratch/oymak_root/oymak0/shared_data/checkpoints/EleutherAI/pythia-160m
python scripts/download.py --repo_id EleutherAI/pythia-410m --checkpoint_dir /scratch/oymak_root/oymak0/shared_data/checkpoints
python scripts/convert_hf_checkpoint.py --checkpoint_dir /scratch/oymak_root/oymak0/shared_data/checkpoints/EleutherAI/pythia-410m

# 2. prepare data hellaswag
python scripts/prepare_flan.py \
--destination_path data/hellaswag \
--checkpoint_dir /scratch/oymak_root/oymak0/shared_data/checkpoints/EleutherAI/pythia-160m \
--subsets "hellaswag_10templates" \
--max_seq_length 2048 

# 3. prepare data lambada
python scripts/prepare_lambada.py \
--destination_path /nfs/turbo/coe-sodalab/shared_data/lambada/ \
--checkpoint_dir /scratch/oymak_root/oymak0/shared_data/checkpoints/EleutherAI/pythia-160m