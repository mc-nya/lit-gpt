# prepare phthia family
python scripts/download.py --repo_id EleutherAI/pythia-160m --checkpoint_dir /scratch/oymak_root/oymak0/shared_data/checkpoints
python scripts/convert_hf_checkpoint.py --checkpoint_dir /scratch/oymak_root/oymak0/shared_data/checkpoints/EleutherAI/pythia-160m
python scripts/download.py --repo_id EleutherAI/pythia-410m --checkpoint_dir /scratch/oymak_root/oymak0/shared_data/checkpoints
python scripts/convert_hf_checkpoint.py --checkpoint_dir /scratch/oymak_root/oymak0/shared_data/checkpoints/EleutherAI/pythia-410m
python scripts/download.py --repo_id EleutherAI/pythia-1b --checkpoint_dir /scratch/oymak_root/oymak0/shared_data/checkpoints
python scripts/convert_hf_checkpoint.py --checkpoint_dir /scratch/oymak_root/oymak0/shared_data/checkpoints/EleutherAI/pythia-1b

# tiny llama
python scripts/download.py --repo_id TinyLlama/TinyLlama-1.1B-Chat-v1.0 --checkpoint_dir /scratch/oymak_root/oymak0/shared_data/checkpoints
python scripts/convert_hf_checkpoint.py --checkpoint_dir /scratch/oymak_root/oymak0/shared_data/checkpoints/TinyLlama/TinyLlama-1.1B-Chat-v1.0

# mistral
python scripts/download.py --repo_id mistralai/Mistral-7B-Instruct-v0.1 --checkpoint_dir /scratch/oymak_root/oymak0/shared_data/checkpoints
python scripts/convert_hf_checkpoint.py --checkpoint_dir /scratch/oymak_root/oymak0/shared_data/checkpoints/mistralai/Mistral-7B-Instruct-v0.1

# stable-lm
python scripts/download.py --repo_id stabilityai/stablelm-base-alpha-3b --checkpoint_dir /scratch/oymak_root/oymak0/shared_data/checkpoints
python scripts/convert_hf_checkpoint.py --checkpoint_dir /scratch/oymak_root/oymak0/shared_data/checkpoints/stabilityai/stablelm-base-alpha-3b
