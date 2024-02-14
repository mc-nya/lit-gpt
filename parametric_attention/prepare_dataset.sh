# prepare pg19 for different models
# 1. pythia
python scripts/prepare_pg19.py \
--destination_path /nfs/turbo/coe-sodalab/shared_data/pg19/ \
--checkpoint_dir /scratch/oymak_root/oymak0/shared_data/checkpoints/EleutherAI/pythia-160m

# 2. tiny llama
python scripts/prepare_pg19.py \
--destination_path /nfs/turbo/coe-sodalab/shared_data/pg19_tinyllama/ \
--checkpoint_dir /scratch/oymak_root/oymak0/shared_data/checkpoints/TinyLlama/TinyLlama-1.1B-Chat-v1.0

# 3. mistral
python scripts/prepare_pg19.py \
--destination_path /nfs/turbo/coe-sodalab/shared_data/pg19_mistral/ \
--checkpoint_dir /scratch/oymak_root/oymak0/shared_data/checkpoints/mistralai/Mistral-7B-Instruct-v0.1

# 4. stable-lm
python scripts/prepare_pg19.py \
--destination_path /nfs/turbo/coe-sodalab/shared_data/pg19_stablelm/ \
--checkpoint_dir /scratch/oymak_root/oymak0/shared_data/checkpoints/stabilityai/stablelm-base-alpha-3b

# prepare lambada for different models
# 1. pythia
python scripts/prepare_lambada.py \
--destination_path /nfs/turbo/coe-sodalab/shared_data/lambada/ \
--checkpoint_dir /scratch/oymak_root/oymak0/shared_data/checkpoints/EleutherAI/pythia-160m

# 2. tiny llama
python scripts/prepare_lambada.py \
--destination_path /nfs/turbo/coe-sodalab/shared_data/lambada_tinyllama/ \
--checkpoint_dir /scratch/oymak_root/oymak0/shared_data/checkpoints/TinyLlama/TinyLlama-1.1B-Chat-v1.0

# 3. mistral
python scripts/prepare_lambada.py \
--destination_path /nfs/turbo/coe-sodalab/shared_data/lambada_mistral/ \
--checkpoint_dir /scratch/oymak_root/oymak0/shared_data/checkpoints/mistralai/Mistral-7B-Instruct-v0.1

