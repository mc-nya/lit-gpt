# 1. prepare model
python scripts/download.py --repo_id EleutherAI/pythia-160m --checkpoint_dir /scratch/jiasi_root/jiasi0/xuechenz/checkpoints
python scripts/convert_hf_checkpoint.py --checkpoint_dir /scratch/jiasi_root/jiasi0/xuechenz/checkpoints/EleutherAI/pythia-160m
python scripts/download.py --repo_id EleutherAI/pythia-410m --checkpoint_dir /scratch/jiasi_root/jiasi0/xuechenz/checkpoints
python scripts/convert_hf_checkpoint.py --checkpoint_dir /scratch/jiasi_root/jiasi0/xuechenz/checkpoints/EleutherAI/pythia-410m

# 2. prepare data hellaswag (text dataset)
python scripts/prepare_flan.py \
--destination_path data/hellaswag \
--checkpoint_dir /scratch/jiasi_root/jiasi0/xuechenz/checkpoints/EleutherAI/pythia-160m \
--subsets "hellaswag_10templates" \
--max_seq_length 2048 

# 3. prepare data lambada (bindata)
python scripts/prepare_lambada.py \
--destination_path data/lambada/ \
--checkpoint_dir /scratch/jiasi_root/jiasi0/xuechenz/checkpoints/EleutherAI/pythia-160m

python scripts/prepare_pg19.py \
--destination_path /nfs/turbo/coe-sodalab/shared_data/pg19/ \
--checkpoint_dir /scratch/oymak_root/oymak0/shared_data/checkpoints/EleutherAI/pythia-160m



# Mistralai experiment
python scripts/download.py --repo_id mistralai/Mistral-7B-Instruct-v0.1 --checkpoint_dir /scratch/oymak_root/oymak0/shared_data/checkpoints
python scripts/convert_hf_checkpoint.py --checkpoint_dir /scratch/oymak_root/oymak0/shared_data/checkpoints/mistralai/Mistral-7B-Instruct-v0.1

python scripts/prepare_lambada.py \
--destination_path /nfs/turbo/coe-sodalab/shared_data/lambada_mistral/ \
--checkpoint_dir /scratch/oymak_root/oymak0/shared_data/checkpoints/mistralai/Mistral-7B-Instruct-v0.1

# copy weight and data to local /tmp/milii folder
mkdir -p /tmp/milii/lambada_mistral
cp -r /nfs/turbo/coe-sodalab/shared_data/lambada_mistral/* /tmp/milii/lambada_mistral
mkdir -p /tmp/milii/mistralai
cp -r /scratch/oymak_root/oymak0/shared_data/checkpoints/mistralai/Mistral-7B-Instruct-v0.1 /tmp/milii/mistralai
rm -rf /tmp/milii

python finetune/full_bindata_statistics.py \
--checkpoint_dir /tmp/milii/mistralai/Mistral-7B-Instruct-v0.1 \
--data_dir /tmp/milii/lambada_mistral/ \
--out_dir "/scratch/oymak_root/oymak0/milii/paramattn/lambada/Mistral-7B/" \
--name "Mistral-7B-Instruct-v0.1" \
--config_file "configs/Mistral-7B.json" \
--devices 4

python finetune/full.py \
--checkpoint_dir /tmp/milii/mistralai/Mistral-7B-Instruct-v0.1 \
--data_dir /tmp/milii/lambada_mistral/ \
--out_dir "/scratch/oymak_root/oymak0/milii/paramattn/lambada/Mistral-7B/" \
--devices 4


# stable-lm experiment
python scripts/download.py --repo_id stabilityai/stablelm-base-alpha-3b --checkpoint_dir /scratch/oymak_root/oymak0/shared_data/checkpoints
python scripts/convert_hf_checkpoint.py --checkpoint_dir /scratch/oymak_root/oymak0/shared_data/checkpoints/stabilityai/stablelm-base-alpha-3b

python scripts/prepare_lambada.py \
--destination_path /nfs/turbo/coe-sodalab/shared_data/lambada_stablelm/ \
--checkpoint_dir /scratch/oymak_root/oymak0/shared_data/checkpoints/stabilityai/stablelm-base-alpha-3b

# copy weight and data to local /tmp/milii folder
mkdir -p /tmp/milii/lambada_stablelm
cp -r /nfs/turbo/coe-sodalab/shared_data/lambada_stablelm/* /tmp/milii/lambada_stablelm
mkdir -p /tmp/milii/stabilityai
cp -r /scratch/oymak_root/oymak0/shared_data/checkpoints/stabilityai/stablelm-base-alpha-3b /tmp/milii/stabilityai
rm -rf /tmp/milii

python finetune/full_bindata_statistics.py \
--checkpoint_dir /tmp/milii/stabilityai/stablelm-base-alpha-3b \
--data_dir /tmp/milii/lambada_stablelm/ \
--out_dir "/scratch/oymak_root/oymak0/milii/paramattn/lambada/stablelm-3B/" \
--name "stablelm-base-alpha-3b" \
--config_file "configs/stablelm-3B.json" \
--devices 4

python finetune/full.py \
--checkpoint_dir /tmp/milii/mistralai/Mistral-7B-Instruct-v0.1 \
--data_dir /tmp/milii/lambada_mistral/ \
--out_dir "/scratch/oymak_root/oymak0/milii/paramattn/lambada/Mistral-7B/" \
--devices 4

# TinyLLama experiment
python scripts/download.py --repo_id TinyLlama/TinyLlama-1.1B-Chat-v1.0 --checkpoint_dir /scratch/oymak_root/oymak0/shared_data/checkpoints
python scripts/convert_hf_checkpoint.py --checkpoint_dir /scratch/oymak_root/oymak0/shared_data/checkpoints/TinyLlama/TinyLlama-1.1B-Chat-v1.0

python scripts/prepare_pg19.py \
--destination_path /nfs/turbo/coe-sodalab/shared_data/pg19_tinyllama/ \
--checkpoint_dir /scratch/oymak_root/oymak0/shared_data/checkpoints/TinyLlama/TinyLlama-1.1B-Chat-v1.0

# copy weight and data to local /tmp/milii folder
mkdir -p /tmp/milii/pg19_tinyllama
cp -r /nfs/turbo/coe-sodalab/shared_data/pg19_tinyllama/* /tmp/milii/pg19_tinyllama
mkdir -p /tmp/milii/TinyLlama
cp -r /scratch/oymak_root/oymak0/shared_data/checkpoints/TinyLlama/TinyLlama-1.1B-Chat-v1.0 /tmp/milii/TinyLlama

