# --checkpoint_dir: The directory where the model is stored
# --data_dir: The directory where the dataset is stored
# --out_dir: The directory where the output will be stored
# --name: The name of the output file, if use wandb, this will be the name of the wandb run
# --config_file: The config file that will be used for the model initialization
# --devices: The number of devices to use for training, if >1 then we use distributed training

# For a text file dataset, we run full_instruct.py with the following command:
# This ensure we load tokenizers from the correct directory and use the correct config file.
python finetune/full_instruct.py \
--checkpoint_dir /scratch/oymak_root/oymak0/shared_data/checkpoints/EleutherAI/pythia-160m \
--data_dir data/hellaswag \
--out_dir "/scratch/oymak_root/oymak0/milii/paramattn/hellaswag/pythia-160m" \
--name "hellaswag-160-pythia" \
--config_file "configs/pythia_160m.json" \
--devices 2 

# For the train.bin/val.bin dataset, we run full_bindata.py
# This is the pre-processed binary data that we use for training, before running, we need to ensure that the processed data uses the correct tokenizers and config file.
# For generating the binary data, please refer to prepare_scripts.sh in the same directory.
python finetune/full_bindata.py \
--checkpoint_dir /scratch/oymak_root/oymak0/shared_data/checkpoints/EleutherAI/pythia-160m \
--data_dir /nfs/turbo/coe-sodalab/shared_data/lambada/ \
--out_dir "/scratch/oymak_root/oymak0/milii/paramattn/lambada/pythia-160m" \
--name "lambada-160-pythia" \
--config_file "configs/pythia_160m.json" \
--devices 2


mkdir -p /tmp/milii/lambada
cp -r /nfs/turbo/coe-sodalab/shared_data/lambada/* /tmp/milii/lambada
mkdir -p /tmp/milii/EleutherAI
cp -r /scratch/oymak_root/oymak0/shared_data/checkpoints/EleutherAI/pythia-160m /tmp/milii/EleutherAI
rm -rf /tmp/milii

python finetune/full_bindata_statistics.py \
--checkpoint_dir /tmp/milii/EleutherAI/pythia-160m \
--data_dir //tmp/milii/lambada/ \
--out_dir "/scratch/oymak_root/oymak0/milii/paramattn/lambada/pythia-160m" \
--name "lambada-160-pythia" \
--config_file "configs/pythia_160m.json" \
--devices 2

# Mistral experiment
python finetune/full_bindata_statistics.py \
--checkpoint_dir /scratch/oymak_root/oymak0/shared_data/checkpoints/mistralai/Mistral-7B-Instruct-v0.1 \
--data_dir /nfs/turbo/coe-sodalab/shared_data/lambada_mistral/ \
--out_dir "/scratch/oymak_root/oymak0/milii/paramattn/lambada/Mistral-7B/" \
--name "Mistral-7B-Instruct-v0.1" \
--config_file "configs/Mistral-7B.json" \
--devices 2


python finetune/full_bindata_finetune.py \
--checkpoint_dir /tmp/milii/EleutherAI/pythia-160m \
--data_dir //tmp/milii/lambada/ \
--out_dir "/scratch/oymak_root/oymak0/milii/paramattn/lambada/pythia-160m" \
--name "lambada-160-pythia" \
--config_file "configs/pythia_160m.json" \
--devices 4

python finetune/full_bindata.py \
--checkpoint_dir /tmp/milii/EleutherAI/pythia-160m \
--data_dir /tmp/milii/lambada/ \
--out_dir "/scratch/oymak_root/oymak0/milii/paramattn/lambada/pythia-debug" \
--name "lambada-160-pythia" \
--config_file "configs/pythia_160m.json" \
--devices 4


python finetune/full_bindata_finetune_param_only.py \
--checkpoint_dir /tmp/milii/EleutherAI/pythia-160m \
--data_dir //tmp/milii/lambada/ \
--out_dir "/scratch/oymak_root/oymak0/milii/paramattn/lambada/pythia-160m" \
--name "lambada-160-pythia" \
--config_file "configs/pythia_160m.json" \
--devices 2

python finetune/full_bindata_sparse.py \
--checkpoint_dir /tmp/milii/EleutherAI/pythia-160m \
--data_dir /tmp/milii/lambada/ \
--out_dir "/scratch/oymak_root/oymak0/milii/paramattn/lambada/pythia-debug" \
--name "lambada-160-pythia" \
--config_file "configs/pythia_160m.json" \
--devices 4
python finetune/full_bindata_sparse.py \
--checkpoint_dir /tmp/milii/EleutherAI/pythia-160m \
--data_dir /tmp/milii/lambada/ \
--out_dir "/scratch/oymak_root/oymak0/milii/paramattn/lambada/pythia-debug" \
--name "lambada-160-pythia" \
--config_file "configs/pythia-160m_4096.json" \
--devices 4

python finetune/full_gpt2_FT_PTS.py \
--checkpoint_dir pretrain_gpt2 \
--data_dir /scratch/oymak_root/oymak0/milii/datasets/openwebtext \
--out_dir "/scratch/oymak_root/oymak0/milii/paramattn/owt/gpt2_FT_PTS" \
--name "gpt2-FT_PTS" \
--config_file "configs/gpt_2_124M_original.json" \
--devices 4

python finetune/full_gpt2_PTS.py \
--checkpoint_dir pretrain_gpt2 \
--data_dir /scratch/oymak_root/oymak0/milii/datasets/openwebtext \
--out_dir "/scratch/oymak_root/oymak0/milii/paramattn/owt/gpt2_PTS" \
--name "gpt2-PTS" \
--config_file "configs/gpt_2_124M_original.json" \
--devices 4

python finetune/full_gpt2_FT.py \
--checkpoint_dir pretrain_gpt2 \
--data_dir /scratch/oymak_root/oymak0/milii/datasets/openwebtext \
--out_dir "/scratch/oymak_root/oymak0/milii/paramattn/owt/gpt2_PTS" \
--name "gpt2-PTS" \
--config_file "configs/gpt_2_124M_original.json" \
--devices 4