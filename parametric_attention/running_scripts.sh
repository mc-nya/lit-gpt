# --checkpoint_dir: The directory where the model is stored
# --data_dir: The directory where the dataset is stored
# --out_dir: The directory where the output will be stored
# --name: The name of the output file, if use wandb, this will be the name of the wandb run
# --config_file: The config file that will be used for the model initialization
# --devices: The number of devices to use for training, if >1 then we use distributed training

# For a text file dataset, we run full_instruct.py with the following command:
# This ensure we load tokenizers from the correct directory and use the correct config file.
python finetune/full_instruct.py \
--checkpoint_dir /scratch/jiasi_root/jiasi0/xuechenz/checkpoints/EleutherAI/pythia-160m \
--data_dir data/hellaswag \
--out_dir "/scratch/jiasi_root/jiasi0/xuechenz/paramattn/hellaswag/pythia-160m" \
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

