# move data to local for faster access
# move lambada for different models
mkdir -p /tmp/milii/lambada && cp -r /nfs/turbo/coe-sodalab/shared_data/lambada/* /tmp/milii/lambada
# move pg19 for different models
mkdir -p /tmp/milii/pg19 && cp -r /nfs/turbo/coe-sodalab/shared_data/pg19/* /tmp/milii/pg19

# move pythia family
mkdir -p /tmp/milii/EleutherAI && cp -r /scratch/oymak_root/oymak0/shared_data/checkpoints/EleutherAI/* /tmp/milii/EleutherAI

# move tiny llama
mkdir -p /tmp/milii/TinyLlama && cp -r /scratch/oymak_root/oymak0/shared_data/checkpoints/TinyLlama/* /tmp/milii/TinyLlama

# move mistral
mkdir -p /tmp/milii/mistralai && cp -r /scratch/oymak_root/oymak0/shared_data/checkpoints/mistralai/* /tmp/milii/mistralai