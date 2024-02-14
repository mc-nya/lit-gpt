srun --jobid=1200954 --pty bash
salloc --nodes=1 --ntasks-per-node=4 --mem-per-cpu=20GB --gres=gpu:4 --time=14-00:00:00 --account=jiasi0 --partition=spgpu