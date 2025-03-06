#!/bin/bash
#SBATCH --job-name=cbr_test_bs # Job name
#SBATCH --partition=gpu 
#SBATCH --export=ALL 
#SBATCH --cpus-per-task=6        # Number of CPU cores per task (adjust as needed)
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH --mem=64G                # Request 16GB of memory (adjust as needed)
#SBATCH --time=40:00:00          # Maximum job runtime (adjust as needed)
#SBATCH --output=log/%x-%j.log

module load miniconda3/24.3.0
module load python/3.9.18-lpwk
module load cuda/11.8.0-r465 
source ~/.bashrc
conda activate leaps3

echo "Running job on $(hostname)"
which python
echo "python-version $(python --version)"
echo "CUDA_DEVICE: $CUDA_VISIBLE_DEVICES"
export CUDA_LAUNCH_BLOCKING=1

python cbr_rnn/main.py --model CBRRNN --data_dir ./datah5/ --objective lm --aux_objective --vocab_file vocab.txt --aux_vocab_file /scratch2/mrenaudin/colorlessgreenRNNs/datah5/aux_labels.txt --trainfname wikitext103_ccgtagged.train.hdf5 --validfname wikitext103_ccgtagged.valid.hdf5 --tied --model_file cbrrnn_test_model.pt --cuda --batch_size 512 --name cbr_whole_test
