#!/bin/bash
#SBATCH --job-name=cbr_8_512 # Job name
#SBATCH --partition=gpu 
#SBATCH --export=ALL 
#SBATCH --cpus-per-task=6        # Number of CPU cores per task (adjust as needed)
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH --mem=32G                # Request 32GB of memory (adjust as needed)
#SBATCH --time=48:00:00          # Maximum job runtime (adjust as needed)
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
conda list 
#single head cbr rnn
#python src/language_models/main.py --data /scratch2/mrenaudin/colorlessgreenRNNs/english_data --name test_cbr_scaled_adam --classmodel 'CBR_RNN' --batch_size 256 --lr 0.001 --cuda
#multi head cbr rnn
#python src/language_models/main.py --data /scratch2/mrenaudin/colorlessgreenRNNs/english_data --name multihead_cbr --classmodel 'CBR_RNN' --batch_size 256 --nheads 2 --lr 0.001 --cuda
#3 heads cbr rnn
#python src/language_models/main.py --data /scratch2/mrenaudin/colorlessgreenRNNs/english_data --name 8_heads_cbr --classmodel 'CBR_RNN' --batch_size 256 --nheads 8 --lr 0.001 --cuda
#cbr rnn growing dimension
#python src/language_models/main.py --data /scratch2/mrenaudin/colorlessgreenRNNs/english_data --name single_cbr_512 --classmodel 'CBR_RNN' --batch_size 256 --emsize 512 --nhid 512 --lr 0.001 --cuda
#lstm
#python src/language_models/main.py --data /scratch2/mrenaudin/colorlessgreenRNNs/english_data --name lstm_adam --classmodel 'RNNModel' --model 'LSTM' --emsize 650 --nhid 650 --batch_size 256 --lr 0.001 --cuda
#single cbr adam a retrain avec parametres par défaut pr checkpoint epoch 0 et 1
python src/language_models/main.py --data /scratch2/mrenaudin/colorlessgreenRNNs/english_data --name 8_heads_cbr_512 --classmodel 'CBR_RNN' --batch_size 256 --nheads 8 --emsize 512 --nhid 512 --lr 0.001 --cuda