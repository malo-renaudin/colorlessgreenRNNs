#!/bin/bash
#SBATCH --job-name=cbr_1h_128_gs_shuffled_bptt100# Job name
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

# Enable device-side assertions

echo "Running job on $(hostname)"
which python
echo "python-version $(python --version)"
echo "CUDA_DEVICE: $CUDA_VISIBLE_DEVICES"


#single head cbr rnn
python src/language_models/main.py --data /scratch2/mrenaudin/colorlessgreenRNNs/english_data --name cbr_1h_128_gs_shuffled_bptt100 --classmodel 'CBR_RNN' --batch_size 256 --emsize 128 --nhid 128 --optimizer 'Adam' --gumbel_softmax --bptt 100 --cuda  #--checkpoint_path /scratch2/mrenaudin/colorlessgreenRNNs/checkpoints/cbr_1h_512/epoch_15.pt --epoch_checkpointed 15
#multi head cbr rnn
#python src/language_models/main.py --data /scratch2/mrenaudin/colorlessgreenRNNs/english_data --name cbr8h128_shuffling --classmodel 'CBR_RNN' --batch_size 256 --nheads 8 --emsize 128 --nhid 128 --optimizer 'Adam' --cuda
#3 heads cbr rnn
#python src/language_models/main.py --data /scratch2/mrenaudin/colorlessgreenRNNs/english_data --name 8_heads_cbr --classmodel 'CBR_RNN' --batch_size 256 --nheads 8 --lr 0.001 --cuda
#cbr rnn growing dimension
#python src/language_models/main.py --data /scratch2/mrenaudin/colorlessgreenRNNs/english_data --name single_cbr_512 --classmodel 'CBR_RNN' --batch_size 256 --emsize 512 --nhid 512 --lr 0.001 --cuda
#lstm
#python src/language_models/main.py --data /scratch2/mrenaudin/colorlessgreenRNNs/english_data --name lstm_adam_full_check_shuffled --classmodel 'RNNModel' --model 'LSTM' --emsize 650 --nhid 650 --batch_size 256 --optimizer 'Adam' --cuda 
#single cbr adam a retrain avec parametres par défaut pr checkpoint epoch 0 et 1
#python src/language_models/main.py --data /scratch2/mrenaudin/colorlessgreenRNNs/english_data --name 8_heads_cbr_512 --classmodel 'CBR_RNN' --batch_size 256 --nheads 8 --emsize 512 --nhid 512 --lr 0.001 --cuda
#stack lstm
#python src/language_models/main.py --data /scratch2/mrenaudin/colorlessgreenRNNs/english_data --name stack_lstm --classmodel 'Stack_LSTM' --batch_size 256 --emsize 256 --nhid 256 --optimizer 'Adam' --memory_dim 650 --memory_size 64 --cuda  #--checkpoint_path /scratch2/mrenaudin/colorlessgreenRNNs/checkpoints/cbr_1h_512/epoch_15.pt --epoch_checkpointed 15
#sparse cell state lstm
#python src/language_models/main.py --data /scratch2/mrenaudin/colorlessgreenRNNs/english_data --name lstm_sparse_cell --classmodel 'RNNModel' --model 'LSTM' --emsize 650 --nhid 650 --batch_size 256 --optimizer 'Adam' --cell_sparsity_lambda 1e-5 --cuda 
