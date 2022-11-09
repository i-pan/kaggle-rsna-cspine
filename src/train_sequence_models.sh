python main.py train configs/chunkseq/chunkseq003.yaml --num-workers 8 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 0

python main.py train configs/chunkseq/chunkseq003.yaml --num-workers 8 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 1

python main.py train configs/chunkseq/chunkseq003.yaml --num-workers 8 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 2

python main.py train configs/chunkseq/chunkseq003.yaml --num-workers 8 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 3

python main.py train configs/chunkseq/chunkseq003.yaml --num-workers 8 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 4


python main.py train configs/chunkseq/chunkseq005.yaml --num-workers 8 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 0

python main.py train configs/chunkseq/chunkseq005.yaml --num-workers 8 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 1

python main.py train configs/chunkseq/chunkseq005.yaml --num-workers 8 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 2

python main.py train configs/chunkseq/chunkseq005.yaml --num-workers 8 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 3

python main.py train configs/chunkseq/chunkseq005.yaml --num-workers 8 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 4


python main.py train configs/chunkseq/chunkseq006.yaml --num-workers 8 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 0

python main.py train configs/chunkseq/chunkseq006.yaml --num-workers 8 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 1

python main.py train configs/chunkseq/chunkseq006.yaml --num-workers 8 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 2

python main.py train configs/chunkseq/chunkseq006.yaml --num-workers 8 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 3

python main.py train configs/chunkseq/chunkseq006.yaml --num-workers 8 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 4
