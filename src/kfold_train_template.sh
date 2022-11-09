ts python main.py train configs/chunkseq/chunkseq401.yaml --num-workers 8 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 0

ts python main.py train configs/chunkseq/chunkseq401.yaml --num-workers 8 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 1

ts python main.py train configs/chunkseq/chunkseq401.yaml --num-workers 8 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 2

ts python main.py train configs/pre/pre200.yaml --num-workers 8 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 2

ts python main.py train configs/pre/pre200.yaml --num-workers 8 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 3

ts python main.py train configs/pre/pre200.yaml --num-workers 8 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 4

ts python main.py train configs/chunk/chunk203.yaml --num-workers 8 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 2

ts python main.py train configs/cascrop/cascrop200.yaml --num-workers 4 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 3

ts python main.py train configs/cascrop/cascrop200.yaml --num-workers 4 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 4

ts python main.py train configs/seg/seg101.yaml --num-workers 4 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 3

ts python main.py train configs/seg/seg101.yaml --num-workers 4 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 4


ts python main.py train configs/chunk/chunk100.yaml --num-workers 4 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 0

ts python main.py train configs/chunk/chunk101.yaml --num-workers 4 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 1

ts python main.py train configs/chunk/chunk101.yaml --num-workers 4 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 2

ts python main.py train configs/chunk/chunk101.yaml --num-workers 4 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 3

ts python main.py train configs/chunk/chunk101.yaml --num-workers 4 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 4

ts python main.py train configs/chunkseq/chunkseq003.yaml --num-workers 4 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 1

ts python main.py train configs/chunkseq/chunkseq003.yaml --num-workers 4 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 2

ts python main.py train configs/chunkseq/chunkseq003.yaml --num-workers 4 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 3

ts python main.py train configs/chunkseq/chunkseq003.yaml --num-workers 4 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 4

ts python main.py train configs/chunkseq/chunkseq003.yaml --num-workers 4 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 0

python main.py train configs/casseq/casseq005.yaml --num-workers 4 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 0

ts python main.py train configs/chunk/chunk004.yaml --num-workers 4 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 1

ts python main.py train configs/chunk/chunk004.yaml --num-workers 4 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 2

ln -s ../../experiments/chunkseq100/sbn/fold0/checkpoints/best.ckpt fold0.ckpt
ln -s ../../experiments/chunkseq100/sbn/fold1/checkpoints/best.ckpt fold1.ckpt
ln -s ../../experiments/chunkseq100/sbn/fold2/checkpoints/best.ckpt fold2.ckpt
ln -s ../../experiments/chunkseq004/sbn/fold3/checkpoints/best.ckpt fold3.ckpt
ln -s ../../experiments/chunkseq004/sbn/fold4/checkpoints/best.ckpt fold4.ckpt

ln -s ../../experiments/caschunk001/sbn/fold0/checkpoints/best.ckpt fold0.ckpt
ln -s ../../experiments/caschunk001/sbn/fold1/checkpoints/best.ckpt fold1.ckpt
ln -s ../../experiments/caschunk001/sbn/fold2/checkpoints/best.ckpt fold2.ckpt
ln -s ../../experiments/caschunk001/sbn/fold3/checkpoints/best.ckpt fold3.ckpt
ln -s ../../experiments/caschunk001/sbn/fold4/checkpoints/best.ckpt fold4.ckpt


ts python main.py train configs/chunk/chunk004.yaml --num-workers 4 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 3

ts python main.py train configs/chunk/chunk004.yaml --num-workers 4 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 4

ts python main.py train configs/seg/seg001.yaml --num-workers 4 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 4

ts python main.py train configs/casseq/casseq004.yaml --num-workers 8 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 0

ts python main.py train configs/casseq/casseq004.yaml --num-workers 8 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 1

ts python main.py train configs/casseq/casseq004.yaml --num-workers 8 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 2

ts python main.py train configs/casseq/casseq004.yaml --num-workers 8 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 3

ts python main.py train configs/casseq/casseq004.yaml --num-workers 8 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 4



ts python main.py train configs/seg/seg000.yaml --num-workers 8 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 0 --log_every_n_steps 5 --check_val_every_n_epoch 10

ts python main.py train configs/mk3d/mk3d008.yaml --num-workers 2 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 1 --log_every_n_steps 5 --check_val_every_n_epoch 10

ts python main.py train configs/mk3d/mk3d008.yaml --num-workers 2 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 2 --log_every_n_steps 5 --check_val_every_n_epoch 10

ts python main.py train configs/mk3d/mk3d008.yaml --num-workers 2 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 3 --log_every_n_steps 5 --check_val_every_n_epoch 10

ts python main.py train configs/mk3d/mk3d008.yaml --num-workers 2 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 4 --log_every_n_steps 5 --check_val_every_n_epoch 10


ts python main.py train configs/mk3d/mk3d001.yaml --num-workers 2 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 0

ts python main.py train configs/mk3d/mk3d001.yaml --num-workers 2 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 1

ts python main.py train configs/mk3d/mk3d001.yaml --num-workers 2 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 2

ts python main.py train configs/mk3d/mk3d001.yaml --num-workers 2 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 3

ts python main.py train configs/mk3d/mk3d001.yaml --num-workers 2 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 4


ts python main.py train configs/seg/seg000.yaml --num-workers 8 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 0 --check_val_every_n_epoch 10

ts python main.py train configs/mk3d/mk3d002.yaml --num-workers 2 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 1

ts python main.py train configs/mk3d/mk3d002.yaml --num-workers 2 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 2

ts python main.py train configs/mk3d/mk3d002.yaml --num-workers 2 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 3

ts python main.py train configs/mk3d/mk3d002.yaml --num-workers 2 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 4


ts python main.py train configs/cas/cas001.yaml --num-workers 4 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 0

ts python main.py train configs/cas/cas001.yaml --num-workers 4 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 1

ts python main.py train configs/cas/cas001.yaml --num-workers 4 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 2

ts python main.py train configs/cas/cas001.yaml --num-workers 4 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 3

ts python main.py train configs/cas/cas001.yaml --num-workers 4 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 4

python main.py train configs/chunk/chunk000.yaml --num-workers 4 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 0 --check_val_every_n_epoch 1


ts python main.py train configs/chunk/chunk003.yaml --num-workers 8 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 0 --check_val_every_n_epoch 1

ts python main.py train configs/chunk/chunk003.yaml --num-workers 8 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 1 --check_val_every_n_epoch 1

ts python main.py train configs/chunk/chunk003.yaml --num-workers 8 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 2 --check_val_every_n_epoch 1

ts python main.py train configs/chunk/chunk003.yaml --num-workers 8 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 3 --check_val_every_n_epoch 1

ts python main.py train configs/chunk/chunk003.yaml --num-workers 8 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 4 --check_val_every_n_epoch 1

ts python main.py train configs/chunk/chunk003.yaml --num-workers 12 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 0 --check_val_every_n_epoch 1

ts python main.py train configs/chunk/chunk002.yaml --num-workers 8 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 0 --seed 87 --group-by-seed

ts python main.py train configs/cas/cas001.yaml --num-workers 16 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 0 --check_val_every_n_epoch 2 --seed 88 --group-by-seed

ts python main.py train configs/pre/pre004.yaml --num-workers 2 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 0 --check_val_every_n_epoch 2 --seed 89 --group-by-seed



python main.py train configs/feat/feat000.yaml --num-workers 8 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 0

ts python main.py train configs/seg/pseudoseg000.yaml --num-workers 4 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 0 --check_val_every_n_epoch 10 --log_every_n_steps 5

ts python main.py train configs/seg/seg000.yaml --num-workers 4 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 1 --check_val_every_n_epoch 10 --log_every_n_steps 5

ts python main.py train configs/seg/seg000.yaml --num-workers 4 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 2 --check_val_every_n_epoch 10 --log_every_n_steps 5

ts python main.py train configs/seg/seg000.yaml --num-workers 4 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 3 --check_val_every_n_epoch 10 --log_every_n_steps 5

ts python main.py train configs/seg/seg000.yaml --num-workers 4 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 4 --check_val_every_n_epoch 10 --log_every_n_steps 5


ts python main.py train configs/chunk/chunk002.yaml --num-workers 4 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 0

ts python main.py train configs/chunk/chunk002.yaml --num-workers 4 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 1

ts python main.py train configs/chunk/chunk002.yaml --num-workers 4 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 2

ts python main.py train configs/chunk/chunk002.yaml --num-workers 4 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 3

ts python main.py train configs/chunk/chunk002.yaml --num-workers 4 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 4

ts python main.py train configs/cas/cas001.yaml --num-workers 16 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 0

ts python main.py train configs/cas/cas001_512.yaml --num-workers 16 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 0

ts python main.py train configs/cas/cas001_640.yaml --num-workers 16 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 0


ts python main.py train configs/pre/pre001.yaml --num-workers 16 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 0 --seed 87 --group-by-seed

ts python main.py train configs/pre/pre001.yaml --num-workers 16 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 0 --seed 88 --group-by-seed

ts python main.py train configs/pre/pre001.yaml --num-workers 16 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 0 --seed 89 --group-by-seed


ln -s ../../experiments/chunkseq300/sbn/fold0/checkpoints/best.ckpt fold0.ckpt
ln -s ../../experiments/chunkseq300/sbn/fold1/checkpoints/best.ckpt fold1.ckpt
ln -s ../../experiments/chunkseq300/sbn/fold2/checkpoints/best.ckpt fold2.ckpt
ln -s ../../experiments/seg101/sbn/fold3/checkpoints/best.ckpt fold3.ckpt
ln -s ../../experiments/seg101/sbn/fold4/checkpoints/best.ckpt fold4.ckpt


(3252+3180+3760+3364+2816) / 5