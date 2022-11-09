conda create -y -n skp python=3.9 pip
conda activate skp
conda install -y pytorch=1.11 torchvision=0.12 torchaudio=0.11 cudatoolkit=11.3 -c pytorch
pip install pytorch-lightning==1.6
conda install -y pandas scikit-image scikit-learn
conda install -y -c conda-forge gdcm
pip install albumentations volumentations-3D
pip install kaggle omegaconf pydicom pytorchvideo nibabel
pip install timm transformers monai wandb