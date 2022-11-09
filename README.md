# RSNA 2022 Cervical Spine Fracture Detection

## Hardware
Ubuntu 20.04 
8-core Intel processor
64 GB RAM
2 NVIDIA RTX 3090 GPUs with 24 GB VRAM

## Environment
```
bash src/environment.sh
conda activate skp
```

## Download Data
```
mkdir data
cd data
kaggle competitions download -c rsna-2022-cervical-spine-fracture-detection
```

Unzip the files. 

## Initial Setup
From `src/etl`:
```
python 00_extract_metadata.py
python 01_convert_to_png.py
python 02_convert_nifti_to_numpy.py
python 03_generate_whole_seg_192x192x192_numpy.py
python 04_create_cv_splits.py
python 05_create_cv_splits_for_whole_cspine_segmentation.py
```

## Train Segmentation Models
From `src`:
```
bash ./train_segmentation_models.sh 
```

## Generate Pseudo-segmentations
From `src/etl`:
```
python 06_pseudosegmentations_for_studies.py
python 07_add_pseudosegmentations_to_training.py
```

## Retrain Segmentation Models
From `src`:
```
bash ./retrain_segmentation_models.sh
```

## Crop Vertebra
From `src/etl`:
```
python 08_vertebra_locations_for_each_level.py
python 09_create_3d_chunk_for_each_vertebra.py
```

## Train 3D CNN Vertebra-level Classification Models
From `src`:
```
bash ./train_x3d_classifiers.sh
```

## Generate Slice-level Pseudo-labels
From `src/etl`:
```
python 10_get_cas.py
python 11_get_cas_pseudolabels.py
python 12_get_cropped_pngs.py
```

## Train TD CNN Classification Models
From `src`:
```
bash ./train_tdcnn_classifiers.py
```

## Extract Features
From `src/etl`:
```
python 13_extract_chunk_features_x3d.py
python 14_extract_chunk_features_tdcnn.py
python 15_fuse_features.py
```

## Train Final Sequence Models
From `src`:
```
bash ./train_sequence_models.py
```

## Inference 
See Kaggle notebook: https://www.kaggle.com/code/vaillant/rsna-c-spine-submission?scriptVersionId=108890272

