# Generative Language-Grounded Policy (GLGP) for Vision-and-Language Navigation

This repository contains the code for the following paper:

Shuhei Kurita and Kyunghyun Cho, *Generative Language-Grounded Policy in Vision-and-Language Navigation with Bayes' Rule*. in ICLR, 2021. ([PDF](https://openreview.net/pdf?id=45uOPa46Kh))
```
@inproceedings{kurita2021glgp,
  title={Generative Language-Grounded Policy in Vision-and-Language Navigation with Bayes' Rule},
  author={Shuhei Kurita and Kyunghyun Cho},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2021}
}
```


*Note:* This repository is forked from the [speaker-follower model](https://github.com/ronghanghu/speaker_follower). This repository is built upon the [Matterport3DSimulator](https://github.com/peteanderson80/Matterport3DSimulator) codebase. Additional details on the Matterport3D Simulator can be found in [`README_Matterport3DSimulator.md`](README_Matterport3DSimulator.md).

## Train and evaluate on the Room-to-Room (R2R) dataset

*Note: we are still cleaning up this repository. We will release the snapshots of the generative & discriminative models soon.*

### Download and preprocess the data

We follow the setup of R2R dataset in the [speaker-follower model](https://github.com/ronghanghu/speaker_follower).
See also their repository for the preparation details. We write down the keypoints here.

1. Repository checkout & Matterport build
```
git clone --recursive https://github.com/shuheikurita/glgp.git
cd glgp
mkdir build && cd build
cmake ..
make
cd ..
```

2. Download the Precomputing ResNet Image Features, and extract them into `img_features/`:
```
mkdir -p img_features/
cd img_features/
wget https://www.dropbox.com/s/o57kxh2mn5rkx4o/ResNet-152-imagenet.zip?dl=1 -O ResNet-152-imagenet.zip
unzip ResNet-152-imagenet.zip
cd ..
```
(Please see the latest download links [here](https://github.com/peteanderson80/Matterport3DSimulator#precomputing-resnet-image-features) when the URL above doesn't work.)

3. Download the R2R dataset and the speaker-follower model's sampled trajectories for data augmentation:
```
./tasks/R2R/data/download.sh
```

### Training

1. Train the speaker model following the speaker-follower model:  
```
python tasks/R2R/train_speaker.py
```
Alternatively, you can download and use the pretrained speaker model in [speaker-follower model](https://github.com/ronghanghu/speaker_follower).

2. Train the generative and discriminative policies

To train the generative policy:
```
python tasks/R2R/train_planner.py \
    --model_name=MODEL_SAVE_NAME \
    --feedback_method=teacher+sample \
    --loss_type=speaker \
    --wo_validation_loss \
    --use_pretraining --pretrain_splits train literal_speaker_data_augmentation_paths \
    --minibatch 25 --log_every 1000 \
    --delta 0.333333333 \
    --speaker_prefix PRETRAINED_SPEAKER
```

`--speaker_prefix PRETRAINED_SPEAKER` is the path to the pretrained "spekaer" model of the speaker-foolower model, e.g., "tasks/R2R/speaker/snapshots/speaker_teacher_imagenet_mean_pooled_train_iter_18500_val_seen-bleu=29.070".

Similarly, you can train the discriminative policy, which is the same model with the "follower" of [speaker-follower model](https://github.com/ronghanghu/speaker_follower).
```
python tasks/R2R/train_planner.py  \
    --model_name=MODEL_SAVE_NAME \
    --feedback_method=teacher+sample \
    --loss_type=follower \
    --wo_validation_loss \
    --use_pretraining --pretrain_splits train literal_speaker_data_augmentation_paths \
    --minibatch 25 --log_every 1000 \
    --delta 0.333333333
```

Thanks to the combination of the mixture of the teacher-forcing and student-forcing, this discriminative policy performs better than the original "follower".
Other useful arguments:
- `--wo_validation_loss` : skip validation loss computation.
- `--wo_eval_until 20000` : skip evaluation until 20000 training instances.
- `--delta 0.333333333` : the ratio of the teacher-forcing and student-forcing.
- `--delta_linear 50000` : linearly decrease the delta. This may slightly help training in some cases.


3. Evaluate the generative policy, discriminative policy, and the combination of both policies.

Generative policy evaluation
```
python -u tasks/R2R/train_planner.py \
    --model_name=gen \
    --feedback_method=teacher \
    --loss_type=follower+speaker \
    --no_validation_loss  \
    --wo_train --no_save --n_iters 1 \
    --speaker_prefix GENERATIVE_MODEL_PATH \
```

Discriminative policy evaluation
```
python -u tasks/R2R/train_planner.py \
    --model_name=disc \
    --feedback_method=teacher \
    --loss_type=follower+speaker \
    --no_validation_loss  \
    --wo_train --no_save --n_iters 1 \
    --follower_prefix DISCRIMINATIVE_MODEL_PATH \
```

Combination of the generative & discriminative policies evaluation
```
python -u tasks/R2R/train_planner.py \
    --model_name=gen+disc \
    --feedback_method=teacher \
    --loss_type=follower+speaker  \
    --no_validation_loss  \
    --speaker_model_name none \
    --wo_train --no_save --n_iters 1 \
    --speaker_prefix  GENERATIVE_MODEL_PATH \
    --follower_prefix DISCRIMINATIVE_MODEL_PATH \
    --beta 0.5
```

Combination of the generative & discriminative policies evaluation + FAST-style back-tracking
```
python -u tasks/R2R/train_planner.py \
    --model_name=gen+disc+fast \
    --feedback_method=teacher \
    --loss_type=follower+speaker \
    --no_validation_loss  \
    --speaker_model_name none \
    --wo_train --no_save --n_iters 1 \
    --speaker_prefix  GENERATIVE_MODEL_PATH \
    --follower_prefix DISCRIMINATIVE_MODEL_PATH \
    --beta 0.5
```

## Acknowledgements

This repository is built upon the [speaker-follower model](https://github.com/ronghanghu/speaker_follower) codebase.
This repository is also built upon the [Matterport3DSimulator](https://github.com/peteanderson80/Matterport3DSimulator) codebase.
