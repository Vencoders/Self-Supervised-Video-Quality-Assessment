# 基于自监督学习与多尺度时空特征融合的视频质量评估(计算机系统应用，2024)

* [paper link]()

# Requirements
- python = 3.8
- pytroch >= 1.3.0
- tensorboardX
- cv2
- scipy

# Usage

## Data preparation

YouTube-8M dataset
- 310 videos were selected from the YouTube-8M database, and five types of distortions were applied to them, including Gaussian blur, contrast, H.264 compression, motion blur, and Gaussian noise.

## Pre-train

`python mytrain3.py --gpu 0 --bs 4 --lr 0.001 --height 128 --width 171 --crop_sz 112 --clip_len 16`

## Fine-tuning and Evaluation
`TEST_*.sh`: train-test in *LIVE/CSIQ/KoNVid-1k/LIVE-VQC* Database. 
`TRAIN_LIVE_TEST_OTHER.sh`: train on LIVE Database, test on other Databases.

# Citation
If you find this work useful or use our code, please consider citing:

```
@InProceedings{Wang20,
  author       = "Jiangliu Wang and Jianbo Jiao and Yunhui Liu",
  title        = "Self-Supervised Video Representation Learning by Pace Prediction",
  booktitle    = "European Conference on Computer Vision",
  year         = "2020",
}
```

