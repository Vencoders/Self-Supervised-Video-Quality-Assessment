# 基于自监督学习与多尺度时空特征融合的视频质量评估(计算机系统应用，2024)

* [paper link](https://kns.cnki.net/kcms2/article/abstract?v=-xbefZa1CdsQuXWKP2LrnZzN1DeN4vVtFtZHkgiVfk-nDnzV4YDjge7kLTcIjvqQyrAZdqJRv_XEDAubYrPU5CemuejIp6eJ323i75mZxBkFTn81sqBMRZk05vbQvTlAyqTPBJadhs4GiGThTjdbCmb0EQOVyhdzxBhkBlR0cFLdg-9a8kFIlp9mGOYpwjem&uniplatform=NZKPT&language=CHS)

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

# Project information
This project has received funding from the National Natural Science Foundation of China(62002172, 62276139, U2001211).

# Author

Li Yu {li.yu@nuist.edu.cn}

Situo Wang {202212490261@nuist.edu.cn}

Yadang Chen {cyd4511632@126.com}

Pan Gao {Pan.Gao@nuaa.edu.cn}

Yubao Sun {sunyb@nuist.edu.cn}

