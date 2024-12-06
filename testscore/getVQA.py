import logging
import os
import time
import torch

from datetime import datetime, timedelta
from torch.utils.tensorboard import SummaryWriter
from dataset_yuv import VideoDataset, DisTypeVideoDataset
from torch.utils.data import DataLoader
from torchvision import transforms

DAY = "{0:%Y-%m-%d}".format(datetime.now())
TIMESTAMP = "{0:%H-%M}".format(datetime.now())

writer = SummaryWriter()


def getlogger(path=None, level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(level)
    rq = time.strftime(TIMESTAMP)
    log_path = './Logss'
    log_path = os.path.join(log_path, path, DAY)
    if not os.path.exists(log_path):
        print("log_path don't existed...")
        os.makedirs(log_path)
    log_name = os.path.join(log_path, rq + '.log')
    logfile = log_name

    print("log file: {}".format(logfile))

    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def getLIVEVQA(cfg, transform_train, transofrm_test, batch_train=1, batch_test=1, train_percent=0.8, idx=-1):
    summary_dir = 'runs/swin_LIVE_VQA/' + DAY
    writer = SummaryWriter(summary_dir)
    check_path = './checkpoints/' + 'LIVE_VQA/'
    logger = getlogger(path='LIVE_VQA')

    channel = 1
    size_x = 224
    size_y = 224
    stride_x = 224
    stride_y = 224

    subj_dataset = './database/VQA/LIVE/live_subj_score_nr_ref.json'
    # video_path = '/home1/server823-2/database/2D-Video/live/videos'
    video_path = '/data/wst/DATABASE/LIVE/live/LIVE_VIDEO/video/'
    batch = {'train': batch_train, 'test': batch_test}

    if cfg.IDX != -1:
        subj_dataset = './VQA/LIVE_VQA/LIVE_LEAVE2_1_LAST.json'

        # subj_dataset = './VQA/LIVE_VQA/LIVE_LEAVE2_' + str(cfg.IDX) + '_LAST.json'
        # subj_dataset = "./VQA/LIVE/test_live_IP.json"
        if cfg.IDX == 100:
            subj_dataset = './database/VQA/LIVE/LIVE_subj_score_TEST.json'
            video_dataset = {
                x: DisTypeVideoDataset(subj_dataset, video_path, x, channel, size_x, size_y, stride_x, stride_y,
                                       frameWant=cfg.VQA.FRAMEWANT, transform=transform_train) for x in ['test']}
            dataloaders = {
                x: DataLoader(video_dataset[x], batch_size=batch[x], shuffle=True, num_workers=cfg.DATA.NUM_WORKERS,
                              drop_last=False) for x in ['test']}
            return writer, check_path, None, dataloaders['test'], logger

    logger.info(f"subj_dataset = {subj_dataset}")

    video_dataset = {x: DisTypeVideoDataset(subj_dataset, video_path, x, channel, size_x, size_y, stride_x, stride_y,
                                            frameWant=cfg.VQA.FRAMEWANT, transform=transform_train) for x in
                     ['train', 'test']}
    dataloaders = {x: DataLoader(video_dataset[x], batch_size=batch[x], shuffle=True, num_workers=cfg.DATA.NUM_WORKERS,
                                 drop_last=False) for x in ['train', 'test']}
    return writer, check_path, dataloaders['train'], dataloaders['test'], logger


def getCSIQVQA(cfg, transform_train, transofrm_test, batch_train=1, batch_test=1, train_percent=0.8, idx=-1):
    summary_dir = 'runs/swin_CSIQ_VQA/' + DAY
    writer = SummaryWriter(summary_dir)
    check_path = './checkpoints/' + 'CSIQ_VQA/'
    logger = getlogger(path='CSIQ_VQA')

    channel = 1
    size_x = 224
    size_y = 224
    stride_x = 224
    stride_y = 224

    subj_dataset = './database/VQA/CSIQ/csiq_subj_score_nr.json'
    video_path = '/data/wst/DATABASE/CSIQ_VQA/videos/'
    batch = {'train': batch_train, 'test': batch_test}

    if cfg.IDX != -1:
        # subj_dataset = './VQA/CSIQ/CSIQ_LEAVE2_' + str(cfg.IDX) + '_LAST.json'
        subj_dataset = "./VQA/CSIQ/test_csiq_MJPEG.json"
        if cfg.IDX == 100:
            subj_dataset = './database/VQA/CSIQ/CSIQ_subj_score_TEST.json'
            video_dataset = {
                x: DisTypeVideoDataset(subj_dataset, video_path, x, channel, size_x, size_y, stride_x, stride_y,
                                       frameWant=cfg.VQA.FRAMEWANT, transform=transform_train) for x in ['test']}
            dataloaders = {
                x: DataLoader(video_dataset[x], batch_size=batch[x], shuffle=True, num_workers=cfg.DATA.NUM_WORKERS,
                              drop_last=False) for x in ['test']}
            return writer, check_path, None, dataloaders['test'], logger

    logger.info(f"subj_dataset = {subj_dataset}")

    video_dataset = {x: DisTypeVideoDataset(subj_dataset, video_path, x, channel, size_x, size_y, stride_x, stride_y,
                                            frameWant=cfg.VQA.FRAMEWANT, transform=transform_train) for x in
                     ['train', 'test']}
    dataloaders = {x: DataLoader(video_dataset[x], batch_size=batch[x], shuffle=True, num_workers=cfg.DATA.NUM_WORKERS,
                                 drop_last=False) for x in ['train', 'test']}
    return writer, check_path, dataloaders['train'], dataloaders['test'], logger


def getVQCVQA(cfg, transform_train, transofrm_test, batch_train=1, batch_test=1, train_percent=0.8, idx=-1):
    summary_dir = 'runs/LIVE-VQC/' + DAY
    writer = SummaryWriter(summary_dir)
    check_path = './checkpoints/' + 'LIVE-VQC/'
    logger = getlogger(path='LIVE-VQC')
    # video_path = '/home1/server823-2/database/2D-Video/LIVE_Video_Quality_Challenge(VQC)_Database/video/'
    video_path = '/data/wst/DATABASE/LIVE-VQC/data/Video/'
    batch = {'train': batch_train, 'test': batch_test}
    channel = 1
    size_x = 224
    size_y = 224

    stride_x = 224
    stride_y = 224

    if cfg.IDX != -1:
        subj_dataset = './VQA/LIVE_VQC/VQC_subj_score_' + str(cfg.IDX) + '.json'
        # subj_dataset = './VQA/LIVE_VQC/VQC_subj_score_0.json'

        if cfg.IDX == 100:
            subj_dataset = './database/VQA/LIVE-VQC/VQC_subj_score_TEST.json'
            video_dataset = {x: VideoDataset(subj_dataset, video_path, x, channel, size_x, size_y, stride_x, stride_y,
                                             frameWant=cfg.VQA.FRAMEWANT, transform=transform_train) for x in ['test']}
            dataloaders = {
                x: DataLoader(video_dataset[x], batch_size=batch[x], shuffle=True, num_workers=cfg.DATA.NUM_WORKERS,
                              drop_last=False) for x in ['test']}
            return writer, check_path, None, dataloaders['test'], logger

    logger.info(f"subj_dataset = {subj_dataset}")

    video_dataset = {x: VideoDataset(subj_dataset, video_path, x, channel, size_x, size_y, stride_x, stride_y,
                                     frameWant=cfg.VQA.FRAMEWANT, transform=transform_train) for x in ['train', 'test']}
    dataloaders = {x: DataLoader(video_dataset[x], batch_size=batch[x], shuffle=True, num_workers=cfg.DATA.NUM_WORKERS,
                                 drop_last=False) for x in ['train', 'test']}
    return writer, check_path, dataloaders['train'], dataloaders['test'], logger


def getKonVQA(cfg, transform_train, transofrm_test, batch_train=1, batch_test=1, train_percent=0.8, idx=-1):
    summary_dir = 'runs/swin_Kon_VQA/' + DAY
    writer = SummaryWriter(summary_dir)
    check_path = './checkpoints/' + 'Kon_VQA/'
    logger = getlogger(path='Kon_VQA')

    ## YUV
    # video_path = '/home1/server823-2/database/2D-Video/kon_yuv'
    # subj_dataset = './database/VQA/konIQ/kon_subj_score_trn80.json'

    # MP4
    # video_path = '/home1/server823-2/database/2D-Video/KoNViD-1k/KoNViD_1k_videos'
    # subj_dataset = './database/VQA/konIQ/kon_subj_score_trn80_mp4.json'
    # video_path = '/data/wst/selfdataset/KON/KoNViD-1k/KoNViD_1k_videos/'
    video_path = '/data/wst/DATABASE/KON/KoNViD-1k/KoNViD_1k_videos/'

    subj_dataset = './database/VQA/konIQ/kon_subj_score_trn80_mp4.json'

    channel = 1
    size_x = 224
    size_y = 224

    stride_x = 224
    stride_y = 224

    batch = {'train': batch_train, 'test': batch_test}

    if cfg.IDX != -1:
        # subj_dataset = './VQA/KON_VQA/KON_hzy_' + str(cfg.IDX) + '_mp4.json'
        subj_dataset = "./VQA/KON_VQA/kon_subj_score_trn80_mp4_n.json"
        if cfg.IDX == 100:
            subj_dataset = './database/VQA/konIQ/kon_subj_score_TEST.json'
            # subj_dataset = './database/VQA/konIQ/kon_subj_score_TEST_mp4.json'
            video_dataset = {x: VideoDataset(subj_dataset, video_path, x, channel, size_x, size_y, stride_x, stride_y,
                                             frameWant=cfg.VQA.FRAMEWANT, transform=transform_train) for x in ['test']}
            dataloaders = {
                x: DataLoader(video_dataset[x], batch_size=batch[x], shuffle=True, num_workers=cfg.DATA.NUM_WORKERS,
                              drop_last=False) for x in ['test']}
            return writer, check_path, None, dataloaders['test'], logger

    logger.info(f"idx = {cfg.IDX}, subj_dataset = {subj_dataset}")

    video_dataset = {x: VideoDataset(subj_dataset, video_path, x, channel, size_x, size_y, stride_x, stride_y,
                                     frameWant=cfg.VQA.FRAMEWANT, transform=transform_train) for x in ['train', 'test']}
    dataloaders = {x: DataLoader(video_dataset[x], batch_size=batch[x], shuffle=True, num_workers=cfg.DATA.NUM_WORKERS,
                                 drop_last=False) for x in ['train', 'test']}
    return writer, check_path, dataloaders['train'], dataloaders['test'], logger
