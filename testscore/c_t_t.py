import argparse
import gc
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Collection

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from timm.utils import AverageMeter
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from config import get_config
from dataset_yuv import DisTypeVideoDataset, VideoDataset
# from database.VQA.dataset_yuv import VideoDataset
from lr_scheduler import build_scheduler
from build import build_model
from optimizer import build_optimizer
from utils import (auto_resume_helper, get_grad_norm, load_checkpoint,
                   reduce_tensor, save_checkpoint)

# from models.build_tsm import build_model
# from models.build_cnn_swin_tsm import build_model
# from models.build_cnn_swin import build_model
# from models.build_ape import build_model

best = 0

DAY = "{0:%Y-%m-%d}".format(datetime.now())
TIMESTAMP = "{0:%H-%M}".format(datetime.now())

writer = SummaryWriter()


def getlogger(path=None, level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(level)
    rq = time.strftime(TIMESTAMP)
    log_path = './CrossLogs'
    log_path = os.path.join(log_path, path, DAY)
    if not os.path.exists(log_path):
        print("log_path don't existed...")
        os.makedirs(log_path)
    log_name = os.path.join(log_path, rq + '.log')
    logfile = log_name

    print("log file: {}".format(logfile))

    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def getLIVEVQA(cfg, transform_train, transofrm_test, batch_train=1, batch_test=1, train_percent=0.8, idx=-1, train=True):
    subj_dataset = './VQA/LIVE_VQA/LIVE_subj_score_TEST.json'
    # video_path = '/data/wst/DATABASE/LIVE/live/LIVE_VIDEO/video/'
    video_path = '/data/wst/DATABASE/LIVE/live/LIVE_VIDEO/video/'
    batch = {'train': batch_train, 'test': batch_test}

    if train:
        summary_dir = 'runs/swin_LIVE_VQA/' + DAY
        writer = SummaryWriter(summary_dir)
        check_path = './checkpoints/' + 'LIVE_VQA/'
        logger = getlogger(path='LIVE_VQA')
        logger.info(f"subj_dataset = {subj_dataset}")

    # channel = 3
    # size_x = 432
    # size_y = 768
    # stride_x = 1
    # stride_y = 1

    channel = 1
    size_x = 224
    size_y = 224
    stride_x = 224
    stride_y = 224

    video_dataset = {x: VideoDataset(subj_dataset, video_path, x, channel, size_x, size_y, stride_x,
                                     stride_y, frameWant=cfg.VQA.FRAMEWANT, transform=transform_train) for x in ['test']}
    dataloaders = {x: DataLoader(video_dataset[x], batch_size=batch[x], shuffle=True,
                                 num_workers=cfg.DATA.NUM_WORKERS, drop_last=False) for x in ['test']}

    if train:
        return writer, check_path, dataloaders['test'], logger
    return dataloaders['test']


def getCSIQVQA(cfg, transform_train, transofrm_test, batch_train=1, batch_test=1, train_percent=0.8, idx=-1, train=True):
    # subj_dataset = './database/VQA/CSIQ/csiq_subj_score_BQTerrace_ParkScene.json'
    subj_dataset = "./VQA/CSIQ/CSIQ_subj_score_TEST.json"

    video_path = '/data/wst/DATABASE/CSIQ_VQA/videos/'
    batch = {'train': batch_train, 'test': batch_test}

    if train:
        summary_dir = 'runs/swin_CSIQ_VQA/' + DAY
        writer = SummaryWriter(summary_dir)
        check_path = './checkpoints/' + 'CSIQ_VQA/'
        logger = getlogger(path='CSIQ_VQA')
        logger.info(f"subj_dataset = {subj_dataset}")

    channel = 1
    size_x = 224
    size_y = 224
    stride_x = 224
    stride_y = 224

    video_dataset = {x: VideoDataset(subj_dataset, video_path, x, channel, size_x, size_y, stride_x,
                                     stride_y, frameWant=cfg.VQA.FRAMEWANT, transform=transform_train) for x in ['test']}
    dataloaders = {x: DataLoader(video_dataset[x], batch_size=batch[x], shuffle=True,
                                 num_workers=cfg.DATA.NUM_WORKERS, drop_last=False) for x in ['test']}

    if train:
        return writer, check_path, dataloaders['test'], logger
    return dataloaders['test']


def getVQCVQA(cfg, transform_train, transofrm_test, batch_train=1, batch_test=1, train_percent=0.8, idx=-1, train=True):
    video_path = '/data/wst/DATABASE/LIVE-VQC/data/Video/'

    subj_dataset = "./VQA/LIVE_VQC/VQC_subj_score_TEST.json"

    if train:
        summary_dir = 'runs/LIVE-VQC/' + DAY
        writer = SummaryWriter(summary_dir)
        check_path = './checkpoints/' + 'LIVE-VQC/'
        logger = getlogger(path='LIVE-VQC')
        logger.info(f"subj_dataset = {subj_dataset}")

    channel = 1
    size_x = 224
    size_y = 224
    # stride_x = 448
    # stride_x = 336
    # stride_y = 336

    stride_x = 224
    stride_y = 224

    # size_x = 960
    # size_y = 540
    # stride_x = 0
    # stride_y = 0
    batch = {'train': batch_train, 'test': batch_test}

    if train:
        video_dataset = {x: VideoDataset(subj_dataset, video_path, x, channel, size_x, size_y, stride_x,
                                         stride_y, frameWant=cfg.VQA.FRAMEWANT, transform=transform_train, crossMode=True) for x in ['test']}
    else:
        video_dataset = {x: VideoDataset(subj_dataset, video_path, x, channel, size_x, size_y, stride_x,
                                         stride_y, frameWant=cfg.VQA.FRAMEWANT, transform=transform_train) for x in ['test']}
    dataloaders = {x: DataLoader(video_dataset[x], batch_size=batch[x], shuffle=True,
                                 num_workers=cfg.DATA.NUM_WORKERS, drop_last=False) for x in ['test']}

    if train:
        return writer, check_path, dataloaders['test'], logger
    return dataloaders['test']


def getKonVQA(cfg, transform_train, transofrm_test, batch_train=1, batch_test=1, train_percent=0.8, idx=-1, train=True):

    video_path = '/data/wst/DATABASE/KON/KoNViD-1k/KoNViD_1k_videos/'
    # subj_dataset = './database/VQA/konIQ/kon_subj_score.json'
    subj_dataset = "./VQA/KON_VQA/kon_subj_score_TEST_mp4.json"

    if train:
        summary_dir = 'runs/swin_Kon_VQA/' + DAY
        writer = SummaryWriter(summary_dir)
        check_path = './checkpoints/' + 'Kon_VQA/'
        logger = getlogger(path='Kon_VQA')
        logger.info(f"subj_dataset = {subj_dataset}")

    channel = 1
    size_x = 224
    size_y = 224

    stride_x = 224
    stride_y = 224

    # size_x = 960
    # size_y = 540
    # stride_x = 0
    # stride_y = 0
    batch = {'train': batch_train, 'test': batch_test}
    video_dataset = {x: VideoDataset(subj_dataset, video_path, x, channel, size_x, size_y, stride_x,
                                     stride_y, frameWant=cfg.VQA.FRAMEWANT, transform=transform_train) for x in ['test']}
    dataloaders = {x: DataLoader(video_dataset[x], batch_size=batch[x], shuffle=True,
                                 num_workers=cfg.DATA.NUM_WORKERS, drop_last=False) for x in ['test']}

    if train:
        return writer, check_path, dataloaders['test'], logger
    return dataloaders['test']


def get_transforms(img_size1, img_size2):

    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(img_size1),
        # transforms.RandomHorizontalFlip(p=0.25),
        transforms.RandomCrop((img_size2, img_size2)),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        # transforms.RandomCrop((img_size, img_size)),
        # transforms.Resize((img_size, img_size)),
        # transforms.ToTensor(),
        # transforms.FiveCrop((img_size, img_size)),
        # transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(img_size1),
        # transforms.RandomCrop(img_size),
        # transforms.CenterCrop((img_size, img_size)),
        # transforms.Resize((img_size, img_size)),
        transforms.RandomCrop((img_size2, img_size2)),
        transforms.ToTensor(),
        # transforms.FiveCrop((img_size, img_size)),
        # transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transform_train, transform_test


def trainLoop(config, train_set, model, device, criterion, optimizer, logger, writer, epoch, lr_scheduler):
    model.train()
    optimizer.zero_grad()
    num_steps = len(train_set)

    epoch_pred, epoch_label = [], []
    totalLoss = 0

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    start = time.time()
    end = time.time()

    for idx, (img, label) in enumerate(tqdm(train_set)):
        # img = img.view(-1, config.MODEL.SWIN.IN_CHANS, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE))
        # B N C D H W
        # img = img.squeeze(1)

        B, N, C, D, H, W = img.size()
        # img = img.view(B, N * C, D, H, W)

        clip = torch.randint(10000000, (1,)) % N
        img = img[:, clip, :, :, :, :].squeeze(1)
        # img = img.squeeze(1)
        # B, C, D, H, W = img.size()

        # B C D H W
        # TODO : FIVE CROP
        # B, C, D, H, W = img.shape
        # img = img.view(B * C, 1, D, H, W)

        img = img.to(device)
        label = label.view(-1, 1).type(torch.float32).to(device)

        pred = model(img)
        # TODO : FIVE CROP
        # pred = torch.mean(pred.view(B, C), dim=1, keepdims=True)
        if config.TRAIN.LOSS != 'mix':
            loss = criterion(pred, label)
        else:
            loss = config.TRAIN.lambda1 * \
                criterion[0](pred, label) + config.TRAIN.lambda2 * \
                criterion[1](pred, label)

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            loss.backward()
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(model.parameters())
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(model.parameters())
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        totalLoss += loss.item()
        epoch_pred.extend(pred.detach().cpu().numpy().reshape(-1))
        epoch_label.extend(label.detach().cpu().numpy().reshape(-1))

        loss_meter.update(loss.item(), label.size(0))
        batch_time.update(time.time() - end)
        norm_meter.update(grad_norm)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

    SROCC = spearmanr(epoch_pred, epoch_label)[0]
    SROCC = abs(SROCC)
    PLCC = pearsonr(epoch_pred, epoch_label)[0]
    PLCC = abs(PLCC)

    epoch_time = time.time() - start

    logger.info(
        f"epoch : {epoch}\n \t\ttrain SROCC : {SROCC:.4f}\n \t\tPLCC : {PLCC:.4f}\n \t\tepoch Loss : {totalLoss:.4f}\n \t\tepoch_time {epoch_time:.4f}\n")

    writer.add_scalar('Loss/Train', totalLoss, epoch)
    writer.add_scalar('SROCC/Train', SROCC, epoch)
    writer.add_scalar('PLCC/Train', PLCC, epoch)


def testLoop(config, test_set, model, device, criterion, logger, writer, epoch, checkpath, dataset):

    model.eval()
    epoch_pred, epoch_label = [], []
    testLoss = 0
    with torch.no_grad():
        for (img, label) in tqdm(test_set):
            # img, label = data["img"], data["label"]
            # img = img.view(-1, 3, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)
            # img = img.squeeze(1)

            B, N, C, D, H, W = img.size()
            # img = img.view(B, N * C, D, H, W)
            img = img.contiguous().view(B * N, C, D, H, W)
            # B, C, D, H, W = img.size()

            # TODO : FIVE CROP
            # B, C, D, H, W = img.shape
            # img = img.view(B * C, 1, D, H, W)

            img = img.to(device)
            label = label.view(-1, 1).to(device)
            pred = model(img)
            pred = torch.mean(pred.view(B, N), dim=1, keepdims=True)

            # TODO : FIVE CROP
            # pred = torch.mean(pred.view(B, C), dim=1, keepdims=True)

            # testLoss += criterion(pred, label)
            # if config.TRAIN.LOSS != 'mix':
            #     testLoss += criterion(pred, label)
            # else:
            #     testLoss += config.TRAIN.lambda1 * criterion[0](pred, label) + config.TRAIN.lambda2 * criterion[1](pred, label)

            epoch_label.extend(label.cpu().numpy().reshape(-1))
            epoch_pred.extend(pred.cpu().numpy().reshape(-1))

        SROCC = spearmanr(epoch_label, epoch_pred)[0]
        SROCC = abs(SROCC)
        PLCC = pearsonr(epoch_pred, epoch_label)[0]
        PLCC = abs(PLCC)

        global best
        if SROCC > best:
            best = SROCC

        print(
            f"epoch : {epoch}\t Best SROCC : {best:.4f}, SROCC : {SROCC:.4f}, PLCC : {PLCC:.4f}, Test Loss : {testLoss:.4f}")
        logger.info(
            f"epoch: {epoch}\t Best SROCC :{best:.4f}\n \t\ttest : SROCC: {SROCC:.4f}\n \t\tPLCC: {PLCC:.4f}\n \t\ttestLoss : {testLoss:.4f}\n")

        writer.add_scalar('Loss/Test', testLoss, epoch)
        writer.add_scalar('SROCC/Test', SROCC, epoch)
        writer.add_scalar('PLCC/Test', PLCC, epoch)

        if SROCC >= best and SROCC >= config.TEST.BESTSROCC:
            best = SROCC
            if not os.path.exists(os.path.join(checkpath, DAY)):
                print(f"checkpoints path : {os.path.join(checkpath, DAY)}")
                os.makedirs(os.path.join(checkpath, DAY))
            torch.save(model.state_dict(), os.path.join(
                checkpath, DAY, TIMESTAMP + '_' + str(best) + '.pth'))
            print(f"save checkpoints, Best SROCC is : {best}")


# def load_pre_trained(config, path):
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     # model = SwinTransformer()
#     model = build_model(config.MODEL.TYPE)
#     # new model dict
#     model_dict = model.state_dict()
#     # load pre trained model
#     pretrained_model = torch.load(path, device)['model']
#     pretrained_model.pop('head.weight')
#     pretrained_model.pop('head.bias')
#     # get the same weight
#     # pretrained_dict = {k: v for k, v in pretrained_model.items() if k in model_dict}
#     # TODO VB_SWIN
#     pretrained_dict = {k: v for k, v in pretrained_model.items() if 'backbone.' + k in model_dict}
#     # overwrite the new model dict
#     model_dict.update(pretrained_dict)
#     # update the dict to new model
#     print(f"length of pretrained dict : {len(model_dict)}")
#     model.load_state_dict(model_dict, strict=False)
#     model.to(device)
#     return model, device
def load_pre_trained(logger, config, path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = SwinTransformer()
    model = build_model(config.MODEL.TYPE)

    # new model dict
    model_dict = model.state_dict()
    # load pre trained model
    if config.MODEL.TYPE != 'pre_train':
        pre_dict = torch.load(path, device)
        pretrained_dict = {k: v for k,
                           v in pre_dict.items() if k in model_dict}
        logger.info(
            f"Model Type : {config.MODEL.TYPE}\t Length Of Pre trained model : {len(pretrained_dict)}")
        model_dict.update(pretrained_dict)
    else:
        pretrained_model = torch.load(path, device)['model']
        if 'head.weight' in pretrained_model:
            pretrained_model.pop('head.weight')
        if 'head.bias' in pretrained_model:
            pretrained_model.pop('head.bias')
        # get the same weight
        # pretrained_dict = {k: v for k, v in pretrained_model.items() if k in model_dict}
        # TODO VB_SWIN
        pretrained_dict = {
            k: v for k, v in pretrained_model.items() if 'backbone.' + k in model_dict}
        # overwrite the new model dict
        model_dict.update(pretrained_dict)
        # update the dict to new model
        logger.info(
            f"Model Type : {config.MODEL.TYPE}\t Length Of Pre trained model : {len(pretrained_dict)}")
        print(f"length of pretrained dict : {len(pretrained_dict)}")

    model.load_state_dict(model_dict, strict=False)
    model.to(device)
    return model, device


def load_model(logger):
    logger.info("Just load model, WithOut Pre-Trained Weight")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(config.MODEL.TYPE)
    model.to(device)
    return model, device


def main(config, idx):

    dataset = config.DATA.DATASET
    dataset_test = config.DATA.DATASET_TEST

    epoch = config.TRAIN.EPOCHS
    batch_train = config.DATA.BATCH_SIZE
    batch_test = config.DATA.BATCH_TEST

    # set model
    # TODO: loss function
    if config.TRAIN.LOSS == 'mse':
        criterion = nn.MSELoss()
    elif config.TRAIN.LOSS == 'plcc':
        criterion = PlccLoss()
    elif config.TRAIN.LOSS == 'mix':
        # criterion = config.TRAIN.lambda1 * nn.MSELoss() + config.TRAIN.lambda2 * PlccLoss()
        criterion = [nn.MSELoss(), PlccLoss()]

    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    img_size = config.DATA.IMG_SIZE

    # transform_train, transform_test = get_transforms(img_size1=540, img_size2=config.DATA.IMG_SIZE)
    # transform_train, transform_test = get_transforms(img_size1=540, img_size2=512)
    transform_train, transform_test = None, None

    if dataset == 'LIVE_IQA':
        writer, check_path, train_set, logger = \
            getLIVE(transform_train, transform_test,
                    batch_train=batch_train, batch_test=batch_train)
    elif dataset == 'KON_IQA':
        writer, check_path, train_set, logger = \
            getKonIQ(transform_train, transform_test, batch_train=batch_train,
                     batch_test=batch_train, train_percent=config.TRAIN.PERCENT)
    elif dataset == 'LIVE_VQA':
        writer, check_path, train_set, logger = \
            getLIVEVQA(config, transform_train, transform_test, batch_train=batch_train,
                       batch_test=batch_train, train_percent=config.TRAIN.PERCENT, idx=idx)
    elif dataset == 'KON_VQA':
        writer, check_path, train_set, logger = \
            getKonVQA(config, transform_train, transform_test, batch_train=batch_train,
                      batch_test=batch_train, train_percent=config.TRAIN.PERCENT, idx=idx)
    elif dataset == 'CSIQ_VQA':
        writer, check_path, train_set, logger = \
            getCSIQVQA(config, transform_train, transform_test, batch_train=batch_train,
                       batch_test=batch_train, train_percent=config.TRAIN.PERCENT, idx=idx)
    elif dataset == 'CVD_VQA':
        writer, check_path, train_set, logger = \
            getCVDVQA(config, transform_train, transform_test, batch_train=batch_train,
                      batch_test=batch_train, train_percent=config.TRAIN.PERCENT, idx=idx)
    elif dataset == 'LIVEM_VQA':
        writer, check_path, train_set, logger = \
            getLIVEMVQA(config, transform_train, transform_test, batch_train=batch_train,
                        batch_test=batch_train, train_percent=config.TRAIN.PERCENT, idx=idx)
    elif dataset == 'VQC_VQA':
        writer, check_path, train_set, logger = \
            getVQCVQA(config, transform_train, transform_test, batch_train=batch_train,
                      batch_test=batch_train, train_percent=config.TRAIN.PERCENT, idx=idx)

    if dataset_test == 'LIVE_IQA':
        test_set = \
            getLIVE(transform_train, transform_test,
                    batch_train=batch_test, batch_test=batch_test)
    elif dataset_test == 'KON_IQA':
        test_set = \
            getKonIQ(transform_train, transform_test, batch_train=batch_test,
                     batch_test=batch_test, train_percent=config.TRAIN.PERCENT)
    elif dataset_test == 'LIVE_VQA':
        test_set = \
            getLIVEVQA(config, transform_train, transform_test, batch_train=batch_test,
                       batch_test=batch_test, train_percent=config.TRAIN.PERCENT, idx=idx, train=False)
    elif dataset_test == 'KON_VQA':
        test_set = \
            getKonVQA(config, transform_train, transform_test, batch_train=batch_test,
                      batch_test=batch_test, train_percent=config.TRAIN.PERCENT, idx=idx, train=False)
    elif dataset_test == 'CSIQ_VQA':
        test_set = \
            getCSIQVQA(config, transform_train, transform_test, batch_train=batch_test,
                       batch_test=batch_test, train_percent=config.TRAIN.PERCENT, idx=idx, train=False)
    elif dataset_test == 'VQC_VQA':
        test_set = \
            getVQCVQA(config, transform_train, transform_test, batch_train=batch_test,
                      batch_test=batch_test, train_percent=config.TRAIN.PERCENT, idx=idx, train=False)

    prefix = os.path.abspath('.')
    # pre_trained_path = os.path.join(prefix, 'pre_trained/swin_tiny_patch4_window7_224.pth')
    pre_trained_path = os.path.join(prefix, config.TRAIN.PRE_TRAINED)

    if config.FINE_TUNE:
        model, device = load_pre_trained(logger, config, pre_trained_path)
    else:
        model, device = load_model(logger)

    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)

    logger.info(f"number of params: {n_parameters}")

    optimizer = build_optimizer(config, model)

    lr_scheduler = build_scheduler(config, optimizer, len(train_set))

    logger.info(f"PRE TRAINED PATH : {config.TRAIN.PRE_TRAINED}")
    # print config
    logger.info(model)

    logger.info(config.dump())

    global best
    best = 0
    for i in range(epoch):
        trainLoop(config, train_set, model, device, criterion, optimizer,
                  logger, writer, epoch=i, lr_scheduler=lr_scheduler)
        testLoop(config, test_set, model, device, criterion, logger,
                 writer, epoch=i, checkpath=check_path, dataset=dataset)
        gc.collect()

    writer.close()
    print("Done!")


def plcc(pred, target):
    n = len(pred)
    if n != len(target):
        raise ValueError('input and target must have the same length')
    # if n < 2:
    #     raise ValueError('input length must greater than 2')
    #     return

    xmean = torch.mean(pred)
    ymean = torch.mean(target)
    xm = pred - xmean
    ym = target - ymean
    bias = 1e-8
    normxm = torch.norm(xm, p=2) + bias
    normym = torch.norm(ym, p=2) + bias

    r = torch.dot(xm/normxm, ym/normym)
    return r


class PlccLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # print(pred.shape, target.shape)
        val = 1.0 - plcc(pred.view(-1).float(), target.view(-1).float())
        return torch.log(val)
        # return 1.0 - plcc(pred.view(-1).float(), target.view(-1).float())


def parse_option():
    parser = argparse.ArgumentParser(
        'Self-Supervised Representation Learning for Video Quality Assessment training and evaluation script', add_help=False)

    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int,
                        help="batch size for single GPU")
    parser.add_argument('--batch-test', type=int, default=24,
                        help="batch test size for single GPU")
    parser.add_argument('--accumulation-steps', type=int,
                        help="gradient accumulation steps")
    parser.add_argument('--dataset', type=str,
                        default='LIVE_VQA', help='dataset')
    parser.add_argument('--dataset_test', type=str,
                        default='CSIQ_VQA', help='dataset')
    parser.add_argument('--model', type=str,
                        default='vb_cnn_transformerv2', help='Model Type')
    parser.add_argument('--frame', type=int, default='100',
                        help='Frame Per Video')
    parser.add_argument('--base_lr', type=float,
                        default='5e-5', help='Base Learning Rate')
    parser.add_argument('--loss', type=str, default='plcc',
                        help='Loss Function')
    parser.add_argument('--best', type=float, default='0.78',
                        help='Best SROCC For Save Checkpoints')
    parser.add_argument('--epoch', type=int,
                        default='100', help='Epoch Number')
    parser.add_argument('--warm_up_epochs', type=int,
                        default='5', help='Warm Up Epoch Number')

    # parser.add_argument('--pre_trained_path', type=str, default="/data/wst/video-pace/pretrain_cks/sr_4_mymodelv4_lr_0.001_len_16_sz_112_mytrain3/2023-10-26/mlpepoch-18.pth.tar", help='1017，去掉颜色抖动，修改对比学习部分')
    # parser.add_argument('--pre_trained_path', type=str, default="/data/wst/video-pace/testscore/pre/1021/mlpepoch-18.pth.tar", help='1017，去掉颜色抖动')
    # parser.add_argument('--pre_trained_path', type=str, default="/data/wst/video-pace/pretrain_cks/sr_4_mymodelv4_lr_0.001_len_16_sz_112_mytrain3/2023-11-11/mlpepoch-18.pth.tar", help='1111，去掉颜色抖动，修改对比学习部分,有tmoconv,128')
    # parser.add_argument('--pre_trained_path', type=str, default="/data/wst/video-pace/testscore/pre/1114/mlpepoch-18.pth.tar", help='1114，去掉颜色抖动，修改对比学习部分,FPN64')
    # parser.add_argument('--pre_trained_path', type=str, default="/data/wst/video-pace/testscore/pre/1130/mlpepoch-19.pth.tar", help='失真类型')
    parser.add_argument('--pre_trained_path', type=str, default="/data/wst/video-pace/testscore/pre/1201/mlpepoch-18.pth.tar", help='1128，FPN64,0.9 0.9 0.6')


    parser.add_argument('--fine_tune', type=bool, help='Fine Tune Or Not')
    parser.add_argument('--five', type=bool,
                        help='FIVE Test Calc Mean-Std Result.')
    parser.add_argument('--idx', type=int, default=-1, help='Index')

    parser.add_argument('--height', type=int, default=256, help='resize height')
    parser.add_argument('--width', type=int, default=256, help='resize width')
    parser.add_argument('--crop_sz', type=int, default=224, help='crop size')
    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


if __name__ == '__main__':

    _, config = parse_option()

    seed = config.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)

    # linear scale the learning rate according to total batch size, may not be optimal
    # linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE
    # linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE
    # linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE
    linear_scaled_lr = config.TRAIN.BASE_LR
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR
    linear_scaled_min_lr = config.TRAIN.MIN_LR

    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        print("USING ACCUMULATION_STEPS")
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    print("base lr :", linear_scaled_lr)
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    print("warmup lr :", linear_scaled_warmup_lr)
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    print("min lr :", linear_scaled_min_lr)
    config.freeze()

    if config.FIVE:
        for idx in range(5):
            main(config, idx)
    else:
        main(config, config.IDX)
