import argparse
import logging
import os
import time
from datetime import datetime, timedelta
import gc
from typing import Collection
from tqdm import tqdm
import scipy.io as sio

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr, kendalltau
from timm.utils import AverageMeter

from config import get_config
# from database.VQA.dataset_yuv import VideoDataset
from lr_scheduler import build_scheduler
from optimizer import build_optimizer

from build import build_model
from getVQA import *
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors

bestSROCC = 0
bestPLCC = 0
bestRMSE = 1e9
bestKROCC = 0

finetune = False

totalTime = 0.0


def check_nan(epoch_pred):
    for idx, val in enumerate(epoch_pred):
        if np.isnan(val):
            epoch_pred[idx] = 0


def dump_data(idx, pred, dmos, dataset):
    import scipy.io as sio
    path = f'./Data/{dataset}/ResNet_{str(finetune)}'
    if not os.path.exists(path):
        os.makedirs(path)
    sio.savemat(os.path.join(
        path, f'{idx}-{TIMESTAMP}.mat'), {'pred': pred, 'dmos': dmos})


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

    for idx, (video, label, disType) in enumerate(tqdm(train_set)):
        B, N, C, D, H, W = video.size()

        clip = torch.randint(10000000, (1,)) % N
        video = video[:, clip, :, :, :, :].squeeze(1)
        video = video.to(device)

        label = label.view(-1, 1).type(torch.float32).to(device)

        pred = model(video)

        if config.TRAIN.LOSS != 'mix':
            loss = criterion(pred, label)
            # # # TODO 添加L2正则化项来防止过拟合
            # l2_regularization = 0
            # for param in model.parameters():
            #     l2_regularization += torch.norm(param, p=2)  # 计算参数的L2范数
            # # 加上正则化项
            # lambda_ = 0.5  # 正则化系数，控制正则化的强度
            # loss += lambda_ * l2_regularization
        else:
            loss = config.TRAIN.lambda1 * \
                criterion[0](pred, label) + config.TRAIN.lambda2 * \
                criterion[1](pred, label)

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            loss.backward()
            # 用于梯度裁剪
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

    check_nan(epoch_pred)

    SROCC = spearmanr(epoch_pred, epoch_label)[0]
    PLCC = pearsonr(epoch_pred, epoch_label)[0]
    epoch_time = time.time() - start

    logger.info(
        f"epoch : {epoch}\n \t\ttrain SROCC : {SROCC:.4f}\n \t\tPLCC : {PLCC:.4f}\n \t\tepoch Loss : {totalLoss:.4f}\n \t\tepoch_time {epoch_time:.4f}\n")

    writer.add_scalar('Loss/Train', totalLoss, epoch)
    writer.add_scalar('SROCC/Train', SROCC, epoch)
    writer.add_scalar('PLCC/Train', PLCC, epoch)





def testLoop(config, test_set, model, device, criterion, logger, writer, epoch, checkpath, dataset, disNum=4):
    features_list = []
    disType_list = []
    colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
              '#911eb4']

    cmap = mcolors.ListedColormap(colors)

    model.eval()

    epoch_pred, epoch_label = [], []
    disListLabel = [[] for _ in range(disNum + 1)]
    disListPred = [[] for _ in range(disNum + 1)]

    testLoss = 0
    beginTime = time.time()
    with torch.no_grad():
        for (img, label, disType) in tqdm(test_set):

            B, N, C, D, H, W = img.size() # B: 8 C: 1 D: 32 N: 3 W: 224 H: 224
            img = img.contiguous().view(B * N, C, D, H, W)
            img = img.to(device)

            label = label.view(-1, 1).to(device)
            pred = model(img)
            pred = torch.mean(pred.view(B, N), dim=1, keepdims=True)



            features_list.append(pred.cpu().numpy())
            disType_list.append(disType.cpu().numpy())




            if config.TRAIN.LOSS != 'mix':
                testLoss += criterion(pred, label)
            else:
                testLoss += config.TRAIN.lambda1 * \
                    criterion[0](pred, label) + config.TRAIN.lambda2 * \
                    criterion[1](pred, label)

            for i in range(B):
                disListLabel[disType[i]].append(
                    float(label[i][0].cpu().numpy()))
                disListPred[disType[i]].append(float(pred[i][0].cpu().numpy()))

            epoch_label.extend(label.cpu().numpy().reshape(-1))
            epoch_pred.extend(pred.cpu().numpy().reshape(-1))

        features = np.vstack(features_list)
        disTypelabels = np.hstack(disType_list)

        global totalTime
        totalTime += time.time() - beginTime

        check_nan(epoch_pred)

        SROCC = spearmanr(epoch_label, epoch_pred)[0]
        PLCC = pearsonr(epoch_pred, epoch_label)[0]   # TODO:注释PLCC
        dump_data(epoch, epoch_pred, epoch_label, dataset)
        RMSE = np.sqrt(
            ((np.array(epoch_pred) - np.array(epoch_label)) ** 2).mean())
        KROCC = kendalltau(epoch_pred, epoch_label)[0]

        disSROCC = [0] * (disNum + 1)
        disPLCC = [0] * (disNum + 1)

        for i in range(1, disNum + 1):
            check_nan(disListPred[i])
            disSROCC[i] = spearmanr(disListLabel[i], disListPred[i])[0]
            disPLCC[i] = pearsonr(disListLabel[i], disListPred[i])[0]  # TODO:注释PLCC

        global bestSROCC, bestPLCC, bestRMSE, bestKROCC
        bestSROCC = max(bestSROCC, SROCC)
        bestPLCC = max(bestPLCC, PLCC)  # TODO:注释PLCC
        bestKROCC = max(bestKROCC, KROCC)
        bestRMSE = min(bestRMSE, RMSE)

        # print(f"epoch : {epoch}\t Best SROCC : {bestSROCC:.4f} Best PLCC :{bestPLCC:.4f} SROCC : {SROCC:.4f}, PLCC : {PLCC:.4f}, Test Loss : {testLoss:.4f}\n")
        # logger.info(f"epoch: {epoch}\t Best SROCC :{bestSROCC:.4f} \tBest PLCC :{bestPLCC:.4f}\tBest KROCC :{bestKROCC:.4f}\tBest RMSE :{bestRMSE:.4f} \n \t\ttest : SROCC: {SROCC:.4f}\n \t\tPLCC: {PLCC:.4f}\n \t\tKROCC: {KROCC:.4f}\n \t\tRMSE: {RMSE:.4f}\n \t\ttestLoss : {testLoss:.4f}\n")
        print(f"epoch : {epoch}\t Best SROCC : {bestSROCC:.4f} Best PLCC :{bestPLCC:.4f} SROCC : {SROCC:.4f}, Test Loss : {testLoss:.4f}\n")
        logger.info(f"epoch: {epoch}\t Best SROCC :{bestSROCC:.4f} \tBest PLCC :{bestPLCC:.4f} \tBest KROCC :{bestKROCC:.4f}\tBest RMSE :{bestRMSE:.4f} \n \t\ttest : SROCC: {SROCC:.4f}\n \t\tKROCC: {KROCC:.4f}\n \t\tRMSE: {RMSE:.4f}\n \t\ttestLoss : {testLoss:.4f}\n")

        logger.info("dis type SROCC: ")
        for i in range(1, disNum + 1):
            logger.info(f"{disSROCC[i]} ")
        logger.info("\n")
        logger.info("dis type PLCC: ")
        for i in range(1, disNum + 1):
            logger.info(f"{disPLCC[i]} ")
        logger.info("\n")

        writer.add_scalar('Loss/Test', testLoss, epoch)
        writer.add_scalar('SROCC/Test', SROCC, epoch)
        writer.add_scalar('PLCC/Test', PLCC, epoch)  # TODO:注释PLCC

        if SROCC >= bestSROCC and SROCC >= config.TEST.BESTSROCC:
            bestSROCC = SROCC
            if not os.path.exists(os.path.join(checkpath, DAY)):
                print(f"checkpoints path : {os.path.join(checkpath, DAY)}")
                os.makedirs(os.path.join(checkpath, DAY))
            torch.save(model.state_dict(), os.path.join(
                checkpath, DAY, TIMESTAMP + '_' + str(bestSROCC) + '.pth'))
            print(f"save checkpoints, Best SROCC is : {bestSROCC}")

    # 2. 使用t-SNE降维
    tsne = TSNE(n_components=2, random_state=42)
    features_tsne = tsne.fit_transform(features)

    # 3. 可视化结果
    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])
    scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=disTypelabels, cmap=cmap, s=25)
    # plt.colorbar(scatter, ticks=range(10))
    # plt.title('t-SNE visualization of extracted features')
    plt.savefig('/data/wst/video-pace/testscore/result-tsne/tsne_visualization'+str(epoch)+'.png')
    plt.close()


def load_pre_trained(logger, config, path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(config.MODEL.TYPE)

    # new model dict
    model_dict = model.state_dict()
    # load pre trained model
    if config.MODEL.TYPE != 'pre_train':
        pre_dict = torch.load(path, device)
        # pre_dict = pre_dict['state_dict']
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

    epoch = config.TRAIN.EPOCHS
    batch_train = config.DATA.BATCH_SIZE
    batch_test = config.DATA.BATCH_TEST

    # loss function
    if config.TRAIN.LOSS == 'mse':
        criterion = nn.MSELoss()
    elif config.TRAIN.LOSS == 'l1':
        criterion = nn.L1Loss()
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

    if dataset == 'LIVE_VQA':
        writer, check_path, train_set, test_set, logger = \
            getLIVEVQA(config, transform_train, transform_test, batch_train=batch_train,
                       batch_test=batch_test, train_percent=config.TRAIN.PERCENT, idx=idx)
    elif dataset == 'KON_VQA':
        writer, check_path, train_set, test_set, logger = \
            getKonVQA(config, transform_train, transform_test, batch_train=batch_train,
                      batch_test=batch_test, train_percent=config.TRAIN.PERCENT, idx=idx)
    elif dataset == 'CSIQ_VQA':
        writer, check_path, train_set, test_set, logger = \
            getCSIQVQA(config, transform_train, transform_test, batch_train=batch_train,
                       batch_test=batch_test, train_percent=config.TRAIN.PERCENT, idx=idx)
    elif dataset == 'VQC_VQA':
        writer, check_path, train_set, test_set, logger = \
            getVQCVQA(config, transform_train, transform_test, batch_train=batch_train,
                      batch_test=batch_test, train_percent=config.TRAIN.PERCENT, idx=idx)

    prefix = os.path.abspath('.')
    pre_trained_path = os.path.join(prefix, config.TRAIN.PRE_TRAINED)

    # fine-tune=true
    if config.FINE_TUNE:
        model, device = load_pre_trained(logger, config, pre_trained_path)
        global finetune
        finetune = True
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

    global bestSROCC
    global bestPLCC
    bestSROCC = 0
    bestPLCC = 0

    disNum = 4
    if dataset == "CSIQ_VQA":
        disNum = 6

    for i in range(epoch):
        trainLoop(config, train_set, model, device, criterion, optimizer,
                  logger, writer, epoch=i, lr_scheduler=lr_scheduler)
        testLoop(config, test_set, model, device, criterion, logger, writer,
                 epoch=i, checkpath=check_path, dataset=dataset, disNum=disNum)
        gc.collect()

    logger.info(
        f"Transformer. Total Cost Time = {totalTime}, epoch = {epoch}, epoch cost time = {totalTime / epoch} seconds")

    writer.close()
    print("Done!")


def plcc(pred, target):
    n = len(pred)
    if n != len(target):
        raise ValueError('input and target must have the same length')
    if n < 2:
        raise ValueError('input length must greater than 2')

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
    # parser.add_argument('--model', type=str,
    #                     default='vb_cnn_transformer', help='Model Type')
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

    # parser.add_argument(
        # '--pre_trained_path', type=str, default="/data/wst/SelfSupervisedVQA/Self-Supervised Representation Learning for Video Quality Assessment/Self-Supervised Representation Learning for Video Quality Assessment/pretrained/pre_trained.pth", help='pretrained weight path'
    # )
    # 单分支测试
    # parser.add_argument('--pre_trained_path', type=str, default="/data/wst/video-pace/pretrain_cks/sr_4_mymodel2_lr_0.002_len_16_sz_112_1/2023-06-17/epoch-18.pth.tar", help='pretrained weight path danfenzhi ceshi')
    # 暹罗网络+lr0.002+epoch18+对比学习+播放速度
    # parser.add_argument('--pre_trained_path', type=str, default='/data/wst/video-pace/pretrain_cks/sr_4_mymodel2_lr_0.002_len_16_sz_112_1/2023-06-16/epoch-18.pth.tar', help='pretrained weight path shuangfenzhi ceshi')
    # 暹罗网络+lr0.003+epoch18+对比学习+播放速度
    # parser.add_argument('--pre_trained_path', type=str, default="/data/wst/video-pace/pretrain_cks/sr_4_mymodel2_lr_0.003_len_16_sz_112_1/2023-06-21/epoch-18.pth.tar", help='pretrained weight path shuangfenzhi ceshi')
    # parser.add_argument('--pre_trained_path', type=str, default="/data/wst/video-pace/pretrain_cks/sr_4_mymodel3_lr_0.003_len_16_sz_112_mytrain3/2023-07-09/epoch-18.pth.tar")
    # 对比学习加mlp层
    # parser.add_argument('--pre_trained_path', type=str, default="/data/wst/video-pace/pretrain_cks/sr_4_mymodel3_lr_0.003_len_16_sz_112_mytrain3/2023-07-10/epoch-18.pth.tar")
    # Restnet部分多尺度
    # parser.add_argument('--pre_trained_path', type=str, default="/data/wst/video-pace/pretrain_cks/sr_4_mymodelv4_lr_0.003_len_16_sz_112_mytrain3/2023-07-11/epoch-18.pth.tar")
    # parser.add_argument('--pre_trained_path', type=str, default="/data/wst/video-pace/pretrain_cks/sr_4_mymodelv4_lr_0.002_len_16_sz_112_mytrain3/2023-07-15/epoch-18.pth.tar")
    # parser.add_argument('--pre_trained_path', type=str, default="/data/wst/video-pace/pretrain_cks/sr_4_mymodelv4_lr_0.002_len_16_sz_112_mytrain3/2023-07-25/epoch-18.pth.tar",help="对比学习mlp+FPN多尺度三层拼接_lr0.002")
    # parser.add_argument('--pre_trained_path', type=str, default="/data/wst/video-pace/pretrain_cks/sr_4_mymodelv4_lr_0.002_len_16_sz_112_mytrain3/2023-07-27/mlpepoch-18.pth.tar",help="对比学习mlp+FPN多尺度4层拼接_lr0.002")
    # parser.add_argument('--pre_trained_path', type=str, default="/data/wst/video-pace/pretrain_cks/sr_4_mymodelv4_lr_0.002_len_16_sz_112_mytrain3/2023-07-29/mlpepoch-18.pth.tar",help="对比学习mlp+FPN多尺度4层拼接+SEAtten_lr0.002,mytrain3的对比学习就是simclr")
    # parser.add_argument('--pre_trained_path', type=str, default="/data/wst/video-pace/pretrain_cks/sr_4_mymodelv4_lr_0.002_len_16_sz_112_mytrain3/2023-08-02/mlpepoch-18.pth.tar",help="对比学习mlp+FPN多尺度4层拼接Atten_lr0.002,mytrain3的对比学习就是simclr")
    # parser.add_argument('--pre_trained_path', type=str, default="/data/wst/video-pace/pretrain_cks/sr_4_mymodelv4_lr_0.002_len_16_sz_112_mytrain3/2023-08-04/mlpepoch-18.pth.tar",help="对比学习mlp+FPN多尺度4层拼接+cbamAtten_lr0.002,mytrain3的对比学习就是simclr,08-03也是")
    # parser.add_argument('--pre_trained_path', type=str, default="/data/wst/video-pace/pretrain_cks/sr_4_mymodelv4_lr_0.003_len_16_sz_112_mytrain3/2023-08-05/mlpepoch-18.pth.tar",help="对比学习mlp+FPN多尺度4层拼接+cbamAtten_lr0.003,mytrain3的对比学习就是simclr,08-03也是")
    # parser.add_argument('--pre_trained_path', type=str, default="/data/wst/video-pace/pretrain_cks/sr_4_mymodelv4_lr_0.002_len_16_sz_112_mytrain3/2023-08-06/mlpepoch-18.pth.tar",help="对比学习mlp+FPN多尺度4层拼接+ecaAtten_lr0.002,mytrain3的对比学习就是simclr,08-03也是")
    # parser.add_argument('--pre_trained_path', type=str, default="/data/wst/video-pace/pretrain_cks/sr_4_mymodelv4_lr_0.003_len_16_sz_112_mytrain3/2023-08-08/mlpepoch-18.pth.tar",help="对比学习+FPN多尺度四层拼接+seatten+swin替换trans+lr0.002")
    # parser.add_argument('--pre_trained_path', type=str, default="/data/wst/video-pace/pretrain_cks/sr_4_mymodelv6_lr_0.002_len_16_sz_112_mytrain3/2023-08-09/mlpepoch-18.pth.tar",help="moco+FPN多尺度四层拼接+后接seatten+swin替换")
    # parser.add_argument('--pre_trained_path', type=str, default="/data/wst/video-pace/pretrain_cks/sr_4_mymodelv4_lr_0.002_len_16_sz_112_mytrain3/2023-08-11/mlpepoch-18.pth.tar",help="moco+FPN多尺度四层拼接+后接seatten+swin替换,修改swin参数版本")
    # parser.add_argument('--pre_trained_path', type=str, default= "/data/wst/video-pace/pretrain_cks/sr_4_mymodelv4_lr_0.002_len_16_sz_112_mytrain3/2023-08-11/mlpepoch-17.pth.tar",help="FPN多尺度四层拼接+后接cbamatten+swin替换,修改swin参数版本")
    # parser.add_argument('--pre_trained_path', type=str, default= "/data/wst/video-pace/pretrain_cks/sr_4_mymodelv4_lr_0.002_len_16_sz_112_mytrain3/2023-08-13/mlpepoch-18.pth.tar",help="FPN多尺度四层拼接256+后接cbamatten+swin替换,修改swin参数版本")
    # parser.add_argument('--pre_trained_path', type=str, default= "/data/wst/video-pace/pretrain_cks/sr_4_mymodelv4_lr_0.002_len_16_sz_112_mytrain3/2023-08-15/mlpepoch-18.pth.tar",help="FPN多尺度四层拼接256+后接seatten+swin替换,修改swin参数版本")
    # parser.add_argument('--pre_trained_path', type=str, default= "/data/wst/video-pace/pretrain_cks/sr_4_mymodelv4_lr_0.002_len_16_sz_112_mytrain3/2023-08-17/mlpepoch-18.pth.tar",help="FPN多尺度四层拼接256+后接imdnatten+swin替换,修改swin参数版本")
    # parser.add_argument('--pre_trained_path', type=str, default="/data/wst/video-pace/pretrain_cks/sr_4_mymodelv4_lr_0.003_len_16_sz_112_mytrain3/2023-08-18/mlpepoch-18.pth.tar",help="FPN多尺度四层拼接256+里接senatten+swin替换,修改swin参数版本")
    # parser.add_argument('--pre_trained_path', type=str, default="/data/wst/video-pace/pretrain_cks/sr_4_mymodelv6_lr_0.002_len_16_sz_112_mytrain3/2023-08-19/mlpepoch-18.pth.tar",help="moco+FPN多尺度四层拼接256+后接senatten+swin替换,修改swin参数版本")
    # parser.add_argument('--pre_trained_path', type=str, default="/data/wst/video-pace/pretrain_cks/sr_4_mymodelv4_lr_0.001_len_16_sz_112_mytrain3/2023-08-20/mlpepoch-18.pth.tar",help="FPN多尺度四层拼接256+后接senatten+swin替换,修改swin参数版[2,2]->[2,4]")
    # parser.add_argument('--pre_trained_path', type=str, default="/data/wst/video-pace/pretrain_cks/sr_4_mymodelv4_lr_0.001_len_16_sz_112_mytrain3/2023-08-22/mlpepoch-18.pth.tar", help="FPN多尺度四层拼接256+接senatten+swin替换,修改swin参数版本")
    # parser.add_argument('--pre_trained_path', type=str, default="/data/wst/video-pace/pretrain_cks/sr_4_mymodelv4_lr_0.001_len_16_sz_112_mytrain3/2023-08-25/mlpepoch-18.pth.tar", help="FPN多尺度四层拼接64+接senatten+swin替换,修改swin参数版本")
    # parser.add_argument('--pre_trained_path', type=str, default="/data/wst/video-pace/pretrain_cks/sr_4_mymodelv4_lr_0.001_len_16_sz_112_mytrain3/2023-08-26/mlpepoch-18.pth.tar", help="FPN多尺度四层拼接128+接senatten+swin替换,修改swin参数版本")
    # parser.add_argument('--pre_trained_path', type=str, default="/data/wst/video-pace/pretrain_cks/sr_4_mymodelv4_lr_0.001_len_16_sz_112_mytrain3/2023-08-31/mlpepoch-18.pth.tar", help='256 [2,2,4]')
    # parser.add_argument('--pre_trained_path', type=str, default="/data/wst/video-pace/pretrain_cks/sr_4_mymodelv4_lr_0.001_len_16_sz_112_mytrain3/2023-09-16/mlpepoch-18.pth.tar", help='256 [2,4] EMA32 lr0.001')
    # parser.add_argument('--pre_trained_path', type=str, default="/data/wst/video-pace/pretrain_cks/sr_4_mymodelv4_lr_0.002_len_16_sz_112_mytrain3/2023-09-19/mlpepoch-18.pth.tar", help='256 [2,4] EMA32 lr0.002')
    # parser.add_argument('--pre_trained_path', type=str, default="/data/wst/video-pace/pretrain_cks/sr_4_mymodelv4_lr_0.0001_len_16_sz_112_mytrain3/2023-09-21/mlpepoch-24.pth.tar", help='256 [2,2,4] SE lr0.0001')
    # parser.add_argument('--pre_trained_path', type=str, default="/data/wst/video-pace/pretrain_cks/sr_4_mymodelv4_lr_0.001_len_16_sz_112_mytrain3/2023-09-29/mlpepoch-18.pth.tar", help='256 [2,2,4] SE lr0.0001')
    # parser.add_argument('--pre_trained_path', type=str, default="/data/wst/video-pace/pretrain_cks/sr_4_mymodelv4_lr_0.001_len_16_sz_112_mytrain3/2023-10-08/mlpepoch-18.pth.tar", help='256 [2,2,4] SE lr0.0001 调制可变形卷积')
    # parser.add_argument('--pre_trained_path', type=str, default="/data/wst/video-pace/pretrain_cks/sr_4_mymodelv4_lr_0.001_len_16_sz_112_mytrain3/2023-10-10/mlpepoch-18.pth.tar", help='256 [2,2,4] SE lr0.0001 调制可变形卷积')
    # parser.add_argument('--pre_trained_path', type=str, default= "/data/wst/video-pace/pretrain_cks/sr_4_mymodelv4_lr_0.001_len_16_sz_112_mytrain3/2023-10-13/mlpepoch-18.pth.tar", help='256 [2,2,4] SimAM lr0.0001')
    # parser.add_argument('--pre_trained_path', type=str, default="/data/wst/video-pace/pretrain_cks/sr_4_mymodelv4_lr_0.0001_len_16_sz_112_mytrain3/2023-10-15/mlpepoch-18.pth.tar", help='256 [2,2,4] gam lr0.0001')
    # parser.add_argument('--pre_trained_path', type=str, default="/data/wst/video-pace/pretrain_cks/sr_4_mymodelv4_lr_0.001_len_16_sz_112_mytrain3/2023-10-17/mlpepoch-18.pth.tar", help='256 [2,2,4] SimAM lr0.001 去掉swin前面的tempconv')
    # parser.add_argument('--pre_trained_path', type=str, default="/data/wst/video-pace/pretrain_cks/sr_4_mymodelv4_lr_0.001_len_16_sz_112_mytrain3/2023-10-21/mlpepoch-18.pth.tar", help='1017，去掉颜色抖动')
    # parser.add_argument('--pre_trained_path', type=str, default="/data/wst/video-pace/pretrain_cks/sr_4_mymodelv4_lr_0.001_len_16_sz_112_mytrain3/2023-10-25/mlpepoch-18.pth.tar", help='1017，去掉颜色抖动, 修改对比学习部分')
    # parser.add_argument('--pre_trained_path', type=str, default="/data/wst/video-pace/pretrain_cks/sr_4_mymodelv4_lr_0.001_len_16_sz_112_mytrain3/2023-10-26/mlpepoch-18.pth.tar", help='1017，去掉颜色抖动，修改对比学习部分')
    # parser.add_argument('--pre_trained_path', type=str, default="/data/wst/video-pace/pretrain_cks/sr_4_mymodelv4_lr_0.001_len_16_sz_112_mytrain3/2023-11-10/mlpepoch-18.pth.tar", help='1110，去掉颜色抖动，修改对比学习部分,无tmpconv,simclr,64')
    # parser.add_argument('--pre_trained_path', type=str, default="/data/wst/video-pace/pretrain_cks/sr_4_mymodelv4_lr_0.001_len_16_sz_112_mytrain3/2023-11-11/mlpepoch-18.pth.tar", help='1111，去掉颜色抖动，修改对比学习部分,有tmoconv,128')
    parser.add_argument('--pre_trained_path', type=str, default="/data/wst/video-pace/testscore/pre/1115ablo/mlpepoch-18.pth.tar", help='1115，消融去掉对比学习,FPN64')
    # parser.add_argument('--pre_trained_path', type=str, default="/data/wst/video-pace/pretrain_cks/sr_4_mymodelv4_lr_0.001_len_16_sz_112_mytrain3/2023-11-16/mlpepoch-18.pth.tar", help='1116，消融去掉失真类型,FPN64')
    # parser.add_argument('--pre_trained_path', type=str, default="/data/wst/video-pace/pretrain_cks/sr_4_mymodelv4_lr_0.001_len_16_sz_112_mytrain3/2023-11-20/mlpepoch-18.pth.tar", help='1116，消融只有对比学习,FPN64')
    # parser.add_argument('--pre_trained_path', type=str, default="/data/wst/video-pace/pretrain_cks/sr_4_mymodelv4_lr_0.001_len_16_sz_112_mytrain3/2023-12-16/mlpepoch-18.pth.tar", help='1116，消融只有对比学习,FPN64')
    # parser.add_argument('--pre_trained_path', type=str, default="/data/wst/video-pace/pretrain_cks/sr_4_mymodelv4_lr_0.001_len_16_sz_112_mytrain3/2024-03-20/mlpepoch-18.pth.tar", help='多尺度消融')


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
