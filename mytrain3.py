import datetime
import logging
import os
import time
import argparse
import numpy as np

import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader
from logClass import *
from datasets.mydata3 import mydata_pace_pretrain3
from models import r21d, r3d, c3d, s3d_g, mymodel3
from utils.video_transforms import RandomCrop, RandomHorizontalFlip, CenterCrop, ClipResize, ToTensor
from utils.shezhizhongzi import seed_everything, worker_init_fn
from functools import partial
from simclrloss.simclr import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='1', help='gpu id')
    parser.add_argument('--height', type=int, default=256, help='resize height')
    parser.add_argument('--width', type=int, default=256, help='resize width')
    parser.add_argument('--clip_len', type=int, default=64, help='64, input clip length')
    parser.add_argument('--crop_sz', type=int, default=224, help='crop size')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--bs', type=int, default=32, help='32, batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers')
    parser.add_argument('--epoch', type=int, default=18, help='total epoch')
    parser.add_argument('--max_sr', type=int, default=4, help='largest sampling rate')
    parser.add_argument('--num_classes', type=int, default=4, help='num of classes')
    parser.add_argument('--max_save', type=int, default=5, help='max save epoch num')
    parser.add_argument('--dataset', type=str, default='ucf101', help='ucf101/k400')
    parser.add_argument('--pf', type=int, default=10, help='print frequency')
    parser.add_argument('--model', type=str, default='mymodel3', help='s3d/r21d/r3d/c3d/mymodel3, pretrain model')

    parser.add_argument('--data_list', type=str, default='/data/wst/video-pace/datasets/zice/datadis.list', help='data list')

    parser.add_argument('--rgb_prefix', type=str, default='/data/wst/DATABASE/videopacedata/data2/data/', help='dataset dir')
    parser.add_argument('--source_list', type=str, default='/data/wst/video-pace/datasets/zice/datasource.list', help='data list')

    parser.add_argument('--source_prefix', type=str, default='/data/wst/DATABASE/videopacedata/data1/data1/', help='dataset dir')


    # 对比学习部分加入的args
    parser.add_argument('--n-views', default=2, type=int, metavar='N',
                        help='Number of views for contrastive learning training.')
    parser.add_argument('--temperature', default=0.07, type=float,
                        help='softmax temperature (default: 0.07)')

    args = parser.parse_args()

    return args


def cosine_distance_loss(original_features, distorted_features):
    """
    Compute the cosine distance loss between original and distorted features.

    Args:
    - original_features: Tensor of shape [batch_size, feature_dim]
    - distorted_features: Tensor of shape [batch_size, feature_dim]

    Returns:
    - loss: Scalar tensor representing the cosine distance loss
    """
    # Normalize features to compute cosine similarity
    original_features = F.normalize(original_features, dim=1)
    distorted_features = F.normalize(distorted_features, dim=1)

    cosine_similarity = torch.mm(original_features, distorted_features.t())
    loss = cosine_similarity.diag().mean()
    return loss

def info_nce_loss(features):

    labels = torch.cat([torch.arange(args.bs) for i in range(args.n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)
    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
    logits = logits / args.temperature
    return logits, labels


def train(args, logger):

    if torch.cuda.is_available():
        # 获取当前选定的 GPU 的编号，默认为 0
        current_device = torch.cuda.current_device()

        # 获取当前 GPU 的名称
        device_name = torch.cuda.get_device_name(current_device)

        # 打印 GPU 的编号和名称
        print(f"当前使用的 GPU 编号: {current_device}")
        print(f"当前使用的 GPU 名称: {device_name}")
    else:
        print("CUDA 不可用，没有检测到 GPU。")

    torch.backends.cudnn.benchmark = True
    # 手动在这加1234来区别文件夹
    exp_name = 'sr_{}_{}_lr_{}_len_{}_sz_{}_mytrain3'.format(args.max_sr, args.model, args.lr, args.clip_len, args.crop_sz)
    # print(exp_name)
    logger.info(exp_name)

    pretrain_cks_path = os.path.join('pretrain_cks', exp_name)
    log_path = os.path.join('visual_logs', exp_name)

    if not os.path.exists(pretrain_cks_path):
        os.makedirs(pretrain_cks_path)

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    transforms_ = transforms.Compose(
        [ClipResize((args.height, args.width)),  # h x w
         RandomCrop(args.crop_sz),
         RandomHorizontalFlip(0.5)]
    )
    # TODO : 把颜色抖动去掉
    color_jitter = transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
    color_jitter = transforms.RandomApply([color_jitter], p=0.8)

    train_dataset = mydata_pace_pretrain3(args.data_list, args.rgb_prefix, args.source_list, args.source_prefix,
                                         clip_len=args.clip_len, max_sr=args.max_sr,
                                         transforms_=transforms_, color_jitter_=color_jitter)

    logger.info(f"len of training data:{len(train_dataset)}")
    dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True, drop_last=True, worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))

    ## 2. init model
    if args.model == 'r21d':
        model = r21d.R2Plus1DNet(num_classes=args.num_classes)
    elif args.model == 'r3d':
        model = r3d.R3DNet(num_classes=args.num_classes)
    elif args.model == 'c3d':
        model = c3d.C3D(num_classes=args.num_classes)
    elif args.model == 's3d':
        model = s3d_g.S3D(num_classes=args.num_classes, space_to_depth=False)
    elif args.model == 'mymodel3':
        model = mymodel3.pre_trained_model3(num_classes=args.num_classes)
    elif args.model == 'mymodelv4':
        model = mymodelv4.pre_trained_model4(num_classes=args.num_classes)

    # 3. define loss and lr
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.005)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)

    # 4. multi gpu
    if torch.cuda.device_count() > 1:
        # print("Let's use", torch.cuda.device_count(), "GPUs!")
        logger.info(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    model.to(device)
    criterion.to(device)

    # writer = SummaryWriter(log_dir=log_path)
    iterations = 1

    model.train()
    for epoch in range(args.epoch):
        start_time = time.time()
        for i, sample in enumerate(dataloader):
            # action_label就是读取的.list文件里的,失真类型索引
            rgb_clip, rgb_clip2, labels, action_label, org_rgb_clip, source_clip1, source_clip2 = sample
            print(action_label)
            rgb_clip = rgb_clip.to(device, dtype=torch.float)
            rgb_clip2 = rgb_clip2.to(device, dtype=torch.float)

            labels = labels.to(device)
            action_label = action_label.to(device)
            org_rgb_clip = org_rgb_clip.to(device, dtype=torch.float)
            source_clip1 = source_clip1.to(device, dtype=torch.float)
            source_clip2 = source_clip2.to(device, dtype=torch.float)

            optimizer.zero_grad()
            # 将视频帧输入到模型返回pace,distype,rgb_clip(对比学习需要到的特征)
            outputs1 = model(rgb_clip, source_clip1)
            outputs2 = model(rgb_clip2, source_clip2)  # TODO 只加播放速度注释
            # org_outputs = model(org_rgb_clip)
            # org_outputs = model(rgb_clip2)
            # 新增 加入对比学习infoNCE需要的特征
            pre_pace, pre_dis, distorted_features1, original_features1 = outputs1[0], outputs1[1], outputs1[2], outputs1[3]
            distorted_features2, original_features2 = outputs2[2], outputs2[3]  # TODO 只加播放速度注释

            # features2 = outputs2[2]

            # org_features = org_outputs[1]
            # 调用infoNCE函数
            dis_features = torch.cat([distorted_features1, distorted_features2], dim=0)  # 注释  # TODO 只加播放速度注释
            org_features = torch.cat([original_features1, original_features2], dim=0)  # 注释  # TODO 只加播放速度注释

            # TODO ： 修改对比学习部分
            # logits, labels2 = info_nce_loss(features)
            # loss = criterion(pre_pace, labels) + criterion(pre_dis, action_label) + 0.5*criterion(logits, labels2)

            dis_logits, dis_labels = info_nce_loss(dis_features)  # TODO 只加播放速度注释
            org_logits, org_labels = info_nce_loss(org_features)  # TODO 只加播放速度注释
            loss1 = cosine_distance_loss(org_features, dis_features)  # TODO 只加播放速度注释

            # loss = 0.8*criterion(pre_pace, labels) + criterion(pre_dis, action_label) + 0.8*criterion(dis_logits, dis_labels) + 0.8*criterion(org_logits, org_labels) + 0.5*loss1
            # TODO:去掉播放速度
            # loss = criterion(pre_dis, action_label) + criterion(dis_logits, dis_labels) + criterion(org_logits, org_labels) + 0.5*loss1
            # TODO:去掉失真类型
            # loss = criterion(pre_pace, labels) + criterion(dis_logits, dis_labels) + criterion(org_logits, org_labels) + 0.5*loss1
            # TODO:只加对比学习
            loss = criterion(dis_logits, dis_labels) + criterion(org_logits, org_labels) + 0.5*loss1
            # TODO：只加播放速度
            # loss = criterion(pre_pace, labels)
            # TODO ： 修改对比学习部分
            loss.backward()
            optimizer.step()

            probs_pace = nn.Softmax(dim=1)(pre_pace)
            probs = nn.Softmax(dim=1)(pre_dis)
            preds_pace = torch.max(probs_pace, 1)[1]
            preds_dis = torch.max(probs, 1)[1]

            acc_pace = torch.sum(preds_pace == labels.data).detach().cpu().numpy().astype(float)
            acc_dis = torch.sum(preds_dis == action_label.data).detach().cpu().numpy().astype(float)
            acc_pace = acc_pace / args.bs
            acc_dis = acc_dis / args.bs

            iterations += 1

            if i % args.pf == 0:
                logger.info(f"[Epoch{epoch + 1}/{i}] Loss: {loss} Acc_Pace: {acc_pace} Acc_Dis: {acc_dis} Time {time.time() - start_time} ")
            start_time = time.time()

        scheduler.step()
        model_saver(model, optimizer, epoch, args.max_save, pretrain_cks_path)

def model_saver(net, optimizer, epoch, max_to_keep, model_save_path):
    # 获取当前日期
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    # 在 model_save_path 的末尾添加当前日期
    model_save_path = os.path.join(model_save_path, date_str)
    # 检查模型保存路径是否存在，若不存在则创建
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    tmp_dir = os.listdir(model_save_path)
    # print(tmp_dir)
    logger.info(tmp_dir)
    tmp_dir.sort()
    if len(tmp_dir) >= max_to_keep:
        os.remove(os.path.join(model_save_path, tmp_dir[0]))

    checkpoint_path = os.path.join(model_save_path, 'mlpepoch-' + '{:02}'.format(epoch + 1) + '.pth.tar')
    torch.save(net.state_dict(), checkpoint_path)

if __name__ == '__main__':

    seed = 11
    seed_everything(seed)
    args = parse_args()
    logger = loggerClass(logLevel='DEBUG', save_level=logging.INFO, is_debug=True).init_logger(2)
    # 设置stream流的最低等级为DEBUG，即为最低logger.debug()的消息就可以打印在控制台
    # 设置file流的最低等级为INFO，即为最低logger.info()的消息就可以保存在文件中
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    # print(device)
    logger.info(device)  # 调用logger，并设置内容的等级为INFO，INFO等级的消息会打印在控制台并且会保存为.log文件中
    train(args, logger)
