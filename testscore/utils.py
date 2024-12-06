# --------------------------------------------------------
# Swin Transformer
# Code Link: https://github.com/microsoft/Swin-Transformer
# --------------------------------------------------------'

import os
import random

import cv2
import numpy as np
import torch
import torch.distributed as dist

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        if 'amp' in checkpoint and config.AMP_OPT_LEVEL != "O0" and checkpoint['config'].AMP_OPT_LEVEL != "O0":
            amp.load_state_dict(checkpoint['amp'])
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy


def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}

    if config.AMP_OPT_LEVEL != "O0":
        save_state['amp'] = amp.state_dict()

    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.gr
        ad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

# def CenterCrop(video_clip, output_size):
#     h, w = video_clip.shape[1:3]
#
#     if isinstance(output_size, int):
#         new_h, new_w = output_size, output_size
#     else:
#         assert len(output_size) == 2
#         new_h, new_w = output_size
#
#     h_start = int((h - new_h) / 2)
#     w_start = int((w - new_w) / 2)
#
#     center_crop_video_clip = video_clip[:, h_start:h_start + new_h,
#                                 w_start:w_start + new_w, :]
#
#     return center_crop_video_clip
#
# def ClipResize(video_clip, output_size):
#     assert isinstance(output_size, (int, tuple))
#     if isinstance(output_size, int):
#         output_size = (output_size, output_size)
#     else:
#         assert len(output_size) == 2
#
#     rsz_video_clip = []
#     new_h, new_w = output_size
#
#     for frame in video_clip:
#         rsz_frame = cv2.resize(frame, (new_w, new_h))
#         rsz_video_clip.append(rsz_frame)
#
#     return np.array(rsz_video_clip)
#
#
# def RandomCrop(video_clip, output_size):
#     assert isinstance(output_size, (int, tuple))
#     if isinstance(output_size, int):
#         output_size = (output_size, output_size)
#     else:
#         assert len(output_size) == 2
#
#     h, w = video_clip.shape[1:3]
#     new_h, new_w = output_size
#
#     h_start = random.randint(0, h-new_h)
#     w_start = random.randint(0, w-new_w)
#
#     rnd_crop_video_clip = video_clip[:, h_start:h_start+new_h, w_start:w_start+new_w, :]
#
#     return rnd_crop_video_clip
#
# def RandomHorizontalFlip(video_clip, p=0.5):
#     if np.random.random() < p:
#         flip_video_clip = np.flip(video_clip, axis=2).copy()
#         return flip_video_clip
#     else:
#         return video_clip