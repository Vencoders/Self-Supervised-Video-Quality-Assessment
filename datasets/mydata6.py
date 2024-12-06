import os
import cv2
import random
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from utils.video_transforms import RandomCrop, RandomHorizontalFlip, CenterCrop, ClipResize, ToTensor
class mydata_pace_pretrain6(Dataset):
    def __init__(self, data_list, rgb_prefix, source_list, source_prefix, clip_len,  max_sr, transforms_=None, color_jitter_=None): # yapf: disable:
        lines = open(data_list)
        self.rgb_lines = list(lines) * 10
        lines2 = open(source_list)
        self.source_line = list(lines2) * 10
        self.rgb_prefix = rgb_prefix
        self.source_prefix = source_prefix

        self.clip_len = clip_len
        self.max_sr = max_sr
        self.toPIL = transforms.ToPILImage()
        self.transforms_ = transforms_
        self.color_jitter_ = color_jitter_

    def __len__(self):
        return len(self.rgb_lines)

    def __getitem__(self, idx):
        rgb_line = self.rgb_lines[idx].strip('\n').split()
        source_line = self.rgb_lines[idx].strip('\n').split()
        # sample_name list里面的视频名称,action_label失真类别,num_frames帧数
        sample_name, action_label, num_frames = rgb_line[0], int(rgb_line[1]), int(rgb_line[2])
        sample_name_source, num_frames = source_line[0], int(source_line[2])
        sample_name = sample_name[:-4]
        sample_name_source = sample_name_source[:-4]
        rgb_dir = os.path.join(self.rgb_prefix, sample_name)
        source_dir = os.path.join(self.source_prefix, sample_name_source)
        sample_rate = random.randint(1, self.max_sr)
        start_frame = random.randint(1, num_frames)

        rgb_clip = self.loop_load_rgb(rgb_dir, start_frame, sample_rate,
                                      self.clip_len, num_frames, sample_name, action_label)
        source_clip = self.loop_load_source(source_dir, start_frame, sample_rate,
                                      self.clip_len, num_frames, sample_name_source, action_label)

        label = sample_rate - 1

        trans_clip1 = self.transforms_(rgb_clip)
        trans_clip2 = self.transforms_(rgb_clip)

        trans_source = self.transforms_(source_clip)

        ## apply different color jittering for each frame in the video clip
        trans_clip_cj1 = []
        for frame in trans_clip1:
            frame = self.toPIL(frame)  # PIL image
            frame = self.color_jitter_(frame)  # tensor [C x H x W]
            frame = np.array(frame)
            trans_clip_cj1.append(frame)
        trans_clip_cj2 = []
        for frame in trans_clip2:
            frame = self.toPIL(frame)  # PIL image
            frame = self.color_jitter_(frame)  # tensor [C x H x W]
            frame = np.array(frame)
            trans_clip_cj2.append(frame)

        trans_source_clip = []
        for frame in trans_source:
            frame = self.toPIL(frame)
            frame = self.color_jitter_(frame)  # tensor [C x H x W]
            frame = np.array(frame)
            trans_source_clip.append((frame))

        trans_clip_cj1 = np.array(trans_clip_cj1).transpose(3, 0, 1, 2)
        trans_clip_cj2 = np.array(trans_clip_cj2).transpose(3, 0, 1, 2)

        trans_source_clip = np.array(trans_source_clip).transpose(3, 0, 1, 2)
        rgb_clip = np.array(rgb_clip).transpose(3, 0, 1, 2)
        # 输出rgb_clip作为原始的 label可以当成播放速度 action_label失真类别
        return trans_clip_cj1, trans_clip_cj2, label, action_label, rgb_clip, trans_source_clip

    def loop_load_rgb(self, video_dir, start_frame, sample_rate, clip_len,
                      num_frames, sample_name, action_label):

        video_clip = []
        idx = 0

        for i in range(clip_len):
            cur_img_path = os.path.join(
                video_dir, "{}_{}_frame{:06}.png".format(sample_name, action_label, start_frame + idx * sample_rate))

            img = cv2.imread(cur_img_path)
            video_clip.append(img)

            if (start_frame + (idx + 1) * sample_rate) > num_frames:
                start_frame = 1
                idx = 0
            else:
                idx += 1

        video_clip = np.array(video_clip)
        return video_clip

    def loop_load_source(self, video_dir, start_frame, sample_rate, clip_len,
                      num_frames, sample_name, action_label):

        video_clip = []
        idx = 0

        for i in range(clip_len):
            cur_img_path = os.path.join(
                video_dir, "{}_0_frame{:06}.png".format(sample_name, start_frame + idx * sample_rate))

            img = cv2.imread(cur_img_path)
            video_clip.append(img)

            if (start_frame + (idx + 1) * sample_rate) > num_frames:
                start_frame = 1
                idx = 0
            else:
                idx += 1

        video_clip = np.array(video_clip)
        return video_clip

    def loop_load_org_rgb(self, video_dir, start_frame, sample_rate, clip_len,
                      num_frames, sample_name, action_label):

        video_clip = []
        idx = 0

        for i in range(clip_len):
            # cur_img_path = os.path.join(
            #     video_dir,
            #     "frame{:06}.jpg".format(start_frame + idx * sample_rate))
            cur_img_path = os.path.join(
                video_dir,
                "{}_0_frame{:06}.png".format(sample_name, start_frame + idx * sample_rate))

            img = cv2.imread(cur_img_path)
            video_clip.append(img)

            if (start_frame + (idx + 1) * sample_rate) > num_frames:
                start_frame = 1
                idx = 0
            else:
                idx += 1

        video_clip = np.array(video_clip)

        return video_clip
