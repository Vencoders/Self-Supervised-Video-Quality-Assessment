import os
import cv2
import random
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from utils.video_transforms import RandomCrop, RandomHorizontalFlip, CenterCrop, ClipResize, ToTensor
# class mydata_pace_pretrain3(Dataset):
#
#     def __init__(self, data_list, rgb_prefix, clip_len,  max_sr, transforms_=None, color_jitter_=None): # yapf: disable:
#         lines = open(data_list)
#         self.rgb_lines = list(lines) * 10
#         self.rgb_prefix = rgb_prefix
#         self.clip_len = clip_len
#         self.max_sr = max_sr
#         self.toPIL = transforms.ToPILImage()
#         self.transforms_ = transforms_
#         self.color_jitter_ = color_jitter_
#
#     def __len__(self):
#         return len(self.rgb_lines)
#
#     def __getitem__(self, idx):
#         rgb_line = self.rgb_lines[idx].strip('\n').split()
#         # sample_name list里面的视频名称,action_label失真类别
#         sample_name, action_label, num_frames = rgb_line[0], int(rgb_line[1]), int(rgb_line[2])
#         sample_name = sample_name[:-4]
#         rgb_dir = os.path.join(self.rgb_prefix, sample_name)
#         sample_rate = random.randint(1, self.max_sr)
#         start_frame = random.randint(1, num_frames)
#
#         rgb_clip = self.loop_load_rgb(rgb_dir, start_frame, sample_rate,
#                                       self.clip_len, num_frames, sample_name, action_label)
#
#         label = sample_rate - 1
#
#         trans_clip = self.transforms_(rgb_clip)
#
#         ## apply different color jittering for each frame in the video clip
#         trans_clip_cj = []
#         for frame in trans_clip:
#             frame = self.toPIL(frame)  # PIL image
#             frame = self.color_jitter_(frame)  # tensor [C x H x W]
#             frame = np.array(frame)
#             trans_clip_cj.append(frame)
#
#         trans_clip_cj = np.array(trans_clip_cj).transpose(3, 0, 1, 2)
#         rgb_clip = np.array(rgb_clip).transpose(3, 0, 1, 2)
#
#         return trans_clip_cj, label, action_label, rgb_clip
#
#     def loop_load_rgb(self, video_dir, start_frame, sample_rate, clip_len,
#                       num_frames, sample_name, action_label):
#
#         video_clip = []
#         idx = 0
#
#         for i in range(clip_len):
#             # cur_img_path = os.path.join(
#             #     video_dir,
#             #     "frame{:06}.jpg".format(start_frame + idx * sample_rate))
#             cur_img_path = os.path.join(
#                 video_dir,
#                 "{}_{}_frame{:06}.png".format(sample_name, action_label, start_frame + idx * sample_rate))
#
#
#
#             img = cv2.imread(cur_img_path)
#             video_clip.append(img)
#
#             if (start_frame + (idx + 1) * sample_rate) > num_frames:
#                 start_frame = 1
#                 idx = 0
#             else:
#                 idx += 1
#
#         video_clip = np.array(video_clip)
#
#         return video_clip
#
#
# if __name__ == '__main__':
#
#     # data_list = '/data/wst/video-pace/datasets/data1.list'
#     # rgb_prefix = '/data/wst/RGB_images/'
#     data_list = '/data/wst/video-pace/datasets/zice/datadis.list'
#     rgb_prefix = '/data/wst/video-pace/data/'
#     # data_list = '/data/wst/video-pace/list/train_ucf101_split2.list'
#     # rgb_prefix = '/data/wst/pacedata/jpegs_256/'
#
#
#     transforms_ = transforms.Compose([
#         ClipResize((128,171)),
#         CenterCrop(112),
#         RandomHorizontalFlip(0.5)
#     ])
#
#     color_jitter = transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
#     rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
#
#     train_dataset = mydata_pace_pretrain3(data_list, rgb_prefix, clip_len=16, max_sr=4,
#                                    transforms_=transforms_, color_jitter_=rnd_color_jitter)
#
#     dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
#
#     for iter, sample in enumerate(dataloader):
#
#         rgb_clip, label = sample
#         print(rgb_clip.shape)
#         rgb_clip = rgb_clip[0].numpy()
#
#         print(rgb_clip.shape)
#
#         rgb_clip = rgb_clip.transpose(1, 2, 3, 0)
#         for i in range(len(rgb_clip)):
#             cur_frame = rgb_clip[i]
#
#             cv2.imshow("img", cur_frame)
#             cv2.waitKey()
import os
import cv2
import random
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from utils.video_transforms import RandomCrop, RandomHorizontalFlip, CenterCrop, ClipResize, ToTensor
class mydata_pace_pretrain3(Dataset):
    # 单分支处理数据不加失真程度
    # def __init__(self, data_list, rgb_prefix, clip_len,  max_sr, transforms_=None, color_jitter_=None): # yapf: disable:
    #     lines = open(data_list)
    #     self.rgb_lines = list(lines) * 10
    #     self.rgb_prefix = rgb_prefix
    #     self.clip_len = clip_len
    #     self.max_sr = max_sr
    #     self.toPIL = transforms.ToPILImage()
    #     self.transforms_ = transforms_
    #     self.color_jitter_ = color_jitter_
    #
    # def __len__(self):
    #     return len(self.rgb_lines)
    #
    # def __getitem__(self, idx):
    #     rgb_line = self.rgb_lines[idx].strip('\n').split()
    #     # sample_name list里面的视频名称,action_label失真类别
    #     sample_name, action_label, num_frames = rgb_line[0], int(rgb_line[1]), int(rgb_line[2])
    #     sample_name = sample_name[:-4]
    #     rgb_dir = os.path.join(self.rgb_prefix, sample_name)
    #     sample_rate = random.randint(1, self.max_sr)
    #     start_frame = random.randint(1, num_frames)
    #
    #     rgb_clip = self.loop_load_rgb(rgb_dir, start_frame, sample_rate,
    #                                   self.clip_len, num_frames, sample_name, action_label)
    #
    #     label = sample_rate - 1
    #
    #     trans_clip = self.transforms_(rgb_clip)
    #
    #     ## apply different color jittering for each frame in the video clip
    #     trans_clip_cj = []
    #     for frame in trans_clip:
    #         frame = self.toPIL(frame)  # PIL image
    #         frame = self.color_jitter_(frame)  # tensor [C x H x W]
    #         frame = np.array(frame)
    #         trans_clip_cj.append(frame)
    #
    #     trans_clip_cj = np.array(trans_clip_cj).transpose(3, 0, 1, 2)
    #     rgb_clip = np.array(rgb_clip).transpose(3, 0, 1, 2)
    #     # 输出rgb_clip作为原始的 label可以当成播放速度 action_label失真类别
    #     return trans_clip_cj, label, action_label, rgb_clip

    # def __init__(self, data_list, rgb_prefix, clip_len,  max_sr, transforms_=None, color_jitter_=None): # yapf: disable:
    #     lines = open(data_list)
    #     self.rgb_lines = list(lines) * 10
    #     self.rgb_prefix = rgb_prefix
    #     self.clip_len = clip_len
    #     self.max_sr = max_sr
    #     self.toPIL = transforms.ToPILImage()
    #     self.transforms_ = transforms_
    #     self.color_jitter_ = color_jitter_
    #
    # def __len__(self):
    #     return len(self.rgb_lines)
    #
    # def __getitem__(self, idx):
    #     rgb_line = self.rgb_lines[idx].strip('\n').split()
    #     # sample_name list里面的视频名称,action_label失真类别
    #     sample_name, action_label, num_frames, dis_level = rgb_line[0], int(rgb_line[1]), int(rgb_line[2]), int(rgb_line[3])
    #     sample_name = sample_name[:-4]
    #     rgb_dir = os.path.join(self.rgb_prefix, sample_name)
    #     sample_rate = random.randint(1, self.max_sr)
    #     start_frame = random.randint(1, num_frames)
    #
    #     rgb_clip = self.loop_load_rgb(rgb_dir, start_frame, sample_rate,
    #                                   self.clip_len, num_frames, sample_name, action_label)
    #
    #     label = sample_rate - 1
    #
    #     trans_clip = self.transforms_(rgb_clip)
    #
    #     ## apply different color jittering for each frame in the video clip
    #     trans_clip_cj = []
    #     for frame in trans_clip:
    #         frame = self.toPIL(frame)  # PIL image
    #         frame = self.color_jitter_(frame)  # tensor [C x H x W]
    #         frame = np.array(frame)
    #         trans_clip_cj.append(frame)
    #
    #     trans_clip_cj = np.array(trans_clip_cj).transpose(3, 0, 1, 2)
    #     rgb_clip = np.array(rgb_clip).transpose(3, 0, 1, 2)
    #     # 输出rgb_clip作为原始的 label可以当成播放速度 action_label失真类别
    #     return trans_clip_cj, label, action_label, rgb_clip, dis_level
    #
    # def loop_load_rgb(self, video_dir, start_frame, sample_rate, clip_len,
    #                   num_frames, sample_name, action_label):
    #
    #     video_clip = []
    #     idx = 0
    #
    #     for i in range(clip_len):
    #         # cur_img_path = os.path.join(
    #         #     video_dir,
    #         #     "frame{:06}.jpg".format(start_frame + idx * sample_rate))
    #         cur_img_path = os.path.join(
    #             video_dir,
    #             "{}_{}_frame{:06}.png".format(sample_name, action_label, start_frame + idx * sample_rate))
    #         img = cv2.imread(cur_img_path)
    #         video_clip.append(img)
    #
    #         if (start_frame + (idx + 1) * sample_rate) > num_frames:
    #             start_frame = 1
    #             idx = 0
    #         else:
    #             idx += 1
    #
    #     video_clip = np.array(video_clip)
    #
    #     return video_clip

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
        # label = sample_rate - 1
        #
        # trans_clip = self.transforms_(rgb_clip)
        #
        # trans_source = self.transforms_(source_clip)
        #
        # ## apply different color jittering for each frame in the video clip
        # trans_clip_cj = []
        # for frame in trans_clip:
        #     frame = self.toPIL(frame)  # PIL image
        #     frame = self.color_jitter_(frame)  # tensor [C x H x W]
        #     frame = np.array(frame)
        #     trans_clip_cj.append(frame)
        #
        # trans_source_clip = []
        # for frame in trans_source:
        #     frame = self.toPIL(frame)
        #     frame = self.color_jitter_(frame)  # tensor [C x H x W]
        #     frame = np.array(frame)
        #     trans_source_clip.append((frame))
        #
        # trans_clip_cj = np.array(trans_clip_cj).transpose(3, 0, 1, 2)
        # trans_source_clip = np.array(trans_source_clip).transpose(3, 0, 1, 2)
        # rgb_clip = np.array(rgb_clip).transpose(3, 0, 1, 2)
        # # 输出rgb_clip作为原始的 label可以当成播放速度 action_label失真类别
        # return trans_clip_cj, label, action_label, rgb_clip, trans_source_clip

        label = sample_rate - 1

        trans_clip1 = self.transforms_(rgb_clip)
        trans_clip2 = self.transforms_(rgb_clip)

        # trans_source = self.transforms_(source_clip)

        # TODO : 去掉颜色抖动
        ## apply different color jittering for each frame in the video clip
        # trans_clip_cj1 = []
        # for frame in trans_clip1:
        #     frame = self.toPIL(frame)  # PIL image
        #     frame = self.color_jitter_(frame)  # tensor [C x H x W]
        #     frame = np.array(frame)
        #     trans_clip_cj1.append(frame)
        # trans_clip_cj2 = []
        # for frame in trans_clip2:
        #     frame = self.toPIL(frame)  # PIL image
        #     frame = self.color_jitter_(frame)  # tensor [C x H x W]
        #     frame = np.array(frame)
        #     trans_clip_cj2.append(frame)
        # trans_source_clip = []
        # for frame in trans_source:
        #     frame = self.toPIL(frame)
        #     frame = self.color_jitter_(frame)  # tensor [C x H x W]
        #     frame = np.array(frame)
        #     trans_source_clip.append((frame))
        #
        # trans_clip_cj1 = np.array(trans_clip_cj1).transpose(3, 0, 1, 2)
        # trans_clip_cj2 = np.array(trans_clip_cj2).transpose(3, 0, 1, 2)
        # trans_source_clip = np.array(trans_source_clip).transpose(3, 0, 1, 2)
        # TODO : 去掉颜色抖动
        trans_clip1 = np.array(trans_clip1).transpose(3, 0, 1, 2)
        trans_clip2 = np.array(trans_clip2).transpose(3, 0, 1, 2)
        # trans_source = np.array(trans_source).transpose(3, 0, 1, 2)

        # TODO : 添加原始12
        trans_source1 = self.transforms_(source_clip)
        trans_source2 = self.transforms_(source_clip)
        trans_source1 = np.array(trans_source1).transpose(3, 0, 1, 2)
        trans_source2 = np.array(trans_source2).transpose(3, 0, 1, 2)

        # TODO

        rgb_clip = np.array(rgb_clip).transpose(3, 0, 1, 2)
        # 输出rgb_clip作为原始的 label可以当成播放速度 action_label失真类别
        # return trans_clip_cj1, trans_clip_cj2, label, action_label, rgb_clip, trans_source_clip
        # return trans_clip1, trans_clip2, label, action_label, rgb_clip, trans_source
        return trans_clip1, trans_clip2, label, action_label, rgb_clip, trans_source1, trans_source2


    # def __init__(self, data_list, rgb_prefix, source_list, source_prefix, clip_len,  max_sr, transforms_=None, color_jitter_=None): # yapf: disable:
    #     lines = open(data_list)
    #     self.rgb_lines = list(lines) * 10
    #     self.rgb_prefix = rgb_prefix
    #
    #     org_lines = open(source_list)
    #     self.rgb_org_lines = list(org_lines) * 10
    #     self.source_prefix = source_prefix
    #
    #     self.clip_len = clip_len
    #     self.max_sr = max_sr
    #     self.toPIL = transforms.ToPILImage()
    #     self.transforms_ = transforms_
    #     self.color_jitter_ = color_jitter_
    #
    # def __len__(self):
    #     return len(self.rgb_lines)
    #
    # def __getitem__(self, idx):
    #
    #     rgb_line = self.rgb_lines[idx].strip('\n').split()
    #     rgb_org_line = self.rgb_org_lines[idx].strip('\n').split()
    #     # print('idx is ', idx)
    #     # sample_name list里面的视频名称,action_label失真类别
    #     # sample_name, action_label, num_frames = rgb_line[0], int(rgb_line[1]), int(rgb_line[2])
    #     # 多添加失真程度
    #     sample_name, action_label, num_frames = rgb_line[0], int(rgb_line[1]), int(rgb_line[2])
    #
    #     sample_name = sample_name[:-4]
    #     rgb_dir = os.path.join(self.rgb_prefix, sample_name)
    #
    #     sample_org_name, num_frames = rgb_org_line[0], int(rgb_org_line[2])
    #     sample_org_name = sample_org_name[:-4]
    #     rgb_org_dir = os.path.join(self.source_prefix, sample_org_name)
    #
    #     sample_rate = random.randint(1, self.max_sr)
    #     start_frame = random.randint(1, num_frames)
    #
    #     rgb_clip = self.loop_load_rgb(rgb_dir, start_frame, sample_rate,
    #                                   self.clip_len, num_frames, sample_name, action_label)
    #     rgb_org_clip = self.loop_load_org_rgb(rgb_org_dir, start_frame, sample_rate,
    #                                   self.clip_len, num_frames, sample_org_name, action_label)
    #
    #     label = sample_rate - 1
    #
    #     trans_clip = self.transforms_(rgb_clip)
    #     ## apply different color jittering for each frame in the video clip
    #     trans_clip_cj = []
    #     for frame in trans_clip:
    #         frame = self.toPIL(frame)  # PIL image
    #         frame = self.color_jitter_(frame)  # tensor [C x H x W]
    #         frame = np.array(frame)
    #         trans_clip_cj.append(frame)
    #
    #     trans_org_clip = self.transforms_(rgb_org_clip)
    #     trans_org_clip_cj = []
    #     for frame in trans_org_clip:
    #         frame = self.toPIL(frame)  # PIL image
    #         frame = self.color_jitter_(frame)  # tensor [C x H x W]
    #         frame = np.array(frame)
    #         trans_org_clip_cj.append(frame)
    #
    #
    #     trans_clip_cj = np.array(trans_clip_cj).transpose(3, 0, 1, 2)
    #     trans_org_clip_cj = np.array(trans_org_clip_cj).transpose(3, 0, 1, 2)
    #     rgb_clip = np.array(rgb_clip).transpose(3, 0, 1, 2)
    #     # 输出rgb_clip作为原始的 label可以当成播放速度 action_label失真类别
    #     # return trans_clip_cj, label, action_label, rgb_clip, trans_org_clip_cj
    #     # 多添加dis_level 失真程度
    #     return trans_clip_cj, label, action_label, rgb_clip, trans_org_clip_cj

    def loop_load_rgb(self, video_dir, start_frame, sample_rate, clip_len,
                      num_frames, sample_name, action_label):

        video_clip = []
        idx = 0

        for i in range(clip_len):
            cur_img_path = os.path.join(
                "/", video_dir, "{}_{}_frame{:06}.png".format(sample_name, action_label, start_frame + idx * sample_rate))

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
                video_dir, "{}_5_frame{:06}.png".format(sample_name, start_frame + idx * sample_rate))

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
                "{}_5_frame{:06}.png".format(sample_name, start_frame + idx * sample_rate))

            img = cv2.imread(cur_img_path)
            video_clip.append(img)

            if (start_frame + (idx + 1) * sample_rate) > num_frames:
                start_frame = 1
                idx = 0
            else:
                idx += 1

        video_clip = np.array(video_clip)

        return video_clip

# if __name__ == '__main__':
#
#     data_list = '/data/wst/video-pace/datasets/zice/datadis.list'
#     rgb_prefix = '/data/wst/video-pace/data/'
#     # data_list = '/data/wst/video-pace/newVPimgdata.list'
#     # rgb_prefix = '/data/wst/newVPimg/'
#     # data_list = '/data/wst/video-pace/list/train_ucf101_split2.list'
#     # rgb_prefix = '/data/wst/pacedata/jpegs_256/'
#     source_list = '/data/wst/video-pace/datasets/zice/datasource.list'
#     source_prefix = '/data/wst/video-pace/data1/'
#
#     transforms_ = transforms.Compose([
#         ClipResize((128,171)),
#         RandomCrop(16),
#         RandomHorizontalFlip(0.5)
#     ])
#
#     color_jitter = transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
#     rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
#
#     train_dataset = mydata_pace_pretrain3(data_list, rgb_prefix, source_list, source_prefix, clip_len=16, max_sr=4,
#                                    transforms_=transforms_, color_jitter_=rnd_color_jitter)
#
#     dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
#
#     for iter, sample in enumerate(dataloader):
#
#         rgb_clip, label = sample
#         print(rgb_clip.shape)
#         rgb_clip = rgb_clip[0].numpy()
#
#         print(rgb_clip.shape)
#
#         rgb_clip = rgb_clip.transpose(1, 2, 3, 0)
#         for i in range(len(rgb_clip)):
#             cur_frame = rgb_clip[i]
#
#             cv2.imshow("img", cur_frame)
#             cv2.waitKey()
