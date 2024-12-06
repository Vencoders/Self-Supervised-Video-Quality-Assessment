# import os
# import cv2
#
# # img_path = '/data/wst/pacedata/jpegs_256/v_SkyDiving_g13_c04/frame000079.jpg'
# img_path = '/data/wst/adatabase/test/1_-1d-rJG-0jI.mp4'
#
# if not os.path.isfile(img_path):
#     print('Image file not found!')
# else:
#     img = cv2.imread(img_path)
#     if img is None:
#         print('Failed to read image file!')
#     else:
#         # do something with the image
#         print('ok')


import os

file_path = '/data/wst/newVPimg/3_video200/3_video200_4_frame000531.png'  # 文件路径

if os.path.exists(file_path):
    print('文件存在')
else:
    print('文件不存在')


# import json
#
# with open('/data/wst/SelfSupervisedVQA/Self-Supervised Representation Learning for Video Quality Assessment/Self-Supervised Representation Learning for Video Quality Assessment/database/VQA/pre_train/youtube8M_0.8_splitByContent.json') as f:
#     data = json.load(f)
#     for key in data.keys():
#         print(key)

# 给文件重新命名
