#------------------------------------------------------------------------------
#   Libraries
#------------------------------------------------------------------------------
import warnings
warnings.filterwarnings('ignore')

import cv2, os
import numpy as np
from random import shuffle

import torch
from torch.utils.data import Dataset, DataLoader

from dataloaders import transforms
# import transforms

# labels_info = [
#     {"hasInstances": False, "category": "void", "catid": 0, "name": "unlabeled", "ignoreInEval": True, "id": 0, "color": [0, 0, 0], "trainId": 0},
#     {"hasInstances": False, "category": "void", "catid": 0, "name": "ego vehicle", "ignoreInEval": True, "id": 1, "color": [0, 0, 0], "trainId": 0},
#     {"hasInstances": False, "category": "void", "catid": 0, "name": "rectification border", "ignoreInEval": True, "id": 2, "color": [0, 0, 0], "trainId": 0},
#     {"hasInstances": False, "category": "void", "catid": 0, "name": "out of roi", "ignoreInEval": True, "id": 3, "color": [0, 0, 0], "trainId": 0},
#     {"hasInstances": False, "category": "void", "catid": 0, "name": "static", "ignoreInEval": True, "id": 4, "color": [0, 0, 0], "trainId": 0},
#     {"hasInstances": False, "category": "void", "catid": 0, "name": "dynamic", "ignoreInEval": True, "id": 5, "color": [111, 74, 0], "trainId": 1},
#     {"hasInstances": False, "category": "void", "catid": 0, "name": "ground", "ignoreInEval": True, "id": 6, "color": [81, 0, 81], "trainId": 0},
#     {"hasInstances": False, "category": "flat", "catid": 1, "name": "road", "ignoreInEval": False, "id": 7, "color": [128, 64, 128], "trainId": 0},
#     {"hasInstances": False, "category": "flat", "catid": 1, "name": "sidewalk", "ignoreInEval": False, "id": 8, "color": [244, 35, 232], "trainId": 0},
#     {"hasInstances": False, "category": "flat", "catid": 1, "name": "parking", "ignoreInEval": True, "id": 9, "color": [250, 170, 160], "trainId": 0},
#     {"hasInstances": False, "category": "flat", "catid": 1, "name": "rail track", "ignoreInEval": True, "id": 10, "color": [230, 150, 140], "trainId": 0},
#     {"hasInstances": False, "category": "construction", "catid": 2, "name": "building", "ignoreInEval": False, "id": 11, "color": [70, 70, 70], "trainId": 0},
#     {"hasInstances": False, "category": "construction", "catid": 2, "name": "wall", "ignoreInEval": False, "id": 12, "color": [102, 102, 156], "trainId": 0},
#     {"hasInstances": False, "category": "construction", "catid": 2, "name": "fence", "ignoreInEval": False, "id": 13, "color": [190, 153, 153], "trainId": 0},
#     {"hasInstances": False, "category": "construction", "catid": 2, "name": "guard rail", "ignoreInEval": True, "id": 14, "color": [180, 165, 180], "trainId": 0},
#     {"hasInstances": False, "category": "construction", "catid": 2, "name": "bridge", "ignoreInEval": True, "id": 15, "color": [150, 100, 100], "trainId": 0},
#     {"hasInstances": False, "category": "construction", "catid": 2, "name": "tunnel", "ignoreInEval": True, "id": 16, "color": [150, 120, 90], "trainId": 0},
#     {"hasInstances": False, "category": "object", "catid": 3, "name": "pole", "ignoreInEval": False, "id": 17, "color": [153, 153, 153], "trainId": 0},
#     {"hasInstances": False, "category": "object", "catid": 3, "name": "polegroup", "ignoreInEval": True, "id": 18, "color": [153, 153, 153], "trainId": 0},
#     {"hasInstances": False, "category": "object", "catid": 3, "name": "traffic light", "ignoreInEval": False, "id": 19, "color": [250, 170, 30], "trainId": 0},
#     {"hasInstances": False, "category": "object", "catid": 3, "name": "traffic sign", "ignoreInEval": False, "id": 20, "color": [220, 220, 0], "trainId": 0},
#     {"hasInstances": False, "category": "nature", "catid": 4, "name": "vegetation", "ignoreInEval": False, "id": 21, "color": [107, 142, 35], "trainId": 0},
#     {"hasInstances": False, "category": "nature", "catid": 4, "name": "terrain", "ignoreInEval": False, "id": 22, "color": [152, 251, 152], "trainId": 0},
#     {"hasInstances": False, "category": "sky", "catid": 5, "name": "sky", "ignoreInEval": False, "id": 23, "color": [70, 130, 180], "trainId": 0},
#     {"hasInstances": True, "category": "human", "catid": 6, "name": "person", "ignoreInEval": False, "id": 24, "color": [220, 20, 60], "trainId": 1},
#     {"hasInstances": True, "category": "human", "catid": 6, "name": "rider", "ignoreInEval": False, "id": 25, "color": [255, 0, 0], "trainId": 1},
#     {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "car", "ignoreInEval": False, "id": 26, "color": [0, 0, 142], "trainId": 1},
#     {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "truck", "ignoreInEval": False, "id": 27, "color": [0, 0, 70], "trainId": 1},
#     {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "bus", "ignoreInEval": False, "id": 28, "color": [0, 60, 100], "trainId": 1},
#     {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "caravan", "ignoreInEval": True, "id": 29, "color": [0, 0, 90], "trainId": 1},
#     {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "trailer", "ignoreInEval": True, "id": 30, "color": [0, 0, 110], "trainId": 1},
#     {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "train", "ignoreInEval": False, "id": 31, "color": [0, 80, 100], "trainId": 1},
#     {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "motorcycle", "ignoreInEval": False, "id": 32, "color": [0, 0, 230], "trainId": 1},
#     {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "bicycle", "ignoreInEval": False, "id": 33, "color": [119, 11, 32], "trainId": 1},
#     {"hasInstances": False, "category": "vehicle", "catid": 7, "name": "license plate", "ignoreInEval": True, "id": -1, "color": [0, 0, 142], "trainId": -1}
# ]


labels_info = [
   {'name': 'unlabeled', 'ignoreInEval': False, 'id': 0, 'trainId': 255, 'color': [0, 0, 0]},
{'name': 'person', 'ignoreInEval': False, 'id': 1, 'trainId': 1, 'color': (215, 0, 0)},
{'name': 'bicycle', 'ignoreInEval': False, 'id': 2, 'trainId': 1, 'color': (140, 60, 255)},
{'name': 'car', 'ignoreInEval': False, 'id': 3, 'trainId': 1, 'color': (2, 136, 0)},
{'name': 'motorcycle', 'ignoreInEval': False, 'id': 4, 'trainId': 1, 'color': (0, 172, 199)},
{'name': 'airplane', 'ignoreInEval': False, 'id': 5, 'trainId': 255, 'color': [0, 0, 0]},
{'name': 'bus', 'ignoreInEval': False, 'id': 6, 'trainId': 1, 'color': (152, 255, 0)},
{'name': 'train', 'ignoreInEval': False, 'id': 7, 'trainId': 255, 'color': [0, 0, 0]},
{'name': 'truck', 'ignoreInEval': False, 'id': 8, 'trainId': 1, 'color': (255, 127, 209)},
{'name': 'boat', 'ignoreInEval': False, 'id': 9, 'trainId': 255, 'color': [0, 0, 0]},
{'name': 'traffic', 'ignoreInEval': False, 'id': 10, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'fire', 'ignoreInEval': False, 'id': 11, 'trainId': 255, 'color': [0, 0, 0]},
{'name': 'street', 'ignoreInEval': False, 'id': 12, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'stop', 'ignoreInEval': False, 'id': 13, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'parking', 'ignoreInEval': False, 'id': 14, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'bench', 'ignoreInEval': False, 'id': 15, 'trainId': 0, 'color': (108, 0, 79)},
{'name': 'bird', 'ignoreInEval': False, 'id': 16, 'trainId': 1, 'color': (255, 165, 48)},
{'name': 'cat', 'ignoreInEval': False, 'id': 17, 'trainId': 1, 'color': (0, 0, 157)},
{'name': 'dog', 'ignoreInEval': False, 'id': 18, 'trainId': 1, 'color': (134, 112, 104)},
{'name': 'horse', 'ignoreInEval': False, 'id': 19, 'trainId': 1, 'color': [0, 0, 0]},
{'name': 'sheep', 'ignoreInEval': False, 'id': 20, 'trainId': 1, 'color': [0, 0, 0]},
{'name': 'cow', 'ignoreInEval': False, 'id': 21, 'trainId': 1, 'color': [0, 0, 0]},
{'name': 'elephant', 'ignoreInEval': False, 'id': 22, 'trainId': 1, 'color': [0, 0, 0]},
{'name': 'bear', 'ignoreInEval': False, 'id': 23, 'trainId': 1, 'color': [0, 0, 0]},
{'name': 'zebra', 'ignoreInEval': False, 'id': 24, 'trainId': 1, 'color': [0, 0, 0]},
{'name': 'giraffe', 'ignoreInEval': False, 'id': 25, 'trainId': 1, 'color': [0, 0, 0]},
{'name': 'hat', 'ignoreInEval': False, 'id': 26, 'trainId': 255, 'color': [0, 0, 0]},
{'name': 'backpack', 'ignoreInEval': False, 'id': 27, 'trainId': 255, 'color': [0, 0, 0]},
{'name': 'umbrella', 'ignoreInEval': False, 'id': 28, 'trainId': 255, 'color': [0, 0, 0]},
{'name': 'shoe', 'ignoreInEval': False, 'id': 29, 'trainId': 255, 'color': [0, 0, 0]},
{'name': 'eye', 'ignoreInEval': False, 'id': 30, 'trainId': 255, 'color': [0, 0, 0]},
{'name': 'handbag', 'ignoreInEval': False, 'id': 31, 'trainId': 255, 'color': [0, 0, 0]},
{'name': 'tie', 'ignoreInEval': False, 'id': 32, 'trainId': 1, 'color': [0, 0, 0]},
{'name': 'suitcase', 'ignoreInEval': False, 'id': 33, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'frisbee', 'ignoreInEval': False, 'id': 34, 'trainId': 1, 'color': [0, 0, 0]},
{'name': 'skis', 'ignoreInEval': False, 'id': 35, 'trainId': 1, 'color': [0, 0, 0]},
{'name': 'snowboard', 'ignoreInEval': False, 'id': 36, 'trainId': 255, 'color': [0, 0, 0]},
{'name': 'sports', 'ignoreInEval': False, 'id': 37, 'trainId': 255, 'color': [0, 0, 0]},
{'name': 'kite', 'ignoreInEval': False, 'id': 38, 'trainId': 255, 'color': [0, 0, 0]},
{'name': 'baseball', 'ignoreInEval': False, 'id': 39, 'trainId': 255, 'color': [0, 0, 0]},
{'name': 'baseball', 'ignoreInEval': False, 'id': 40, 'trainId': 255, 'color': [0, 0, 0]},
{'name': 'skateboard', 'ignoreInEval': False, 'id': 41, 'trainId': 255, 'color': [0, 0, 0]},
{'name': 'surfboard', 'ignoreInEval': False, 'id': 42, 'trainId': 255, 'color': [0, 0, 0]},
{'name': 'tennis', 'ignoreInEval': False, 'id': 43, 'trainId': 255, 'color': [0, 0, 0]},
{'name': 'bottle', 'ignoreInEval': False, 'id': 44, 'trainId': 0, 'color': (0, 253, 207)},
{'name': 'plate', 'ignoreInEval': False, 'id': 45, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'wine', 'ignoreInEval': False, 'id': 46, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'cup', 'ignoreInEval': False, 'id': 47, 'trainId': 0, 'color': (188, 183, 255)},
{'name': 'fork', 'ignoreInEval': False, 'id': 48, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'knife', 'ignoreInEval': False, 'id': 49, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'spoon', 'ignoreInEval': False, 'id': 50, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'bowl', 'ignoreInEval': False, 'id': 51, 'trainId': 255, 'color': [0, 0, 0]},
{'name': 'banana', 'ignoreInEval': False, 'id': 52, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'apple', 'ignoreInEval': False, 'id': 53, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'sandwich', 'ignoreInEval': False, 'id': 54, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'orange', 'ignoreInEval': False, 'id': 55, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'broccoli', 'ignoreInEval': False, 'id': 56, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'carrot', 'ignoreInEval': False, 'id': 57, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'hot', 'ignoreInEval': False, 'id': 58, 'trainId': 255, 'color': [0, 0, 0]},
{'name': 'pizza', 'ignoreInEval': False, 'id': 59, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'donut', 'ignoreInEval': False, 'id': 60, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'cake', 'ignoreInEval': False, 'id': 61, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'chair', 'ignoreInEval': False, 'id': 62, 'trainId': 0, 'color': (0, 73, 66)},
{'name': 'couch', 'ignoreInEval': False, 'id': 63, 'trainId': 0, 'color': (220, 179, 175)},
{'name': 'potted', 'ignoreInEval': False, 'id': 64, 'trainId': 255, 'color': [0, 0, 0]},
{'name': 'bed', 'ignoreInEval': False, 'id': 65, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'mirror', 'ignoreInEval': False, 'id': 66, 'trainId': 255, 'color': [0, 0, 0]},
{'name': 'dining', 'ignoreInEval': False, 'id': 67, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'window', 'ignoreInEval': False, 'id': 68, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'desk', 'ignoreInEval': False, 'id': 69, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'toilet', 'ignoreInEval': False, 'id': 70, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'door', 'ignoreInEval': False, 'id': 71, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'tv', 'ignoreInEval': False, 'id': 72, 'trainId': 0, 'color': (79, 42, 0)},
{'name': 'laptop', 'ignoreInEval': False, 'id': 73, 'trainId': 0, 'color': (149, 180, 122)},
{'name': 'mouse', 'ignoreInEval': False, 'id': 74, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'remote', 'ignoreInEval': False, 'id': 75, 'trainId': 255, 'color': [0, 0, 0]},
{'name': 'keyboard', 'ignoreInEval': False, 'id': 76, 'trainId': 0, 'color': (192, 4, 185)},
{'name': 'cell', 'ignoreInEval': False, 'id': 77, 'trainId': 255, 'color': [0, 0, 0]},
{'name': 'microwave', 'ignoreInEval': False, 'id': 78, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'oven', 'ignoreInEval': False, 'id': 79, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'toaster', 'ignoreInEval': False, 'id': 80, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'sink', 'ignoreInEval': False, 'id': 81, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'refrigerator', 'ignoreInEval': False, 'id': 82, 'trainId': 0, 'color': (37, 102, 162)},
{'name': 'blender', 'ignoreInEval': False, 'id': 83, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'book', 'ignoreInEval': False, 'id': 84, 'trainId': 0, 'color': (40, 0, 65)},
{'name': 'clock', 'ignoreInEval': False, 'id': 85, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'vase', 'ignoreInEval': False, 'id': 86, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'scissors', 'ignoreInEval': False, 'id': 87, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'teddy', 'ignoreInEval': False, 'id': 88, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'hair', 'ignoreInEval': False, 'id': 89, 'trainId': 1, 'color': [0, 0, 0]},
{'name': 'toothbrush', 'ignoreInEval': False, 'id': 90, 'trainId': 255, 'color': [0, 0, 0]},
{'name': 'hair', 'ignoreInEval': False, 'id': 91, 'trainId': 255, 'color': [0, 0, 0]},
{'name': 'banner', 'ignoreInEval': False, 'id': 92, 'trainId': 255, 'color': [0, 0, 0]},
{'name': 'blanket', 'ignoreInEval': False, 'id': 93, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'branch', 'ignoreInEval': False, 'id': 94, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'bridge', 'ignoreInEval': False, 'id': 95, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'building', 'ignoreInEval': False, 'id': 96, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'bush', 'ignoreInEval': False, 'id': 97, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'cabinet', 'ignoreInEval': False, 'id': 98, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'cage', 'ignoreInEval': False, 'id': 99, 'trainId': 255, 'color': [0, 0, 0]},
{'name': 'cardboard', 'ignoreInEval': False, 'id': 100, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'carpet', 'ignoreInEval': False, 'id': 101, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'ceiling', 'ignoreInEval': False, 'id': 102, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'ceiling', 'ignoreInEval': False, 'id': 103, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'cloth', 'ignoreInEval': False, 'id': 104, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'clothes', 'ignoreInEval': False, 'id': 105, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'clouds', 'ignoreInEval': False, 'id': 106, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'counter', 'ignoreInEval': False, 'id': 107, 'trainId': 255, 'color': [0, 0, 0]},
{'name': 'cupboard', 'ignoreInEval': False, 'id': 108, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'curtain', 'ignoreInEval': False, 'id': 109, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'desk', 'ignoreInEval': False, 'id': 110, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'dirt', 'ignoreInEval': False, 'id': 111, 'trainId': 255, 'color': [0, 0, 0]},
{'name': 'door', 'ignoreInEval': False, 'id': 112, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'fence', 'ignoreInEval': False, 'id': 113, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'floor', 'ignoreInEval': False, 'id': 114, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'floor', 'ignoreInEval': False, 'id': 115, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'floor', 'ignoreInEval': False, 'id': 116, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'floor', 'ignoreInEval': False, 'id': 117, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'floor', 'ignoreInEval': False, 'id': 118, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'flower', 'ignoreInEval': False, 'id': 119, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'fog', 'ignoreInEval': False, 'id': 120, 'trainId': 255, 'color': [0, 0, 0]},
{'name': 'food', 'ignoreInEval': False, 'id': 121, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'fruit', 'ignoreInEval': False, 'id': 122, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'furniture', 'ignoreInEval': False, 'id': 123, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'grass', 'ignoreInEval': False, 'id': 124, 'trainId': 255, 'color': [0, 0, 0]},
{'name': 'gravel', 'ignoreInEval': False, 'id': 125, 'trainId': 255, 'color': [0, 0, 0]},
{'name': 'ground', 'ignoreInEval': False, 'id': 126, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'hill', 'ignoreInEval': False, 'id': 127, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'house', 'ignoreInEval': False, 'id': 128, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'leaves', 'ignoreInEval': False, 'id': 129, 'trainId': 255, 'color': [0, 0, 0]},
{'name': 'light', 'ignoreInEval': False, 'id': 130, 'trainId': 255, 'color': [0, 0, 0]},
{'name': 'mat', 'ignoreInEval': False, 'id': 131, 'trainId': 255, 'color': [0, 0, 0]},
{'name': 'metal', 'ignoreInEval': False, 'id': 132, 'trainId': 255, 'color': [0, 0, 0]},
{'name': 'mirror', 'ignoreInEval': False, 'id': 133, 'trainId': 255, 'color': [0, 0, 0]},
{'name': 'moss', 'ignoreInEval': False, 'id': 134, 'trainId': 255, 'color': [0, 0, 0]},
{'name': 'mountain', 'ignoreInEval': False, 'id': 135, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'mud', 'ignoreInEval': False, 'id': 136, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'napkin', 'ignoreInEval': False, 'id': 137, 'trainId': 255, 'color': [0, 0, 0]},
{'name': 'net', 'ignoreInEval': False, 'id': 138, 'trainId': 255, 'color': [0, 0, 0]},
{'name': 'paper', 'ignoreInEval': False, 'id': 139, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'pavement', 'ignoreInEval': False, 'id': 140, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'pillow', 'ignoreInEval': False, 'id': 141, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'plant', 'ignoreInEval': False, 'id': 142, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'plastic', 'ignoreInEval': False, 'id': 143, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'platform', 'ignoreInEval': False, 'id': 144, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'playingfield', 'ignoreInEval': False, 'id': 145, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'railing', 'ignoreInEval': False, 'id': 146, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'railroad', 'ignoreInEval': False, 'id': 147, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'river', 'ignoreInEval': False, 'id': 148, 'trainId': 255, 'color': [0, 0, 0]},
{'name': 'road', 'ignoreInEval': False, 'id': 149, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'rock', 'ignoreInEval': False, 'id': 150, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'roof', 'ignoreInEval': False, 'id': 151, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'rug', 'ignoreInEval': False, 'id': 152, 'trainId': 255, 'color': [0, 0, 0]},
{'name': 'salad', 'ignoreInEval': False, 'id': 153, 'trainId': 255, 'color': [0, 0, 0]},
{'name': 'sand', 'ignoreInEval': False, 'id': 154, 'trainId': 255, 'color': [0, 0, 0]},
{'name': 'sea', 'ignoreInEval': False, 'id': 155, 'trainId': 255, 'color': [0, 0, 0]},
{'name': 'shelf', 'ignoreInEval': False, 'id': 156, 'trainId': 255, 'color': [0, 0, 0]},
{'name': 'sky', 'ignoreInEval': False, 'id': 157, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'skyscraper', 'ignoreInEval': False, 'id': 158, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'snow', 'ignoreInEval': False, 'id': 159, 'trainId': 255, 'color': [0, 0, 0]},
{'name': 'solid', 'ignoreInEval': False, 'id': 160, 'trainId': 255, 'color': [0, 0, 0]},
{'name': 'stairs', 'ignoreInEval': False, 'id': 161, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'stone', 'ignoreInEval': False, 'id': 162, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'straw', 'ignoreInEval': False, 'id': 163, 'trainId': 255, 'color': [0, 0, 0]},
{'name': 'structural', 'ignoreInEval': False, 'id': 164, 'trainId': 255, 'color': [0, 0, 0]},
{'name': 'table', 'ignoreInEval': False, 'id': 165, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'tent', 'ignoreInEval': False, 'id': 166, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'textile', 'ignoreInEval': False, 'id': 167, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'towel', 'ignoreInEval': False, 'id': 168, 'trainId': 255, 'color': [0, 0, 0]},
{'name': 'tree', 'ignoreInEval': False, 'id': 169, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'vegetable', 'ignoreInEval': False, 'id': 170, 'trainId': 255, 'color': [0, 0, 0]},
{'name': 'wall', 'ignoreInEval': False, 'id': 171, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'wall', 'ignoreInEval': False, 'id': 172, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'wall', 'ignoreInEval': False, 'id': 173, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'wall', 'ignoreInEval': False, 'id': 174, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'wall', 'ignoreInEval': False, 'id': 175, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'wall', 'ignoreInEval': False, 'id': 176, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'wall', 'ignoreInEval': False, 'id': 177, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'water', 'ignoreInEval': False, 'id': 178, 'trainId': 255, 'color': [0, 0, 0]},
{'name': 'waterdrops', 'ignoreInEval': False, 'id': 179, 'trainId': 255, 'color': [0, 0, 0]},
{'name': 'window', 'ignoreInEval': False, 'id': 180, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'window', 'ignoreInEval': False, 'id': 181, 'trainId': 0, 'color': [0, 0, 0]},
{'name': 'wood', 'ignoreInEval': False, 'id': 182, 'trainId': 0, 'color': [0, 0, 0]}
]


#------------------------------------------------------------------------------
#	DataLoader for Semantic Segmentation
#------------------------------------------------------------------------------
class SegmentationDataLoader(object):
	def __init__(self, pairs_file, color_channel="RGB", resize=224, padding_value=0,
				crop_range=[0.75, 1.0], flip_hor=0.5, rotate=0.3, angle=10, noise_std=5,
				normalize=True, one_hot=False, is_training=True,
				shuffle=True, batch_size=1, n_workers=1, pin_memory=True):

		# Storage parameters
		super(SegmentationDataLoader, self).__init__()
		self.pairs_file = pairs_file
		self.color_channel = color_channel
		self.resize = resize
		self.padding_value = padding_value
		self.crop_range = crop_range
		self.flip_hor = flip_hor
		self.rotate = rotate
		self.angle = angle
		self.noise_std = noise_std
		self.normalize = normalize
		self.one_hot = one_hot
		self.is_training = is_training
		self.shuffle = shuffle
		self.batch_size = batch_size
		self.n_workers = n_workers
		self.pin_memory = pin_memory

		# Dataset
		self.dataset = SegmentationDataset(
			pairs_file=self.pairs_file,
			color_channel=self.color_channel,
			resize=self.resize,
			padding_value=self.padding_value,
			crop_range=self.crop_range,
			flip_hor=self.flip_hor,
			rotate=self.rotate,
			angle=self.angle,
			noise_std=self.noise_std,
			normalize=self.normalize,
			one_hot=self.one_hot,
			is_training=self.is_training,
		)

	@property
	def loader(self):
		return DataLoader(
			self.dataset,
			batch_size=self.batch_size,
			shuffle=self.shuffle,
			num_workers=self.n_workers,
			pin_memory=self.pin_memory,
		)


#------------------------------------------------------------------------------
#	Dataset for Semantic Segmentation
#------------------------------------------------------------------------------
class SegmentationDataset(Dataset):
	"""
	The dataset requires label is a grayscale image with value {0,1,...,C-1},
	where C is the number of classes.
	"""
	def __init__(self, pairs_file, color_channel="RGB", resize=512, padding_value=0,
		is_training=True, noise_std=5, crop_range=[0.75, 1.0], flip_hor=0.5, rotate=0.3, angle=10,
		one_hot=False, normalize=True, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]):

		# Get list of image and label files
		self.image_files, self.label_files = [], []
		fp = open(pairs_file, "r")
		lines = fp.read().split("\n")
		lines = [line.strip() for line in lines if len(line)]
		lines = [line.split(", ") for line in lines]

		#specific for cityscape
		self.n_cats = 2
		self.lb_ignore = 255
		self.lb_map = np.arange(256).astype(np.uint8)
    # lb_map = [0 ,1,2,... 255]
		for el in labels_info:
			self.lb_map[el['id']] = el['trainId']

		print("[Dataset] Checking file paths...")
		error_flg = False
		for line in lines:
			image_file, label_file = line
			if not os.path.exists(image_file):
				print("%s does not exist!" % (image_file))
				error_flg = True
			if not os.path.exists(label_file):
				print("%s does not exist!" % (label_file))
				error_flg = True
			self.image_files.append(image_file)
			self.label_files.append(label_file)
		if error_flg:
			raise ValueError("Some file paths are corrupted! Please re-check your file paths!")
		print("[Dataset] Number of sample pairs:", len(self.image_files))

		# Parameters
		self.color_channel = color_channel
		self.resize = resize
		self.padding_value = padding_value
		self.is_training = is_training
		self.noise_std = noise_std
		self.crop_range = crop_range
		self.flip_hor = flip_hor
		self.rotate = rotate
		self.angle = angle
		self.one_hot = one_hot
		self.normalize = normalize
		self.mean = np.array(mean)[None,None,:]
		self.std = np.array(std)[None,None,:]

	def __len__(self):
		return len(self.image_files)

	def __getitem__(self, idx):
		# Read image and label
		img_file, label_file = self.image_files[idx], self.label_files[idx]
		image = cv2.imread(img_file)[...,::-1]
		label = cv2.imread(label_file, 0)
    
		#chnagethe id to trainid if any
		label = self.lb_map[label]


		# Augmentation if in training phase
		if self.is_training:
			image = transforms.random_noise(image, std=self.noise_std)
			image, label = transforms.flip_horizon(image, label, self.flip_hor)
			image, label = transforms.rotate_90(image, label, self.rotate)
			image, label = transforms.rotate_angle(image, label, self.angle)
			image, label = transforms.random_crop(image, label, self.crop_range)

		# Resize: the greater side is refered, the rest is padded
		image = transforms.resize_image(image, expected_size=self.resize, pad_value=self.padding_value, mode=cv2.INTER_LINEAR)
		label = transforms.resize_image(label, expected_size=self.resize, pad_value=self.padding_value, mode=cv2.INTER_NEAREST)

		# Preprocess image
		if self.normalize:
			image = image.astype(np.float32) / 255.0
			image = (image - self.mean) / self.std
		image = np.transpose(image, axes=(2, 0, 1))

		# Preprocess label
		# label[label>0] = 1 #here for every pixel, if the value is greater than 0 then replace it with 1
		# if self.one_hot:
		# 	label = (np.arange(label.max()+1) == label[...,None]).astype(int)
		label[label < 0] = 255
		label[label > (self.n_cats - 1)] = 255

		# Convert to tensor and return
		image = torch.tensor(image.copy(), dtype=torch.float32)
		label = torch.tensor(label.copy(), dtype=torch.float32)

		return image, label