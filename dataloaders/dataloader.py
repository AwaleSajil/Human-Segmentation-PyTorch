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

labels_info = [
    {"hasInstances": False, "category": "void", "catid": 0, "name": "unlabeled", "ignoreInEval": True, "id": 0, "color": [0, 0, 0], "trainId": 255},
    {"hasInstances": False, "category": "void", "catid": 0, "name": "ego vehicle", "ignoreInEval": True, "id": 1, "color": [0, 0, 0], "trainId": 255},
    {"hasInstances": False, "category": "void", "catid": 0, "name": "rectification border", "ignoreInEval": True, "id": 2, "color": [0, 0, 0], "trainId": 255},
    {"hasInstances": False, "category": "void", "catid": 0, "name": "out of roi", "ignoreInEval": True, "id": 3, "color": [0, 0, 0], "trainId": 255},
    {"hasInstances": False, "category": "void", "catid": 0, "name": "static", "ignoreInEval": True, "id": 4, "color": [0, 0, 0], "trainId": 255},
    {"hasInstances": False, "category": "void", "catid": 0, "name": "dynamic", "ignoreInEval": True, "id": 5, "color": [111, 74, 0], "trainId": 255},
    {"hasInstances": False, "category": "void", "catid": 0, "name": "ground", "ignoreInEval": True, "id": 6, "color": [81, 0, 81], "trainId": 255},
    {"hasInstances": False, "category": "flat", "catid": 1, "name": "road", "ignoreInEval": False, "id": 7, "color": [128, 64, 128], "trainId": 0},
    {"hasInstances": False, "category": "flat", "catid": 1, "name": "sidewalk", "ignoreInEval": False, "id": 8, "color": [244, 35, 232], "trainId": 1},
    {"hasInstances": False, "category": "flat", "catid": 1, "name": "parking", "ignoreInEval": True, "id": 9, "color": [250, 170, 160], "trainId": 255},
    {"hasInstances": False, "category": "flat", "catid": 1, "name": "rail track", "ignoreInEval": True, "id": 10, "color": [230, 150, 140], "trainId": 255},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "building", "ignoreInEval": False, "id": 11, "color": [70, 70, 70], "trainId": 2},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "wall", "ignoreInEval": False, "id": 12, "color": [102, 102, 156], "trainId": 3},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "fence", "ignoreInEval": False, "id": 13, "color": [190, 153, 153], "trainId": 4},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "guard rail", "ignoreInEval": True, "id": 14, "color": [180, 165, 180], "trainId": 255},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "bridge", "ignoreInEval": True, "id": 15, "color": [150, 100, 100], "trainId": 255},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "tunnel", "ignoreInEval": True, "id": 16, "color": [150, 120, 90], "trainId": 255},
    {"hasInstances": False, "category": "object", "catid": 3, "name": "pole", "ignoreInEval": False, "id": 17, "color": [153, 153, 153], "trainId": 5},
    {"hasInstances": False, "category": "object", "catid": 3, "name": "polegroup", "ignoreInEval": True, "id": 18, "color": [153, 153, 153], "trainId": 255},
    {"hasInstances": False, "category": "object", "catid": 3, "name": "traffic light", "ignoreInEval": False, "id": 19, "color": [250, 170, 30], "trainId": 6},
    {"hasInstances": False, "category": "object", "catid": 3, "name": "traffic sign", "ignoreInEval": False, "id": 20, "color": [220, 220, 0], "trainId": 7},
    {"hasInstances": False, "category": "nature", "catid": 4, "name": "vegetation", "ignoreInEval": False, "id": 21, "color": [107, 142, 35], "trainId": 8},
    {"hasInstances": False, "category": "nature", "catid": 4, "name": "terrain", "ignoreInEval": False, "id": 22, "color": [152, 251, 152], "trainId": 9},
    {"hasInstances": False, "category": "sky", "catid": 5, "name": "sky", "ignoreInEval": False, "id": 23, "color": [70, 130, 180], "trainId": 10},
    {"hasInstances": True, "category": "human", "catid": 6, "name": "person", "ignoreInEval": False, "id": 24, "color": [220, 20, 60], "trainId": 11},
    {"hasInstances": True, "category": "human", "catid": 6, "name": "rider", "ignoreInEval": False, "id": 25, "color": [255, 0, 0], "trainId": 12},
    {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "car", "ignoreInEval": False, "id": 26, "color": [0, 0, 142], "trainId": 13},
    {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "truck", "ignoreInEval": False, "id": 27, "color": [0, 0, 70], "trainId": 14},
    {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "bus", "ignoreInEval": False, "id": 28, "color": [0, 60, 100], "trainId": 15},
    {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "caravan", "ignoreInEval": True, "id": 29, "color": [0, 0, 90], "trainId": 255},
    {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "trailer", "ignoreInEval": True, "id": 30, "color": [0, 0, 110], "trainId": 255},
    {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "train", "ignoreInEval": False, "id": 31, "color": [0, 80, 100], "trainId": 16},
    {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "motorcycle", "ignoreInEval": False, "id": 32, "color": [0, 0, 230], "trainId": 17},
    {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "bicycle", "ignoreInEval": False, "id": 33, "color": [119, 11, 32], "trainId": 18},
    {"hasInstances": False, "category": "vehicle", "catid": 7, "name": "license plate", "ignoreInEval": True, "id": -1, "color": [0, 0, 142], "trainId": -1}
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
		self.n_cats = 19
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