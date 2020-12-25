#------------------------------------------------------------------------------
#   Library
#------------------------------------------------------------------------------
import cv2
import numpy as np


#------------------------------------------------------------------------------
#   Random crop
#------------------------------------------------------------------------------
def random_crop(prev_image, image, label, crop_range):
	"""
	cropped image is a square.

	image (ndarray) with shape [H,W,3]
	label (ndarray) with shape [H,W]
	crop_ratio (list) contains 2 bounds
	"""
	##### Exception #####
	if crop_range[0]==crop_range[1] and crop_range[0]==1.0:
		return prev_image, image, label

	# Get random crop_ratio
	crop_ratio = np.random.choice(np.linspace(crop_range[0], crop_range[1], num=10), size=())
	
	# Get random coordinates
	H, W = label.shape
	size = H if H<W else W
	size = int(size*crop_ratio)
	max_i, max_j = H-size, W-size
	i = np.random.choice(np.arange(0, max_i+1), size=())
	j = np.random.choice(np.arange(0, max_j+1), size=())

	# Crop
	prev_image_cropped = prev_image[i:i+size, j:j+size, :]
	image_cropped = image[i:i+size, j:j+size, :]
	label_cropped = label[i:i+size, j:j+size]
	return prev_image_cropped, image_cropped, label_cropped


#------------------------------------------------------------------------------
#   Horizontal flip
#------------------------------------------------------------------------------
def flip_horizon(prev_image, image, label, prob):
	if prob:
		if np.random.choice([False, True], size=(), p=[1-prob, prob]):
			prev_image = np.flip(prev_image, axis=1)
			image = np.flip(image, axis=1)
			label = np.flip(label, axis=1)
	return prev_image, image, label


#------------------------------------------------------------------------------
#   Rotate 90
#------------------------------------------------------------------------------
def rotate_90(prev_image, image, label, prob):
	if prob:
		k = np.random.choice([-1, 0, 1], size=(), p=[prob/2, 1-prob, prob/2])
		if k:
			prev_image = np.rot90(prev_image, k=k, axes=(0,1))
			image = np.rot90(image, k=k, axes=(0,1))
			label = np.rot90(label, k=k, axes=(0,1))
	return prev_image, image, label


#------------------------------------------------------------------------------
#   Rotate angle
#------------------------------------------------------------------------------
def rotate_angle(prev_image, image, label, angle_max):
	if angle_max:
		# Random angle in range [-angle_max, angle_max]
		angle = np.random.choice(np.linspace(-angle_max, angle_max, num=21), size=())

		# Get parameters for affine transform
		(h, w) = image.shape[:2]
		(cX, cY) = (w // 2, h // 2)

		M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
		cos = np.abs(M[0, 0])
		sin = np.abs(M[0, 1])

		nW = int((h * sin) + (w * cos))
		nH = int((h * cos) + (w * sin))

		M[0, 2] += (nW / 2) - cX
		M[1, 2] += (nH / 2) - cY

		# Perform transform
		prev_image = cv2.warpAffine(prev_image, M, (nW, nH))
		image = cv2.warpAffine(image, M, (nW, nH))
		label = cv2.warpAffine(label, M, (nW, nH))
	return prev_image, image, label


#------------------------------------------------------------------------------
#  Gaussian noise
#------------------------------------------------------------------------------
def random_noise(image, std):
	if std:
		noise = np.random.normal(0, std, size=image.shape)
		image = image + noise
		image[image<0] = 0
		image[image>255] = 255
		image = image.astype(np.uint8)
	return image


#------------------------------------------------------------------------------
#  Resize image
#------------------------------------------------------------------------------
def resize_image(image, expected_size, pad_value, ret_params=False, mode=cv2.INTER_LINEAR):
	"""
	image (ndarray) with either shape of [H,W,3] for RGB or [H,W] for grayscale.
	Padding is added so that the content of image is in the center.
	"""
	h, w = image.shape[:2]
	if w>h:
		w_new = int(expected_size)
		h_new = int(h * w_new / w)
		image = cv2.resize(image, (w_new, h_new), interpolation=mode)

		pad_up = (w_new - h_new) // 2
		pad_down = w_new - h_new - pad_up
		if len(image.shape)==3:
			pad_width = ((pad_up, pad_down), (0,0), (0,0))
			constant_values=((pad_value, pad_value), (0,0), (0,0))
		elif len(image.shape)==2:
			pad_width = ((pad_up, pad_down), (0,0))
			constant_values=((pad_value, pad_value), (0,0))

		image = np.pad(
			image,
			pad_width=pad_width,
			mode="constant",
			constant_values=constant_values,
		)
		if ret_params:
			return image, pad_up, 0, h_new, w_new
		else:
			return image

	elif w<h:
		h_new = int(expected_size)
		w_new = int(w * h_new / h)
		image = cv2.resize(image, (w_new, h_new), interpolation=mode)

		pad_left = (h_new - w_new) // 2
		pad_right = h_new - w_new - pad_left
		if len(image.shape)==3:
			pad_width = ((0,0), (pad_left, pad_right), (0,0))
			constant_values=((0,0), (pad_value, pad_value), (0,0))
		elif len(image.shape)==2:
			pad_width = ((0,0), (pad_left, pad_right))
			constant_values=((0,0), (pad_value, pad_value))

		image = np.pad(
			image,
			pad_width=pad_width,
			mode="constant",
			constant_values=constant_values,
		)
		if ret_params:
			return image, 0, pad_left, h_new, w_new
		else:
			return image

	else:
		image = cv2.resize(image, (expected_size, expected_size), interpolation=mode)
		if ret_params:
			return image, 0, 0, expected_size, expected_size
		else:
			return image
