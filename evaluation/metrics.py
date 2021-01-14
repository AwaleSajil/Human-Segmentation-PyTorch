#------------------------------------------------------------------------------
#   Libraries
#------------------------------------------------------------------------------
import torch
from torch.nn import functional as F


#------------------------------------------------------------------------------
#   Fundamental metrics
#------------------------------------------------------------------------------
def miou(logits, targets, eps=1e-6):
	# this function miou not great when you have to ignonre calss see mIOU_corrected() for that
	"""
	logits: (torch.float32)  shape (N, C, H, W)
	targets: (torch.float32) shape (N, H, W), value {0,1,...,C-1}
	"""
	outputs = torch.argmax(logits, dim=1, keepdim=True).type(torch.int64)
	targets = torch.unsqueeze(targets, dim=1).type(torch.int64)
# 	outputs = torch.zeros_like(logits).scatter_(dim=1, index=outputs, src=torch.tensor(1.0)).type(torch.int8)
# 	targets = torch.zeros_like(logits).scatter_(dim=1, index=targets, src=torch.tensor(1.0)).type(torch.int8)
	

	outputs = torch.zeros_like(logits).scatter_(dim=1, index=outputs, src=torch.ones_like(logits)).type(torch.int8)
	targets = torch.zeros_like(logits).scatter_(dim=1, index=targets, src=torch.ones_like(logits)).type(torch.int8)
	
	# import sys
	# print(outputs.size())
	# print(targets.size())
	# sys.exit()

	inter = (outputs & targets).type(torch.float32).sum(dim=(2,3))
	union = (outputs | targets).type(torch.float32).sum(dim=(2,3))
	iou = inter / (union + eps)
	return iou.mean()

def mIOU_corrected(pred, label, num_classes=19):
	pred = torch.argmax(logits, dim=1, keepdim=True).type(torch.int64).squeeze(1)
	label = label.type(torch.int64)
	iou_list = list()
	present_iou_list = list()
	pred = pred.view(-1)
	label = label.view(-1)

	# Note: Following for loop goes from 0 to (num_classes-1)
	# and ignore_index is num_classes, thus ignore_index is
	# not considered in computation of IoU.
	for sem_class in range(num_classes):
		pred_inds = (pred == sem_class)
		target_inds = (label == sem_class)
		if target_inds.long().sum().item() == 0:
			iou_now = float('nan')
		else: 
			intersection_now = (pred_inds[target_inds]).long().sum().item()
			union_now = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection_now
			iou_now = float(intersection_now) / float(union_now)
			present_iou_list.append(iou_now)
		iou_list.append(iou_now)
	return np.mean(present_iou_list)


def iou_with_sigmoid(sigmoid, targets, eps=1e-6):
	"""
	sigmoid: (torch.float32) shape (N, 1, H, W)
	targets: (torch.float32) shape (N, H, W), value {0,1}
	"""
	outputs = torch.squeeze(sigmoid, dim=1).type(torch.int8)
	targets = targets.type(torch.int8)

	inter = (outputs & targets).type(torch.float32).sum(dim=(1,2))
	union = (outputs | targets).type(torch.float32).sum(dim=(1,2))
	iou = inter / (union + eps)
	return iou.mean()


#------------------------------------------------------------------------------
#   Custom IoU for BiSeNet
#------------------------------------------------------------------------------
def custom_bisenet_miou(logits, targets):
	"""
	logits: (torch.float32) (main_out, feat_os16_sup, feat_os32_sup) of shape (N, C, H, W)
	targets: (torch.float32) shape (N, H, W), value {0,1,...,C-1}
	"""
	if type(logits)==tuple:
		return miou(logits[0], targets)
	else:
		return miou(logits, targets)


#------------------------------------------------------------------------------
#   Custom IoU for PSPNet
#------------------------------------------------------------------------------
def custom_pspnet_miou(logits, targets):
	"""
	logits: (torch.float32) (main_out, aux_out) of shape (N, C, H, W), (N, C, H/8, W/8)
	targets: (torch.float32) shape (N, H, W), value {0,1,...,C-1}
	"""
	if type(logits)==tuple:
		return miou(logits[0], targets)
	else:
		return miou(logits, targets)


#------------------------------------------------------------------------------
#   Custom IoU for BiSeNet
#------------------------------------------------------------------------------
def custom_icnet_miou(logits, targets):
	"""
	logits: (torch.float32)
		[train_mode] (x_124_cls, x_12_cls, x_24_cls) of shape
						(N, C, H/4, W/4), (N, C, H/8, W/8), (N, C, H/16, W/16)

		[valid_mode] x_124_cls of shape (N, C, H, W)

	targets: (torch.float32) shape (N, H, W), value {0,1,...,C-1}
	"""
	if type(logits)==tuple:
		targets = torch.unsqueeze(targets, dim=1)
		targets = F.interpolate(targets, size=logits[0].shape[-2:], mode='bilinear', align_corners=True)[:,0,...]
		return miou(logits[0], targets)
	else:
		return miou(logits, targets)


#------------------------------------------------------------------------------
#   Custom IoU for ICNet-semantic-branchv0.0.1

#------------------------------------------------------------------------------
def custom_icnet_miou_semantic_branch_v0_0_1(logits, targets):
	"""
	logits: (torch.float32)
		[train_mode] (x_124_cls, x_12_cls, x_24_cls) of shape
						(N, C, H/4, W/4), (N, C, H/8, W/8), (N, C, H/16, W/16)

		[valid_mode] x_124_cls of shape (N, C, H, W)

	targets: (torch.float32) shape (N, H, W), value {0,1,...,C-1}
	"""
	if type(logits)==tuple:
		targets = torch.unsqueeze(targets, dim=1)
		targets = F.interpolate(targets, size=logits[0].shape[-2:], mode='nearest')[:,0,...]
		return mIOU_corrected(logits[0], targets)
	else:
		return mIOU_corrected(logits, targets)
