#------------------------------------------------------------------------------
#	Libraries
#------------------------------------------------------------------------------
import cv2, torch, argparse
import time
import numpy as np
from torch.nn import functional as F

from models import UNet
from models import ICNet
from models import BiSeNet
from models import DeepLabV3Plus
from models import UNetPlus

from dataloaders import transforms
from utils import utils

import glob
from tqdm import tqdm


#------------------------------------------------------------------------------
#   Argument parsing
#------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Arguments for the script")

parser.add_argument('--use_cuda', action='store_true', default=False,
                    help='Use GPU acceleration')

parser.add_argument('--model', type = str, default="ICNet",
                    help='Choose models between ICnet, UNet, BiSeNet and DeepLabV3Plus and UNetPlus')

parser.add_argument('--input_sz', type=int, default=320,
                    help='Input size')

parser.add_argument('--checkpoint', type=str, default="/media/antiaegis/storing/FORGERY/segmentation/checkpoints/HumanSeg/UNet_MobileNetV2/model_best.pth",
                    help='Path to the trained model file')

parser.add_argument('--inputDir', type=str, default="",
                    help='Path to the input image directory')

parser.add_argument('--outputMaskDir', type=str, default="",
                    help='Path to the output directory')\

parser.add_argument('--outputOverlayDir', type=str, default="",
                    help='Path to the output directory')

args = parser.parse_args()


#------------------------------------------------------------------------------
#	Create model and load weights
#------------------------------------------------------------------------------

model = None
if str(args.model) == 'UNet':
    model = UNet(
        backbone="mobilenetv2",
        num_classes=2,
    pretrained_backbone=None
    )
elif str(args.model) == 'BiSeNet':
    model = BiSeNet(
        backbone="resnet18",
        num_classes=2,
    pretrained_backbone=None
    )
elif str(args.model) == 'DeepLabV3Plus':
    model = DeepLabV3Plus(
        backbone="resnet18",
        num_classes=2,
    pretrained_backbone=None
    )
elif str(args.model) == 'UNetPlus':
    model = UNetPlus(
        backbone="resnet18",
        num_classes=2,
    pretrained_backbone=None
    )
else:
    model = ICNet(
    backbone="resnet18",
    num_classes=2,
	pretrained_backbone=None
)


if args.use_cuda:
	model = model.cuda()
trained_dict = torch.load(args.checkpoint, map_location="cpu")['state_dict']
model.load_state_dict(trained_dict, strict=False)
model.eval()

def path_leaf(path):
  import ntpath
  head, tail = ntpath.split(path)
  return tail or ntpath.basename(head)

def predict(imagePath):
    #predict image
    frame = cv2.imread(imagePath)
    image = frame[...,::-1]
    h, w = image.shape[:2]

    inferTime = None

    X, pad_up, pad_left, h_new, w_new = utils.preprocessing(image, expected_size=args.input_sz, pad_value=0)

    with torch.no_grad():
        if args.use_cuda:
            start = time.time()
            mask = model(X.cuda())
            end = time.time()
            inferTime = end-start
            mask = mask[..., pad_up: pad_up+h_new, pad_left: pad_left+w_new]
            mask = F.interpolate(mask, size=(h,w), mode='bilinear', align_corners=True)
            mask = F.softmax(mask, dim=1)
            mask = mask[0,1,...].cpu().numpy()
        else:
            start = time.time()
            mask = model(X)
            end = time.time()
            inferTime = end-start
            mask = mask[..., pad_up: pad_up+h_new, pad_left: pad_left+w_new]
            mask = F.interpolate(mask, size=(h,w), mode='bilinear', align_corners=True)
            mask = F.softmax(mask, dim=1)
            mask = mask[0,1,...].numpy()

        #mask outputSection
        booleanMask = mask >= 0.5

        return booleanMask, frame, inferTime


imagedirs = str(args.inputDir) + str("*")
inferTimes = []
for imagepath in tqdm(glob.iglob(imagedirs)):
    imageName = path_leaf(imagepath)

    booleanMask, frame, inferTime = predict(imagepath)
    inferTimes.append(inferTime)

    if str(args.outputMaskDir) != "":
        booleanMaskRGB = cv2.cvtColor(np.float32(booleanMask)*255,cv2.COLOR_GRAY2RGB)
        savingpath = str(args.outputMaskDir) + str(imageName)
        cv2.imwrite(savingpath, booleanMaskRGB)

    if str(args.outputOverlayDir) != "":
        overLayImage = cv2.addWeighted(np.asarray(frame, np.float64),0.7,np.asarray(booleanMaskRGB, np.float64),0.3,0.0)
        savingpath = str(args.outputOverlayDir) + str(imageName)
        cv2.imwrite(savingpath, overLayImage)

inferTimes = np.array(inferTimes)
print("Total images given = " + str(inferTimes.shape[0]))
print("Mean Inference Time = " + str(np.mean(inferTimes)) + " Seconds")
