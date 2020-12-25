#Demo for this create pair can be found at :https://colab.research.google.com/drive/1WP-iY6hVHEVtzKhIxsCYdrEnwUplZYm4?usp=sharing
#------------------------------------------------------------------------------
#	Libraries
#------------------------------------------------------------------------------
import os, cv2, sys
import numpy as np
from glob import glob
from tqdm import tqdm
from random import shuffle, seed
from multiprocessing import Pool, Manager
import argparse

#------------------------------------------------------------------------------
#	Parse from command line arguments
#------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Create data,target pir files')

parser.add_argument('-d', '--data', default=[], nargs='+', help='directory path to images')

parser.add_argument('-l', '--label', default=[], nargs='+', help='directory path to label images')

parser.add_argument('-br', '--balance_ratio', default=[0.0, 1.0], nargs=2, help='Filter dataset with forground and background ratio')

parser.add_argument('-s', '--split', default=0.9, type = float, help='train/test split ratio')

parser.add_argument('--train', action='store_true', default=False, help='to crrete only train pair files')

parser.add_argument('--val', action='store_true', default=False, help='to crrete only val pair files')

CLargs = parser.parse_args()

def path_leaf(path):
  import ntpath
  head, tail = ntpath.split(path)
  return tail or ntpath.basename(head)

def numericalConvert(fullPath):
  filename = path_leaf(fullPath)
  #remove extension
  name = filename.split('.')[:-1]
  name = ''.join(name)
  name = name.split('_')
  name = ''.join(name)
  return int(name)

def completePath(path):
  pathCom = None
  if (path[-1]) != "/":
    pathCom = path + "/*.*"
  else:
    pathCom = path + "*.*"
  return pathCom

def folderPaths(path):
  p = None
  if (path[-1]) != "/":
    p = path + "/*"
  else:
    p = path + "*"
  return p


#------------------------------------------------------------------------------
#	Main execution
#------------------------------------------------------------------------------
# Get files
image_files = []
datas = sorted(CLargs.data)

label_files = []
data_labels = sorted(CLargs.label)

##Loop through label images
LpathCom = completePath(data_labels[0])
label_paths = sorted(glob(LpathCom), key = numericalConvert)

Dpath = folderPaths(datas[0])
# Dpath = sorted(glob(Dpath))

for label_path in label_paths:
  label_name = ''.join((path_leaf(label_path)).split('.')[:-1])
  file_number = int(label_name.split('_')[0])
  label_number= int(''.join(label_name.split('_')[1:]))

  #compute datapath corresponding to current label
  ExpectedDataPath = Dpath[:-1] + str(format(file_number, '04d')) + '/' + path_leaf(label_path)
  #check if this exists
  if os.path.exists(ExpectedDataPath):
    label_files.append(label_path)
    image_files.append(ExpectedDataPath)

#generete Previous Image Path list
prev_image_files = []
for image_file in image_files:
  imgdir = '/'.join((str(image_file)).split('/')[:-1] + [''])
  image_name_w_e = path_leaf(image_file)
  image_name = image_name_w_e.split('.')[0]
  image_number = image_name.split('_')[-1]
  image_number_int = int(image_number)
  prev_image_number_int = image_number_int - 1
  prev_image_number = None
  if (image_number == str(image_number_int)):
    #in non preceeding 0 format
    prev_image_number = str(prev_image_number_int)
  else:
    #with formated preceeding zeros
    formatstr = '0' + str(len(image_number)) + 'd'
    prev_image_number = str(format(prev_image_number_int, formatstr))

  #finally generete expected prev_image_filename
  image_name_pre_components = '_'.join(image_name.split('_')[:-1] + [''])
  prev_image_name_w_e = image_name_pre_components + prev_image_number + '.' + image_name_w_e.split('.')[-1]\

  prev_image_file = imgdir + prev_image_name_w_e

  ##NOW chwck if the prev_image_file exists
  if os.path.exists(prev_image_file):
    prev_image_files.append(prev_image_file)
  else:
    prev_image_files.append(None)



# assert len(image_files)==len(label_files)
n_files = len(image_files)

# Shuffle
seed(0)
shuffle(prev_image_files)
seed(0)
shuffle(image_files)
seed(0)
shuffle(label_files)


#------------------------------------------------------------------------------
#   Some Useful function defination
#------------------------------------------------------------------------------

def writePairFile(FILE_NAME, index):
  global prev_image_files, image_files, label_files
  fp = open(FILE_NAME, "w")
  for idx in index:
      prev_image_file, image_file, label_file = prev_image_files[idx], image_files[idx], label_files[idx]
      line = "%s, %s, %s" % (prev_image_file, image_file, label_file)
      fp.writelines(line + "\n")

def checkDataset(FILE_NAME, description = 'dataset'):
  fp = open(FILE_NAME, 'r')
  lines = fp.read().split("\n")
  lines = [line.strip() for line in lines if len(line)]
  lines = [line.split(", ") for line in lines]

  print("Checking %d %s samples..." % (len(lines), description))
  for line in lines:
      prev_image_files, image_file, label_file = line
      if not os.path.exists(prev_image_files):
          print("%s does not exist!" % (prev_image_files))
      if not os.path.exists(image_file):
          print("%s does not exist!" % (image_file))
      if not os.path.exists(label_file):
          print("%s does not exist!" % (label_file))

# Count number of pixels belong to categories
manager = Manager()
foregrounds = manager.list([])
backgrounds = manager.list([])

def pool_func(args):
    label_file = args
    img = cv2.imread(label_file, 0)
    foreground = np.sum((img>0).astype(np.uint8)) / img.size
    background = np.sum((img==0).astype(np.uint8)) / img.size
    foregrounds.append(foreground)
    backgrounds.append(background)

pools = Pool(processes=8)
args = label_files
for _ in tqdm(pools.imap_unordered(pool_func, args), total=len(label_files)):
    pass

foregrounds = [element for element in foregrounds]
backgrounds = [element for element in backgrounds]

print("foregrounds:", sum(foregrounds)/n_files)
print("backgrounds:", sum(backgrounds)/n_files)
print("ratio:", sum(foregrounds) / sum(backgrounds))

# Divide into 3 groups: small, averg, and large
RATIO = [float(CLargs.balance_ratio[0]), float(CLargs.balance_ratio[1])]
# averg_ind = []
selected_ind = []
for idx, foreground in enumerate(foregrounds):
    if RATIO[0] <= foreground <= RATIO[1]:
        #also make sure it has prev_image_file
        if prev_image_files[idx] != None:
          selected_ind.append(idx)

print("Number of selected indices:", len(selected_ind))

# Split train/valid
RATIO = CLargs.split
TRAIN_FILE = "dataset/train_mask.txt"
VALID_FILE = "dataset/valid_mask.txt"



if not (CLargs.train or CLargs.val):
  shuffle(selected_ind)
  ind_train = selected_ind[:int(RATIO*len(selected_ind))]
  ind_valid = selected_ind[int(RATIO*len(selected_ind)):]
  print("Number of training samples:", len(ind_train))
  print("Number of validating samples:", len(ind_valid))

  writePairFile(TRAIN_FILE, ind_train)
  writePairFile(VALID_FILE, ind_valid)
  #checking dataset
  checkDataset(TRAIN_FILE, description = 'training')
  checkDataset(VALID_FILE, description = 'validating')

elif (CLargs.train):
  #only generate train pairs
  shuffle(selected_ind)
  print("Number of all samples (train):", len(selected_ind))

  writePairFile(TRAIN_FILE, selected_ind)
  checkDataset(TRAIN_FILE, description = 'training')

else:
  #if CLargs.val
  #only generate val pairs
  print("Number of all samples (val):", len(selected_ind))

  writePairFile(VALID_FILE, selected_ind)
  checkDataset(VALID_FILE, description = 'validating')
