
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------# CONSTANTS
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Static directories
IMG_FOLDER = './data/train_test_SKU'
ANNOT_PATH = './data/SKU110K/annotations'
LABEL_PATH = './data/train_test_SKU'
BAD_PATH='./data/bad_data'
SRC_PATH = "./data/SKU110K/images"
DST_PATH = "./data/train_test_SKU"

# Data handling
ANNOT_COLS = ['img_name', 'x1', 'y1', 'x2', 'y2', 'type', 'total_width', 'total_height']
CRITERIA = ['area','n_bboxes']

# Colors
BLUE =  (255, 0, 0)  
GREEN = (0, 255, 0)  
RED =   (0, 0, 255)
