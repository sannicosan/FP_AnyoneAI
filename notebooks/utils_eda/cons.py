
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------# CONSTANTS
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Static directories
DATA_PATH = '../data'
IMG_FOLDER  = DATA_PATH + '/train_test_SKU'
ANNOT_PATH  = DATA_PATH + '/SKU110K/annotations'
LABEL_PATH  = DATA_PATH + '/train_test_SKU'
BAD_PATH    = DATA_PATH + '/bad_data'
SRC_PATH    = DATA_PATH + '/SKU110K/images'
DST_PATH    = DATA_PATH + '/train_test_SKU'

# Data handling
ANNOT_COLS = ['img_name', 'x1', 'y1', 'x2', 'y2', 'type', 'total_width', 'total_height']
CRITERIA = ['area','n_bboxes']

# Colors
BLUE =  (255, 0, 0)  
GREEN = (0, 255, 0)  
RED =   (0, 0, 255)
