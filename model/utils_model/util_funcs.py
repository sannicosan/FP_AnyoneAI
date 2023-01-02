#-----------------------------------------------------------------------------------------------------------------------------
# # PREAMBLE
#-----------------------------------------------------------------------------------------------------------------------------

import os
import pandas as pd
import numpy as  np
from PIL import Image

# Self-made
from utils_model import bboxes
from utils_model import cons

#-----------------------------------------------------------------------------------------------------------------------------
# Functions
#-----------------------------------------------------------------------------------------------------------------------------


def read_csv_chunks(img_set: str='train', chunksize: int=10000) -> pd.io.parsers.TextFileReader:
    """ 
    Creates a dataframe chunk to iterate over, in order to read the bounding 
    boxes coordinates in the annotation csv files.

    Parameters
    ----------
    img_set: str
        The image set (train,test or val).
    chunksize: int
        Size of the Dataframe chunk.
    Returns
    ----------
    chunk: TextFileReader 
        Iterator to read Dataframe in chunks.
    """
    # Build path to annotation file
    ttv = img_set.split('_')[0]
    annot_file = 'annotations_' + ttv + '.csv'
    annotation_path = os.path.join(cons.ANNOT_PATH, annot_file)
    
    return pd.read_csv(annotation_path, names= cons.ANNOT_COLS, chunksize=chunksize)

def drop_missing_img(imgs: pd.Series) -> pd.Series:
    """ 
    Get a list of failed images (based on criterion)

    Parameters
    ----------
    imgs: pd.DataFrame
        The dataframe containing the images and it's tags. 

    Returns
    ----------
    clean_imgs: pd.Series
        Series of existent images.
    """

    # Check if the img_name exist. Drop it if it doesn't 
    for img in imgs.index:
            # Build path to image
        ttv = img.split('_')[0]
        img_path = os.path.join(cons.IMG_FOLDER, ttv,'images',img) 

        if not os.path.exists(img_path): 
            imgs.drop(img, inplace= True)
    
    
    return imgs

def get_failed_imgs(tags_df: pd.DataFrame, criterion: str = 'area', thresh: float = 0.01, verbose: bool = True) -> list:
    """ 
    Get a list of failed images (based on criterion)

    Parameters
    ----------
    tag_df: pd.DataFrame
        The dataframe containing the images and it's tags. 
    criterion: str
        The criterion to use to decide for a failed images.
        It can be either 'area' or 'number of bboxes'
    thresh: float
        The threshold to use fo filter for failed images. 
        If criterion == 'area': this corresponds to the lower quantile.
        If criterion == 'n_bboxes': this corresponds to the min number of bounding boxes.
    verbose: bool
        Wether extra information will be printed or not.
    Returns
    ----------
    failed_imgs: list
        List of failed image names.
    """
    # Check for valid criterion
    if criterion not in cons.CRITERIA:
        raise ValueError("Criterion used not valid. Enter either 'area' or 'n_bboxes'.")
    
    
    if criterion == 'n_bboxes':
        
        # Count images with less that `thresh` tags
        n_tags_per_image = tags_df.groupby('img_name').size().sort_values()
        failed_imgs = n_tags_per_image[n_tags_per_image < thresh]
      
        # Check if the img_name exist. Drop it if it doesn't 
        failed_imgs = drop_missing_img(failed_imgs)
      
        # Prints information
        if verbose: 
            n_failed = failed_imgs.shape[0]
            print('Number of failed images: ', n_failed)
            print(f'List of failed images:\n{failed_imgs}')
            print(f'\nThis represents {n_failed/len(tags_df)*100}% of the images')

    elif criterion == 'area': 
        
        # Getting area realted dataframe and bbox_arad cover
        areas_df = bboxes.get_bboxes_total_area(tags_df)
        bbox_area_cover = areas_df.groupby('img_name').bbox_area_perc.sum().sort_values()
        
        FAIL_THRESH = np.quantile(bbox_area_cover,q = thresh)
        failed_imgs = bbox_area_cover[bbox_area_cover < FAIL_THRESH]
        
        # Check if the img_name exist. Drop it if it doesn't 
        failed_imgs = drop_missing_img(failed_imgs)
        
        # Prints information
        if verbose: 
            print('Treshold used:', FAIL_THRESH)
            print('# failed images: ', len(failed_imgs))

    return failed_imgs            
            
def detected_corrupted_imgs(tags_df: pd.DataFrame) -> list:
    """ 
    Detects corrupted images in the dataset
    ----------
    tag_df: pd.DataFrame
        The dataframe containing the images and it's tags. 
    Returns
    ----------
    corrupted_imgs: list
        List of corrupted images sorted by name. 
    """
        
    img_list = set(tags_df.index)

    corrupted_imgs = []
    for img_name in img_list:
        
        # Build path to img
        folder = img_name.split('_')[0]
        img_path = os.path.join(cons.IMG_FOLDER,folder,'images',img_name)
        # Read img
        try: 
            img = Image.open(img_path)
            img.getdata()[0]
        except:
            # Append the corrupted img_name and continue
            corrupted_imgs.append(img_name)
            continue
       
    return sorted(corrupted_imgs)
    
def to_yolov5_coords(original_tags_df:pd.DataFrame) -> pd.DataFrame:
    """ 
    Converts the bboxes coordinates from format `xmin, ymin, xmax, ymax`
    to `center_x, center_y, width_x, height_y`  
    ----------
    original_label_df: pd.DataFrame
        Dataframe containing the image names and it's tags using
        `xmin, ymin, xmax, ymax`coordinates 
    Returns
    ----------
    yolo_labels_df: pd.DataFrame
        Dataframe with the bboxes coordinates converted to yolo
        format: `class_id,center_x, center_y, width_x, height_y`  
    """ 
    
    # Get original coordinates
    Width =     original_tags_df.total_width
    Height =    original_tags_df.total_height
    
    xmin_coords =   original_tags_df.x1 
    ymin_coords =   original_tags_df.y1
    xmax_coords =   original_tags_df.x2
    ymax_coords =   original_tags_df.y2
    
    # Compute YOLO coordinates
    center_x =  ((xmax_coords.values + xmin_coords.values) //2) / Width 
    center_y =  ((ymax_coords.values + ymin_coords.values) //2) / Height  
    width_x  =  (xmax_coords.values - xmin_coords.values) / Width
    height_y =  (ymax_coords.values - ymin_coords.values) / Height
    
    # Create the new Dataframe with the corresponding columns
    yolo_labels_df = pd.DataFrame()
    
    yolo_labels_df.index = original_tags_df.index
    yolo_labels_df['class_id']  =   1
    yolo_labels_df['center_x']  =   center_x
    yolo_labels_df['center_y']  =   center_y
    yolo_labels_df['width_bb']  =   width_x
    yolo_labels_df['height_bb'] =   height_y
    
    return yolo_labels_df


def labels_to_txt(yolo_labels_df: pd.DataFrame, label_path:str = cons.LABEL_PATH ) -> None:
    
    """ 
    Convert a dataframe to a series of text files (yolo coordinates)
    ----------
    yolo_labels_df: pd.DataFrame
        Dataframe containing the image names and it's tags using
        yolo coordinates 
    Returns
    ----------
    None 
    """ 
    
    # Preamble
    img_set = sorted(set(yolo_labels_df.index))
    formatter = ['%d'] + ['%1.16f']*4              
    
    # Create folder
    ttv = img_set[0].split('_')[0]
    folder = os.path.join(label_path, ttv ,'labels')
    os.makedirs(folder, exist_ok= True)
    
    for img in img_set:
        
        # Filepath
        filename = img.split('.')[:-1]
        filename = '.'.join(filename) + '.txt'
        filepath = os.path.join(folder, filename)
        
        # Get coordinate values
        values = np.array(yolo_labels_df.loc[img].values)
        if values.ndim == 1: values = values.reshape(1,-1)
                 
        # Save to text file
        np.savetxt(filepath,values, fmt= formatter)
        if os.path.exists(filepath): print(f' {filename} saved')
    
def pick_random_imgs(start, end, size, img_set:str = 'train') -> None:

    img_list = []
    # List of images
    while (len(img_list) < size):
        
        img_number = np.random.randint(low = start, high = end + 1)
        img_name = img_set+ '_' + str(img_number) + '.jpg'
        path_img = os.path.join(cons.IMG_FOLDER ,img_set, 'images',img_name)
        if os.path.exists(path_img) and img_number not in img_list:
            img_list.append(img_number)
        
    img_names = [ (img_set+'_' + str(img_i) + '.jpg', img_set+'_' + str(img_i) + '.txt' ) for img_i in img_list]
    
    # Create new folder
    folder = os.path.join(cons.DATA_PATH, img_set + '_' + str(size))
   
    os.makedirs(os.path.join(folder, 'images'),exist_ok= True)            # ./data/train_100/images
    os.makedirs(os.path.join(folder, 'labels'),exist_ok= True)            # ./data/train_100/labels
    
    # Link images
    for img_name,lbl_name in img_names:
        
        src_path_img = os.path.join(cons.IMG_FOLDER ,img_set, 'images',img_name)
        src_path_lbl = os.path.join(cons.IMG_FOLDER ,img_set, 'labels',lbl_name)
        img_path = os.path.join(folder,'images', img_name)
        lbl_path = os.path.join(folder,'labels', lbl_name)
        
        if not os.path.exists(img_path):          
            os.link(src_path_img,img_path)
            os.link(src_path_lbl,lbl_path)
            print(f'{img_path} created.')