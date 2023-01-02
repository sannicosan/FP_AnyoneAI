#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PREAMBLE
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Python imports
import os 
import cv2
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

# Self-made
from utils_eda import util_funcs 
from utils_eda import cons

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Functions
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def get_bbox_coords(img_name: str='train_0.jpg') -> pd.DataFrame:
    """ 
    Gets the box coordinates of given image in a dataframe format.

    Parameters
    ----------
    img_name: str
        The image name.
    Returns
    ----------
    box_coords: pd.DataFrame 
        Dataframe with the box coordinates for given image.
    """
    try: 
        
        img_set = img_name.split('_')[0]
        # Search the image in the csv in chunks
        for chunk_df in util_funcs.read_csv_chunks(img_set):
            
            img_df = chunk_df[ chunk_df.img_name == img_name ]
            if not img_df.empty: break
        
        # Get the coordinates        
        box_coords = img_df[[ 'x1', 'y1', 'x2', 'y2']] 
        
        return box_coords
    
    except NameError:
        print('Image name does not exist in the dataset' )
        


def get_bboxes(img_path: str = os.path.join(cons.IMG_FOLDER,'train/images/train_0.jpg')) -> pd.DataFrame:
    """ 
    It gets the coordinates of the bounding boxes corresponding to
    the image passsed in `img_path`.

    Parameters
    ----------
    img_path: str
        Path to image.
    ----------
    box_coordinates: pd.DataFrame
        DataFrame with coordinates of the bounding boxes.
    """
 
    # Read the image
    img_name = os.path.split(img_path)[-1]
    box_coordinates = get_bbox_coords(img_name)
    
    return box_coordinates

def plot_bboxes(img_path: str = os.path.join(cons.IMG_FOLDER,'train/images/train_0.jpg'), box_coordinates = pd.DataFrame(),axes: Axes = None, skip_plot: bool = False, style: str = 'bbox'):
    """ 
    It plots the bounding boxes in green when products are present.
    If there are missing products, then red bboxes are drawn.

    Parameters
    ----------
    img_path: str
        Path to image.
        
    box_coordinates: pd.DataFrame
        Contains the image coordinates to plot. 
        Default = None: searches for the coordinates in the static dataset (.csv)
        stored under `data/SKU110K/annotations/annotations.csv`.
        
    axes: matplotlib.axes.Axes (Optional)
        Axes in which to plot the image.
        
    skip_plot: bool
        Wether to skip or not the plot of the image.
        
    style: str
        The style of the bounding boxes. Use:
        - bbox: for standard bboxes
        - heatmap: heatmap version for (missing products only)
        
    Returns
    ----------
    img: np.array (Optional)
        Image plotted.
    """
    #Read the image
    img = cv2.imread(img_path)
    img2 = img.copy()
    
    if style == 'bbox':
        
        # Get BBox coordinates
        if box_coordinates.empty:
            box_coordinates = get_bboxes(img_path=img_path)
        
        # Plot all boxes
        for _, box_coords in box_coordinates.iterrows():
            
            x1, y1, x2, y2 = (box_coords.x1, box_coords.y1, box_coords.x2, box_coords.y2)
            img = cv2.rectangle(img, (x1, y1), (x2, y2), cons.GREEN, thickness=5)

                  
    
    elif style == 'heatmap':
        
        # Change all pixels to black and draw white rectangles in bounding boxxes of class 0
        # This is because we want all pixels white or black
        img[:,:] = (0,0,0)    
        
        for _,row in box_coordinates.loc[box_coordinates["class"]==0,:].iterrows():
            
            img  = cv2.rectangle(img, (row["xmin"], row["ymin"]), (row["xmax"], row["ymax"]), (255,255,255),-1)
       
        #For using the distance transformation we need a black/white mask
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     
        #Distance transformation the core of the code
        img = cv2.distanceTransform(img, cv2.DIST_L1, maskSize=5).astype(np.uint8)
       
        #Try different colormaps an see the difference
        img = cv2.applyColorMap(img, cv2.COLORMAP_PLASMA)
        
        #For each class 0 we iterate joining the heatmap pixels to the image
        #The 0.7 and 0.3 after the pixel selection is for transparency. Try different values
        img3 = img2
        
        for _,row in box_coordinates.loc[box_coordinates["class"]==0,:].iterrows():
            
            img3  = cv2.addWeighted( img[row["ymin"]:row["ymax"],row["xmin"]:row["xmax"]], 0.7, img2[row["ymin"]:row["ymax"],row["xmin"]:row["xmax"]], 0.3, 0)
            img2[row["ymin"]:row["ymax"],row["xmin"]:row["xmax"]] = img3

        img = img2
        
    # Plot image with boxes
    if not skip_plot:
        if axes:
            axes.imshow(img)
        else:
            plt.imshow(img)   

    return img

#Non Max Supression: best bounding box
def NMS(img_tuple, boxes, overlapThresh = 0.4):
    
    """
    Receives `boxes` as a `numpy.ndarray` and gets the best bounding 
    box when there is overlapping bounding boxes.

    Parameters
    ----------
    boxes : numpy.ndarray
        Array with all the bounding boxes in the image.

    Returns
    -------
    best_bboxes: pd.DataFrame
        Dataframe with only the best bounding boxes, 
        in the format: ["xmin","ymin","xmax","ymax","class"]
    """
    
    #return an empty list, if no boxes given
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]  # x coordinate of the top-left corner
    y1 = boxes[:, 1]  # y coordinate of the top-left corner
    x2 = boxes[:, 2]  # x coordinate of the bottom-right corner
    y2 = boxes[:, 3]  # y coordinate of the bottom-right corner
    # compute the area of the bounding boxes and sort the bounding
    
    # boxes by the bottom-right y-coordinate of the bounding box
    areas = (x2 - x1 + 1) * (y2 - y1 + 1) # We have a least a box of one pixel, therefore the +1
    indices = np.arange(len(x1))
    for i,box in enumerate(boxes):

        
        temp_indices = indices[indices!=i]
        xx1 = np.maximum(box[0], boxes[temp_indices,0])
        yy1 = np.maximum(box[1], boxes[temp_indices,1])
        xx2 = np.minimum(box[2], boxes[temp_indices,2])
        yy2 = np.minimum(box[3], boxes[temp_indices,3])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / areas[temp_indices]
        
        if np.any(overlap) > overlapThresh:
            
            if box[4] == 0.0:
                continue                            #[ADDED]: Never delete missing boxxes

            indices = indices[indices != i]
            
    best_bboxes =   boxes[indices].astype(int)
    
    img_name = img_tuple[0]
    img_size = img_tuple[1]
    
    best_bboxes_df = pd.DataFrame(data = best_bboxes, index= [img_name]*len(best_bboxes), columns=["x1","y1","x2","y2","class"])
    best_bboxes_df['total_height'] = img_size[0]
    best_bboxes_df['total_width'] = img_size[1]
    
    
    return best_bboxes_df
        

## Computes the bounding boxes areas.
def get_bbox_area(box_coords: tuple, plot_f: bool = True):
    """ 
    It computes the bounding boxes compound area and the image total area.

    Parameters
    ----------
    box_coords: tuple
        Bounding box coordinates (x1,y1,x2,y2)
    plot_f: booL
        Wether to plot or not the image.
    Returns
    ----------
    areas: tuple
        Bounding box area and total area
    """

    x1, y1, x2, y2 = box_coords
    bb_area = (y1-y2)*(x1-x2)
    
    return bb_area


## Computes the fraction of area covered by the bounding boxes.
def get_bboxes_total_area(tags_df: pd.DataFrame):
    """ 
    Get the total area of the bounding boxes and the 
    proportion of area covered by the bounding boxes.

    Parameters
    ----------
    tag_df: pd.DataFrame
        The dataframe containing the images and it's tags. 
        
    Returns
    ----------
    aread_df: pd.DataFrame
        Dataframe with area related information.
    """    
    # Calculate total_area of the images
    tags_df['total_area'] = tags_df.total_height * tags_df.total_width

    # Calculate area of bboxes in the images
    tags_df['bbox_area'] = tags_df.apply(lambda r: get_bbox_area( (r.x1,r.y1,r.x2,r.y2) ), axis = 1)

    # See percentage of area covered by bboxes 
    tags_df['bbox_area_perc']  = tags_df.bbox_area / tags_df.total_area

    # Get only areas related columns
    areas_df = tags_df[ ['total_area','bbox_area','bbox_area_perc'] ]
    # areas_df.set_index('img_name', inplace= True)
    
    return areas_df