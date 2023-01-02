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
from utils_model import util_funcs 
from utils_model import cons
from enum import Enum
from PIL import Image

import settings

#from settings import CLASSES, COLORMAP


class CLASSES(Enum):
  PRODUCT = 3
  MISSING = 2

# COLORMAP PER CLASS
class COLORMAPS(Enum):
  PRODUCT = 'COLORMAP_TURBO'
  MISSING = 'COLORMAP_RAINBOW'



#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# COMMON FUNCTIONS
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
    print(img_name)
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
    
    if style == 'bbox':
        
        # Get BBox coordinates
        if box_coordinates.empty:
            box_coordinates = get_bboxes(img_path=img_path)
        
        # Plot all boxes
        for _, box_coords in box_coordinates.iterrows():
            
            x1, y1, x2, y2 = (box_coords.x1, box_coords.y1, box_coords.x2, box_coords.y2)
            if box_coords['class'] == CLASSES.PRODUCT.value:
                img = cv2.rectangle(img, (x1, y1), (x2, y2), cons.BLUE, thickness=5)
            else:
                img = cv2.rectangle(img, (x1, y1), (x2, y2), cons.RED, thickness=7)
                  
    
    elif style == 'heatmap':
        
        for cls in CLASSES:
          img = apply_heatmap(img,box_coordinates.loc[box_coordinates['class']==cls.value,:], getattr(cv2, COLORMAPS[cls.name].value))

    # Plot image with boxes
    if not skip_plot:
        if axes:
            axes.imshow(img)
        else:
            plt.imshow(img)   

    return img


#Non Max Supression: best bounding box
def NMS(boxes, overlapThresh = 0.4):
    
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
    
    
    best_bboxes_df = pd.DataFrame(data = best_bboxes, columns=["x1","y1","x2","y2","class"])
    
    
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



#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# FUNCTIONS written by @JoseCisneros
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def apply_heatmap(img, bboxes, colormap):
  h, w , _ = img.shape
  img_bw = np.zeros((h,w,1), np.uint8)
  for _, row in bboxes.iterrows():
    img_bw[row["y1"]:row["y2"], row["x1"]:row["x2"]] = 255
  img_bw = cv2.distanceTransform(img_bw, cv2.DIST_L1, maskSize=5).astype(np.uint8)
  img_bw = cv2.applyColorMap(img_bw, colormap)

  for _, row in bboxes.iterrows():
    merged_bbox  = cv2.addWeighted( img_bw[row["y1"]:row["y2"],row["x1"]:row["x2"]], 0.8, img[row["y1"]:row["y2"],row["x1"]:row["x2"]], 0.2, 0)
    img[row["y1"]:row["y2"],row["x1"]:row["x2"]] = merged_bbox

  return img

def euristic_detection(img_path: str, box_coordinates):
  #Read the image
  img = cv2.imread(img_path)
  x_min,x_max,y_min,y_max = box_coordinates["x1"].min(), box_coordinates["x2"].max(), box_coordinates["y1"].min(), box_coordinates["y2"].max()
  height, width, channels = img.shape

  white = np.zeros((height, width), np.uint8)

  for _,row in box_coordinates.iterrows():

      white  = cv2.rectangle(white, (row["x1"]+24, row["y1"]+6), (row["x2"]-24, row["y2"]-6),
          (255,0,0),-1)

  white =  255 - white

  crop = img[y_min:y_max, x_min:x_max]
  cv2.imwrite(settings.UPLOAD_FOLDER + "crop.png", crop)

  white = white[y_min:y_max, x_min:x_max] 
  dist = cv2.distanceTransform(white, cv2.DIST_L1 , maskSize=3).astype(np.uint8)

  heatmap_img = cv2.applyColorMap(dist, cv2.COLORMAP_JET)

  hsv=cv2.cvtColor(heatmap_img,cv2.COLOR_BGR2HSV)

  lowerValues = np.array([100, 50, 70])
  upperValues = np.array([128, 255, 255])

  bluepenMask = cv2.inRange(hsv, lowerValues, upperValues)


  heatmap_img[bluepenMask>0] = (255,255,255)

  cv2.imwrite(settings.UPLOAD_FOLDER +  "pivot.png", heatmap_img)

  pivot = Image.open(settings.UPLOAD_FOLDER +  "pivot.png")
  pivot = pivot.convert("RGBA")
  datas = pivot.getdata()
 
  newData = []
  print(type(datas))
  for item in datas:
      if item[0] == 255 and item[1] == 255 and item[2] == 255:
          newData.append((255, 255, 255, 0))
      else:
          newData.append(item)

  pivot.putdata(newData)
  # pivot.save("./New.png", "PNG")

  background = Image.open(settings.UPLOAD_FOLDER +  "crop.png")

  background.paste(pivot, (0, 0), pivot)

  background.save(settings.UPLOAD_FOLDER + "alpha_imposed.png", "PNG")


  #This is for contour
  heatmap_img = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2GRAY)


  ret, thresh = cv2.threshold(heatmap_img , 240, 255, cv2.THRESH_BINARY)
  # thresh = 255 - thresh
  # print(thresh)
  cv2.imwrite(settings.UPLOAD_FOLDER +  'output_imgtest.png', thresh  )

  
  #backtorgb = cv2.cvtColor(thresh,cv2.COLOR_GRAY2RGB)
  # backtorgb = cv2.merge((thresh,thresh,thresh))
  # print(backtorgb)
  # crop2 = crop + backtorgb
  crop2 = crop.copy()
  h,w= thresh.shape
  for y in range(0, h):
        for x in range(0, w):
            if thresh[y,x] < 255:
                crop2[y,x] = (0,0,255)

  cv2.imwrite(settings.UPLOAD_FOLDER + "crop2.jpg", crop2  )

  contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE ) #cv2.CHAIN_APPROX_NONE
  cv2.drawContours(crop , contours, -1, (0,0,255), 7)

  cv2.imwrite(settings.UPLOAD_FOLDER + "contour.jpg", crop  )

  return crop2

def euristic_detection2(img,name):
    '''
    recives path to images and saves two images, full heatmap and heatmap imposed
    '''
    
    img = cv2.imread(img)
    output = model(img)
    df = output.pandas().xyxy[0][["xmin","xmax","ymin","ymax"]].astype(int)
    x_min,x_max,y_min,y_max = df["xmin"].min(), df["xmax"].max(), df["ymin"].min(), df["ymax"].max()

    height, width, channels = img.shape

    white = np.zeros((height, width), np.uint8)

    for _,i in df.iterrows():

        white  = cv2.rectangle(white, (i["xmin"]+24, i["ymin"]+6), (i["xmax"]-24, i["ymax"]-6),
            (255,0,0),-1)

    white =  255 - white

    crop = img[y_min:y_max, x_min:x_max]
    cv2.imwrite("crop.png", crop)

    white = white[y_min:y_max, x_min:x_max] 
    dist = cv2.distanceTransform(white, cv2.DIST_L1 , maskSize=3).astype(np.uint8)

    heatmap_img = cv2.applyColorMap(dist, cv2.COLORMAP_JET)
    
    super_imposed_img = cv2.addWeighted(heatmap_img, 0.5, crop, 0.5, 0)
    cv2.imwrite(f"{name}_full_heat.jpg", super_imposed_img)


    
    hsv=cv2.cvtColor(heatmap_img,cv2.COLOR_BGR2HSV)

    lowerValues = np.array([100, 50, 70])
    upperValues = np.array([128, 255, 255])

    bluepenMask = cv2.inRange(hsv, lowerValues, upperValues)

    
    heatmap_img[bluepenMask>0] = (255,255,255)

    print(1)
    cv2.imwrite("pivot.png", heatmap_img)

    pivot = Image.open("pivot.png")
    pivot = pivot.convert("RGBA")
    datas = pivot.getdata()
 
    newData = []
    print(type(datas))
    for item in datas:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
    print(1)
    pivot.putdata(newData)
   # pivot.save("./New.png", "PNG")

    background = Image.open("crop.png")

    background.paste(pivot, (0, 0), pivot)

    background.save(f"./{name}_alpha_imposed.png", "PNG")


    #This is for contour
    heatmap_img = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2GRAY)


    ret, thresh = cv2.threshold(heatmap_img , 240, 255, cv2.THRESH_BINARY)
    cv2.imwrite('output_imgtest.png', thresh  )

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE ) #cv2.CHAIN_APPROX_NONE
    cv2.drawContours(crop , contours, -1, (0,0,255), 7)

    cv2.imwrite(f"{name}_contour.jpg", crop  )



