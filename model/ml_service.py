
#-----------------------------------------------------------------------------------------------------------------------------
# PREAMBLE
#-----------------------------------------------------------------------------------------------------------------------------
import json
import os
import time
import sys

import numpy as np
import pandas as pd
import cv2
import redis
import settings
from get_model import get_model

from utils_model.bboxes import plot_bboxes, NMS, euristic_detection

#-----------------------------------------------------------------------------------------------------------------------------
# INITIALIZATIONS
#-----------------------------------------------------------------------------------------------------------------------------

# Connect to Redis and assign to variable `db``
# Make use of settings.py module to get Redis settings like host, port, etc.
# Make use of settings.py module to get Redis settings like host, port, etc.
db = redis.Redis(
                host = settings.REDIS_IP ,
                port = settings.REDIS_PORT,
                db = settings.REDIS_DB_ID
                )

# Load your ML model and assign to variable `model`
model = get_model()


#-----------------------------------------------------------------------------------------------------------------------------
# FUNCTIONS
#-----------------------------------------------------------------------------------------------------------------------------

# Creating image with bboxes 
def predict_bboxes(img_name):
    """
    Loads the original image and logs the new image
    with the bounding boxes. It stores it a new folder
    called response. 

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    # Load original image
    orig_img_path = os.path.join(settings.UPLOAD_FOLDER,img_name)
    img_orig = cv2.imread(orig_img_path)
    
    # Get bounding boxes
    output = model(img_orig)
    df = output.pandas().xyxy[0]
    # df = df.sort_values("class")
    bboxes = df[["xmin","ymin","xmax","ymax","class"]]
    # Non-Max Supression: Filter only best bounding boxes
    best_bboxes = NMS(bboxes.to_numpy(), overlapThresh= settings.OVERLAP_THRESH)
    
    # Build image name and path
    extension = '.' + img_name.split('.')[-1]
    img_base_name = img_name.split('.')[:-1]
    
    ## 1. BBox
    img_name1 =  ''.join(img_base_name) + '_bbox' + extension
    pred_img_path = os.path.join(settings.PREDICTIONS_FOLDER, img_name1)  
    # Predict (draw all bounding boxes) and store
    img_pred = plot_bboxes(orig_img_path, box_coordinates= best_bboxes, skip_plot = True ) 
    cv2.imwrite(pred_img_path, img_pred)                    # store as: "predictions/<img_name_bbox.jpg>"
    
    ## 2. Heatmap 
    img_name2 =  ''.join(img_base_name) + '_heat' + extension
    pred_img_path = os.path.join(settings.PREDICTIONS_FOLDER, img_name2)  
    # Predict (draw all bounding boxes with heatmp) and store
    img_pred = plot_bboxes(orig_img_path, box_coordinates= best_bboxes, skip_plot = True, style = 'heatmap' ) 
    cv2.imwrite(pred_img_path,img_pred)                    # store as: "predictions/<img_name_heatmap.jpg>"
                        

def classify_process():
    """
    Loop indefinitely asking Redis for new jobs.
    When a new job arrives, takes it from the Redis queue, uses the loaded ML
    model to get predictions and stores the results back in Redis using
    the original job ID so other services can see it was processed and access
    the results.

    Load image from the corresponding folder based on the image name
    received, then, run our ML model to get predictions.
    """
    while True:
        
        # 1. Read the job from Redis
        _ , msg= db.brpop(settings.REDIS_QUEUE)                                                     # queue_name, msg <- 
        # print(f'Message from user: {msg}')
        
        # 2. Decode image_name
        msg_dict = json.loads(msg)
        img_name = msg_dict['image_name']
        job_id =  msg_dict['id']
        
        # 3. Predict
        predict_bboxes(img_name)
        
        pred_dict = {
                    "mAP": "[TO BE IMPLEMENTED]",
                    }
        
        # 4. Store in Redis
        db.set(job_id,json.dumps(pred_dict))

        # Don't forget to sleep for a bit at the end
        time.sleep(settings.SERVER_SLEEP)


if __name__ == "__main__":
    # Now launch process
    print("Launching ML service...")
    classify_process()
