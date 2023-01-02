#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PREAMBLE
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import argparse
import os

import cons

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#  Python Arg Parser
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Prepare data folder")
    parser.add_argument(
        "--source_folder",
        type=str,
        help=(
            "Full path to the directory having all the store images. E.g. "
            "`/home/app/src/data/SKU110K/images/`."
        ),
        required= False
    )
    parser.add_argument(
        "--destination_folder",
        type=str,
        help=(
            "Full path to the directory where the reordered images will be stored"
            "w.g. `/home/app/src/data//train_test_SKU`"
        ),
        required= False
    )

    args = parser.parse_args()

    return args

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#  Script: Prepare Data
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def prepare_data(src_path: str = cons.SRC_PATH, dst_path: str = cons.DST_PATH):
    """ 
    Separates the images into train, test and validation subfolders.

    Parameters
    ----------
    src_path: str
        Source path from where data/images can be found. 
    dst_path: str
        Destination path where to store reordered images.
    """
    # run bash: python3 scripts/prepare_data.py "./data/SKU110K/images" "./data/train_test_SKU"
    
    for parent_dir,_,files in os.walk(src_path):
        
        for file in files:
            
            if file:  
                # Path to original image
                img_path = os.path.join(parent_dir,file)
                
                # Create correct target directory (train,test or validation folders)
                ttv = file.split("_")[0]
                trgt_folder = os.path.join(dst_path, ttv, 'images')
                trgt_path = os.path.join(trgt_folder,file)

                # Create the directory and link images
                os.makedirs(trgt_folder, exist_ok = True)
                if not os.path.exists(trgt_path):
                    print(file)
                    os.link(img_path,trgt_path)
                
             
if __name__ == "__main__":
    
    args = parse_args()
    
    if args._get_args():
        prepare_data(args.source_folder, args.destination_folder)                               
    else: 
        prepare_data()
