import hashlib
import os
import settings 
from posixpath import splitext

def allowed_file(filename):
    """
    Checks if the format for the file received is acceptable. For this
    particular case, we must accept only image files. This is, files with
    extension ".png", ".jpg", ".jpeg" or ".gif".

    Parameters
    ----------
    filename : str
        Filename from werkzeug.datastructures.FileStorage file.

    Returns
    -------
    bool
        True if the file is an image, False otherwise.
    """
    extension = filename.split('.')[-1]
    return extension.lower() in settings.ALLOWED_EXTENSIONS and '.' in filename


def get_file_hash(file):
    """
    Returns a new filename based on the file content using MD5 hashing.
    It uses hashlib.md5() function from Python standard library to get
    the hash.

    Parameters
    ----------
    file : werkzeug.datastructures.FileStorage
        File sent by user.

    Returns
    -------
    str
        New filename based in md5 file hash.
    """
    #  DONE
    extension = '.' + file.filename.split('.')[-1].lower()
    
    # ## Checks if 'test\' directory comes before file.filename 
    #     ## -> When running the docker test:      file.filename := 'tests/bash
    #     # '
    #     ## -> When running from the Web UI:      file.filename := 'dog.jpeg' 
        
    # # if os.path.dirname(file.filename):
    # #     filepath = os.path.join(file.filename)
    # # else:
    # #     filepath = os.path.join('tests', file.filename)
    
    # filepath= os.path.join(settings.UPLOAD_FOLDER, file.filename)
    
    # # print(f'CURRENT WORKING DIRECTORY: {os.listdir()}')
    # with open(filepath,'rb') as f:
    #     data = f.read()
    #     name_md5 = hashlib.md5(data).hexdigest() + extension  # Example:  "0b3f45b266a97d7029dde7c2ba372093" + +".png"
    
    # return name_md5
    file_contents = file.read()
    md5hash = hashlib.md5(file_contents).hexdigest()
    extension = splitext(file.filename)[1]
    hashed_name = md5hash + extension
    file.seek(0)
    return hashed_name