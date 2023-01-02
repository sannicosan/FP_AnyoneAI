import json
import time
from uuid import uuid4

import redis
import settings

# [DONE] Connect to Redis and assign to variable `db``
# Make use of settings.py module to get Redis settings like host, port, etc.
db = redis.Redis(
                host = settings.REDIS_IP ,
                port = settings.REDIS_PORT,
                db = settings.REDIS_DB_ID
                )


def model_predict(image_name):
    """
    Receives an image name and queues the job into Redis.
    Will loop until getting the answer from our ML service.

    Parameters
    ----------
    image_name : str
        Name for the image uploaded by the user.

    Returns
    -------
    prediction, score : tuple(str, float)
        Model predicted class as a string and the corresponding confidence
        score as a number.
    """

    # Assign an unique ID for this job and add it to the queue.
    # We need to assing this ID because we must be able to keep track
    # of this particular job across all the services
    job_id = str(uuid4())
    
    #  Create a dict with the job metadata we will send through Redis.
    job_data = json.dumps(  {
                            "id": job_id,
                            "image_name": image_name
                            }                           )

    #  Send the job to the model service using Redis
    
    db.lpush(
            settings.REDIS_QUEUE,
            job_data
            )
    

    # Loop until we received the response from our ML model
    while True:
        # Attempt to get model predictions using job_id
        output = db.get(job_id)                                                  
        # print(f'MSG from ML: {output}')
      
        if output:
            db.delete(job_id)
            break

        # Sleep some time waiting for model results
        time.sleep(settings.API_SLEEP)


    #  Change the output format
    output_dict =json.loads(output)                                                
    mAP = output_dict.values()
    
    return mAP
