# FINAL PROJECT
---------------------------------------------------
  Object detection for in-store inventory management.

## Team Members
---------------------------------------------------
  + Federico Saban - federicosaban10@gmail.com
  + José Cisneros - jcisneros@meicaperu.com
  + Jimmy Llaiqui - jim.llr01@gmail.com
  + Jonathan Castillo - jcasjar@gmail.com
  + Laura Argüello  - lau.bluee3@gmail.com
  + Nicolás Sánchez - nicolassanca95@gmail.com


# Data Preparation steps

1 - First download the data from S3 AWS with AWS CLI:

AWS CLI
- Install: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
- Configure: aws configure
- List data: aws s3 ls s3://anyoneai-datasets/SKU-110K/SKU110K_fixed/
- Download data: aws s3 sync s3://anyoneai-datasets/SKU-110K/SKU110K_fixed/ SKU110K_fixed

2 - Then we prepare the data into the correct folder structure: 

- LOCAL
```bash 
$ python3 scripts/prepare_data.py "./data/SKU110K/images" "./data/train_test_SKU"
```
- AWS
```bash 
$ python3 scripts/prepare_data.py "data/SKU110K/images" "data/train_test_SKU"
```

3 - Create the yolo dataset i.e folder structure and .txt files with the labels
- Execute the `build_yolo_dataset.ipynb` Jupyter notebook 

4 - Remove all failed and corrupted images (and tag .txt files)



# Project containter - Docker

## Install

You can use `Docker` to easily install all the needed packages and libraries:

- **CPU:**

```bash
$ docker build -t fp_sannicosan -f docker/Dockerfile .
$ sudo docker build -t fp_sannicosan --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -f docker/Dockerfile . #fp_sannicosan on AWS
```

- **GPU:**

```bash
$ docker build -t fp_sannicosan -f docker/Dockerfile_gpu .
$ sudo docker build -t fp_sannicosan --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -f docker/Dockerfile_gpu .  # on AWS
```

### Run Docker

- **CPU:**
```bash
$ docker run --rm --net host -it -v $(pwd):/home/app/src fp_sannicosan bash
```
- **GPU:**
```bash
$ sudo docker run --rm --ipc=host --gpus all -it -v $(pwd):/home/app/src --workdir /home/app/src fp_sannicosan 
$ sudo docker run --rm --ipc=host --gpus all -it -v $(pwd):/home/app/src --workdir /home/app/src fp_sannicosan  # on AWS
```
### Docker API Services

To run the services using compose:

```bash
$ docker-compose up --build -d
```
### Connect to AWS instance
ssh -i "~/.ssh/id_ed25519" sannicosan@ec2-3-135-229-79.us-east-2.compute.amazonaws.com
cd /home/sannicosan/


## Run Project

It doesn't matter if you are inside or outside a Docker container, in order to execute the project you need to launch a Jupyter notebook server running:

First, create the tuneL:
$  ssh -N -T -L 8880:localhost:8880 sannicosan@ec2-3-135-229-79.us-east-2.compute.amazonaws.com

```bash
$ jupyter notebook
```
or
$ sudo docker run -p 8880:8880 --rm --ipc=host -it -v "$(pwd):/home/app/src" --workdir /home/app/src fp_sannicosan bash 
$ jupyter notebook --port 8880 --ip 0.0.0.0


