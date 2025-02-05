# Breast Lesion Detection Experiments on Breast Ultra-Sound Video Datasets Using Spatial Temporal model (STNet)
The goal of this project is to develop, evaluate, and experiment with STNET as a Spatial Temporal model for automatically identifying and localizing breast lesions in ultrasound videos.

Main Focus:

    - Asses STNet's ability to use Spatial-Temporal information for identifying and localizing breast lesions in ultrasound videos.
    - Evaluate the model’s performance on publicly available datasets.
    
## Usage
## Installation
### Requirements
* Linux, CUDA>=9.2, GCC>=5.4
  
* Python>=3.7

    We recommend you to use Anaconda to create a conda environment:
    ```bash
    conda create -n STNet python=3.7 pip
    ```
    Then, activate the environment:
    ```bash
    conda activate STNet
    ```
  
* PyTorch>=1.5.1, torchvision>=0.6.1 (following instructions [here](https://pytorch.org/))

    For example, if your CUDA version is 9.2, you could install pytorch and torchvision as following:
    ```bash
    conda install pytorch=1.5.1 torchvision=0.6.1 cudatoolkit=9.2 -c pytorch
    ```
  
* Other requirements
    ```bash
    pip install -r requirements.txt
    ```

### Compiling CUDA operators
```bash
cd ./models/ops
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```

### Dataset preparation

The dataset is provided by [CVA-Net](http://arxiv.org/abs/2207.00141), which is available for only non-commercial use in
research or educational purpose. 
As long as you use the database for these purposes, you can edit or process images and annotations in this database.
Please contact the authors of [CVA-Net](http://arxiv.org/abs/2207.00141) for getting the access to the dataset.
```
code_root/
Miccai 2022 BUV Dataset/
      ├── rawframes/
      ├── imagenet_vid_train_15frames.json
      ├── imagenet_vid_val.json
      ├── annotations/
          ├── instances_imagenet_vid_train_15frames.json
          └── instances_imagenet_vid_val.json
```

### Training

```bash
./tools/run_dist_launch.sh 1 ./configs/configs.sh
```


### Testing
Download trained weights [here](https://drive.google.com/file/d/1YteSJa9OO29YW7buzJ2LD2xjl-0BaiuB/view?usp=drive_link) and put it in `./checkpoints/`.
Then you can test it by running the following command:

```bash
./tools/run_dist_launch.sh 1 ./configs/test.sh 
```

## Notes
The code of this repository is built on https://github.com/AlfredQin/STNet/tree/main.
