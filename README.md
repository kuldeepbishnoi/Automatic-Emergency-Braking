# Automatic Emergency Braking System
This repository contains code for a project whose goal is to Implement **Automatic Emergency Braking System** using a monocular camera. This model is trained on tusimple lane dataset.

#### Automonomus Emergency Braking System
<p align="center"><img src="Output_GIF/EBS.gif"\></p>


## Algorithm Used: 
- LaneNet 
- DeepSORT (YOLO v3) 
- Automatic Emergency Braking System 


## Brief Intro
### LaneNet
LaneNet algorithm is a state of art deep convolution neural network which is used to detect lanes and it is implemented using tensorflow.

#### YOLO v4 - LaneNet
<p align="center">YOLO v4 - LaneNet<img src="Output_GIF/laneNet.gif"\></p>


### DeepSORT
YOLOv4 algorithm also uses deep convolutional neural networks to perform object detections. Then we take this output of YOLOv4 feed these object detections into DeepSORT in order to create a highly accurate vehicle tracker.

#### YOLO v4 - DeepSORT
<p align="center"><img src="Output_GIF/deepSORT.gif"\></p>


## Installation
Required package could be installed by running the given command.

1. Download Github Repository
2. Install Anaconda & Run the following commands in anaconda prompt
```
pip3 install -r requirements.txt
```

3. Activate the environment by using the given command
```
activate EBS
```

4.(If you want to run your own dataset)
## Test model
You can test a video on the trained model as follows

```
python tools/test_lanenet.py --weights_path /PATH/TO/YOUR/CKPT_FILE_PATH 
--image_path ./data/tusimple_test_image/0.jpg
```
