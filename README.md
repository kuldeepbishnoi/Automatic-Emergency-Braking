# Automatic Emergency Braking System
This repository contains code for a project whose goal is to Implement **Automatic Emergency Braking System** using a monocular camera. This model is trained on tusimple lane dataset.



## Algorithm Used: 
- LaneNet 
- DeepSORT (YOLO v3) 
- Automatic Emergency Braking System 


## Brief Intro
### LaneNet
LaneNet algorithm is a state of art deep convolution neural network which is used to detect lanes and it is implemented using tensorflow.
<p align="center">**LaneNet**<img src="Output_GIF/laneNet.gif"\></p>

### DeepSORT
We use YOLO v3 algorithm to perform vehicle detections. Then we take this output of YOLO v3 and feed it to these vehicle detections into DeepSORT in order to create a highly accurate vehicle tracker.
<p align="center">**DeepSORT**<img src="Output_GIF/deepSORT.gif"\></p>

### Automatic Emergency Braking System
<p align="center">**Automatic Emergency Braking System**<img src="Output_GIF/EBS.gif"\></p>


## Installation
Required package could be installed by running the given command.

1. Download Github Repository
2. Install [Anaconda](https://anaconda.org/) & Run the following commands in anaconda prompt
```
pip3 install -r requirements.txt
```

3. Activate the environment by using the given command
```
activate EBS
```

## Test model
You can test the given test frames on the trained model as follows
```
python test_ebs.py 
```
