# Automatic Emergency Braking System
This repository contains code for a project whose goal is to Implement **Automatic Emergency Braking System** using a monocular camera. This model is trained on [tusimple lane dataset](https://github.com/TuSimple/tusimple-benchmark).



## Algorithm Used: 
- LaneNet 
- DeepSORT (YOLO v3) 
- Automatic Emergency Braking System 


## Brief Intro
### LaneNet
LaneNet algorithm is a state of art deep convolution neural network which is used to detect lanes and is implemented using tensorflow.
<p align="center">
  LaneNet Output
  <img src="Output_GIF/laneNet.gif"\>
</p>

### DeepSORT
We use YOLO v3 algorithm to perform vehicle detections. Then we take this output feed it to DeepSORT in order to create a highly accurate vehicle tracker.
<p align="center">
  DeepSORT Output
  <img src="Output_GIF/deepSORT.gif"\>
</p>

### Automatic Emergency Braking System
This feature can sense incoming(traffic coming to ego lanes) and slow(as well as stopped) traffic ahead and urgently apply the brakes.
<p align="center">
  Automatic Emergency Braking System Output
  <img src="Output_GIF/EBS.gif"\>
</p>


## Installation
Required package could be installed by following the given steps.

1. Download Github Repository
2. Install [Anaconda](https://anaconda.org)
```
pip3 install -r requirements.txt
```


## Test model
You can test the provided test frames on the model by following the given steps.
1. Activate the environment in anaconda prompt by using the given command.
```
activate EBS
```
2.Run the test_ebs file by using the given command.
```
python test_ebs.py 
```
