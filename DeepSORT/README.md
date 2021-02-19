
# Introduction

The purpose of this little project is to add object tracking to YOLO and Deep Sort and achieve real-time multiple object tracking


# Related repositories
  
  https://github.com/nwojke/deep_sort
  
  https://github.com/Qidian213/deep_sort_yolov3
  
  https://github.com/Akhtar303/Vehicle-Detection-and-Tracking-Usig-YOLO-and-Deep-Sort-with-Keras-and-Tensorflow
  
# Quick Start

1. Download Weights for Yolo and Deep Sort : https://drive.google.com/drive/folders/1zIncm9JVFY99a8wIXQ2MNgi12Wx7DwzH?usp=sharing
  
```

# Dependencies

The code is compatible with Python 2.7 and 3. The following dependencies are needed to run the tracker:

    NumPy
    sklean
    OpenCV

Additionally, feature generation requires TensorFlow-1.4.0
 
 
# Test
 use : 'video_capture = cv2.VideoCapture('path to video')' use a video file or 'video_capture = cv2.VideoCapture(0)' use camera
 
 speed : when only run yolo detection about 11-13 fps  , after add deep_sort about 11.5 fps
 
 
 
  It can also tracks Person too and performs well .
  
  before run inference define video path given to video capture function
  
  Run Inference :  python demo.py



