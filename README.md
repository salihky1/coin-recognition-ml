# Turkish Coin Detection Projects

This repository contains two different approaches for detecting and calculating Turkish coin values from a camera feed.

# Project 1: Deep Learning Based Coin Classification

This project uses a Convolutional Neural Network (MobileNetV2) to classify Turkish coins (1 TL, 5 TL, 50 kuruş, 25 kuruş) and calculate the total value in real time.

# Features
- Real-time coin detection from camera stream  
- Background removal and coin segmentation  
- Deep learning model based on MobileNetV2  
- Automatic dataset collection  
- Data augmentation for robust training  
- Saves detection results to an Excel file  
- Calculates total money in Turkish Lira  

# Classes
- 1 TL  
- 5 TL  
- 50 Kuruş  
- 25 Kuruş  

# Requirements
- Python 3.x  
- OpenCV  
- NumPy  
- TensorFlow / Keras  
- OpenPyXL  

Install dependencies:
```bash
pip install opencv-python numpy tensorflow openpyxl
Usage (Project 1)
Run the script and select from the menu:

1 - Collect Data
2 - Train Model
3 - Real-Time Detection

The camera stream is taken from an IP camera or mobile phone camera using an HTTP stream URL.

Detected coins are classified by the trained model and their values are summed and displayed on the screen.
Each time the total value changes, it is saved to an Excel file.

Project 2: Classical Image Processing Based Coin Detection
This project detects Turkish coins using traditional image processing techniques without deep learning.

The system segments coins from the background and classifies them based on their contour area.

Features
Real-time coin detection

Otsu thresholding for segmentation

Morphological operations (erosion, hole filling)

Connected component analysis

Contour detection

Coin classification using area thresholds

Saves results to an Excel file

Calculates total money in Turkish Lira

Coin Classification (Area Based)
5 TL

1 TL

50 Kuruş

25 Kuruş

10 Kuruş

Coins are identified by comparing their contour area to predefined ranges.

Requirements (Project 2)
Python 3.x

OpenCV

NumPy

SciPy

OpenPyXL

Install dependencies:

pip install opencv-python numpy scipy openpyxl
Usage (Project 2)
Run the script directly:

python coin_detection_area.py
Coins are detected from the live camera feed and labeled according to their size.
The total detected value is displayed on the screen and logged into an Excel file.

Camera Setup
Both projects use an IP camera stream:

http://<IP_ADDRESS>:8080/video
You can use applications such as:

IP Webcam (Android)

DroidCam

Any IP camera providing MJPEG stream

Output
Real-time annotated video stream

Total detected money displayed on screen

Excel file containing:

Date

Time

Total Money (TL)

Comparison of Approaches
Deep Learning Approach:

More robust to lighting and angle changes

Requires dataset and training

Higher computational cost

Classical Image Processing Approach:

No training required

Faster and simpler

Sensitive to lighting and scale changes
