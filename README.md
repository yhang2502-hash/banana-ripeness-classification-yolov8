# Vision-Based Banana Ripeness Classification Using YOLOv8

## Project Overview
This project is my Final Year Project for banana ripeness classification using YOLOv8 image classification. The system classifies banana ripeness into four categories: Green, Partially Ripe, Ripe, and Overripe.

The project compares three preprocessing methods:
1. Raw image input
2. Masked image with CLAHE enhancement
3. Isolated banana image with CLAHE enhancement

## Objectives
- To classify banana ripeness using computer vision and deep learning.
- To compare different image preprocessing methods.
- To evaluate the model using offline testing and real-time webcam testing.
- To build a simple real-time GUI for prediction display.

## Technologies Used
- Python
- YOLOv8
- OpenCV
- Tkinter
- Matplotlib
- CSV data logging

## Methodology
The system uses YOLOv8 classification with image preprocessing. Three methods were tested:

### Method 1: Raw Image
Original banana images were used directly.

### Method 2: Masked + CLAHE
A color mask was applied, followed by CLAHE enhancement.

### Method 3: Isolated + CLAHE
The banana region was isolated from the background and enhanced using CLAHE.

## Results
Offline testing showed that Method 2 achieved the highest accuracy at 91.99%.

Real-time webcam testing showed that Method 3 performed best with 94.62% accuracy, especially under different lighting conditions.

## Hardware Setup
The real-time testing setup included:
- Webcam
- Ring light
- Banana samples
- Laptop running Python GUI

## Project Demo
The GUI displays:
- Actual class
- Predicted class
- Accuracy
- Majority prediction
- Stability result
- Green highlight when prediction is correct and confidence is above the threshold

## Conclusion
Method 2 performed best in offline testing, while Method 3 was more suitable for real-time practical use because it was more robust to background and lighting changes.

## Author
Yu Hang Chok
