# Banana Ripeness Classification using YOLOv8

This is my Final Year Project (FYP) on vision-based banana ripeness classification using YOLOv8, OpenCV image preprocessing, and real-time camera GUI testing.

## Project Overview

The system classifies banana ripeness into four classes:

- Green
- Partially Ripe
- Ripe
- Overripe

## Methods

Three image processing methods were compared:

1. Raw image input
2. Masked image with CLAHE enhancement
3. Isolated banana image with CLAHE enhancement

## Technologies Used

- Python
- YOLOv8
- OpenCV
- Tkinter
- Matplotlib
- Pandas
- Computer Vision

## Results

Offline testing showed that Method 2 achieved the highest accuracy.

Real-time webcam testing showed that Method 3 performed better in practical camera testing because it was more robust to lighting and background conditions.

## Hardware Setup

The real-time testing setup included:

- Webcam
- Ring light
- Laptop
- Banana samples
- Python GUI system

## Folder Structure

```text
src/              - Main source code
preprocessing/    - Image preprocessing code
results/          - Result graphs and evaluation reports
hardware_setup/   - Hardware setup photos and notes
reports/          - FYP report, poster, or summary
sample_images/    - Small sample images only
