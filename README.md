# Automated Vehicle Count and Data Export Tool 🚗

## Overview
The **Automated Vehicle Count and Data Export Tool** is a computer vision application built using **YOLOv8**, **OpenCV**, and **SORT** (Simple Online and Realtime Tracking) algorithms. It allows you to upload a video, automatically count vehicles crossing a line, and export the vehicle data in CSV or Excel formats. 

This tool is designed for applications such as traffic monitoring, vehicle flow analysis, and automated vehicle tracking in videos.

---

## Features
- **Vehicle Counting**: Counts vehicles (cars, trucks, etc.) that cross a predefined line in the video.
- **Real-Time Processing**: Processes video frame by frame and provides real-time feedback.
- **Data Export**: Allows users to export the detected vehicle data in CSV and Excel formats.
- **Customizable Mask**: Optionally apply a mask to define the area for vehicle tracking.
- **Interactive Interface**: Streamlit-based user interface for easy interaction and display of results.

---

## Requirements

### Python Libraries
Make sure you have Python 3.7 or higher installed and the following libraries:

1. **Streamlit**: For creating the interactive web interface.
2. **OpenCV**: For video processing and manipulation.
3. **cvzone**: For advanced OpenCV functions like drawing bounding boxes and vehicle IDs.
4. **Ultralytics YOLO**: For detecting vehicles in video frames.
5. **pandas**: For handling and exporting vehicle data.
6. **sort**: For tracking the vehicles across frames.
7. **openpyxl**: For exporting data to Excel.

To install the required libraries, you can use `pip`:

```bash
pip install streamlit opencv-python cvzone ultralytics pandas openpyxl sort
```
## Setup Instructions
** Step 1**: Prepare YOLO Weights
   Download the pre-trained YOLOv8 weights file (e.g., yolov8n.pt) from Ultralytics YOLO repository and place it in the directory ../Yolo_weights/
** Step 2**:Prepare Mask Image (Optional)
If you want to apply a region-of-interest mask, ensure you have a mask1.jpg image in the project directory. The mask will restrict vehicle detection to a specific area in the video.

**Step 3**:Launch the Application
```
streamlit run app.py
```
## How to Use
**Step 1**:Upload Video
1. Click on the "Upload Video" button to upload a video file (MP4, AVI, MOV formats supported).
2. The video will be processed in real-time, and vehicles will be detected and counted
**Step 2**:View Vehicle Count
1. As the video is processed, the number of vehicles crossing the line will be displayed in real-time.
2. The vehicles crossing the line will be highlighted with bounding boxes and unique IDs.
** Step 3**:Data Export
Once the video is processed:
1. The total count of vehicles is displayed at the top of the screen.
2. ou can download the vehicle data in either CSV or Excel format by clicking the respective download buttons.

## Video Processing Details
The application uses YOLOv8 to detect vehicles in the video frames.
The SORT algorithm is used to track vehicles and maintain unique IDs for each detected vehicle.
A predefined line is set in the video, and when a vehicle crosses this line, it's counted and recorded.

##Project Structure
```
/vehicle-counting-tool
│
├── app.py                     # Main Streamlit app
├── logo1.png                  # Application logo
├── mask1.jpg                  # Optional mask for region-of-interest (can be omitted)
├── requirements.txt           # List of Python dependencies
└── /Yolo_weights              # Folder containing YOLOv8 weights (e.g., yolov8n.pt)
```

   

