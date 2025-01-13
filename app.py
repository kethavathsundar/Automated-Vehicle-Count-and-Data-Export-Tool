import streamlit as st
import cv2
import cvzone
import numpy as np
import math
import pandas as pd
from io import BytesIO
from ultralytics import YOLO
from sort import *

# Load YOLO model
model = YOLO("../Yolo_weights/yolov8n.pt")  # Ensure this path is correct or adjust it

# Define class names (for COCO dataset)
classNames = [
    'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa',
    'potted plant', 'bed', 'dining table', 'toilet', 'TV monitor', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# Define Streamlit interface
st.title("🚗 Automated Vehicle Count and Data Export Tool 🚗")
st.markdown("""
    **Welcome to the Vehicle Counting Application!**
    This application allows you to upload a video, count vehicles crossing a line, and export the data in CSV or Excel formats.
""")

uploaded_video = st.file_uploader("Upload a video file (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])

# Load the logo image
logo = cv2.imread('logo1.png')  # Adjust the path to your logo image file
logo = cv2.resize(logo, (150, 150))  # Resize the logo for better placement

# Data collection for export
vehicle_data = []

if uploaded_video is not None:
    # Save uploaded video to a temporary file
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_video.read())

    # Initialize video capture
    cap = cv2.VideoCapture("temp_video.mp4")

    # Load mask (optional)
    mask = cv2.imread('mask1.jpg')

    # Initialize tracker and limits
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    limits = [300, 520, 970, 520]  # Line coordinates
    totalcount = []
    line_color = (0, 0, 255)  # Default line color (Red)
    crossed_ids = set()  # Set to keep track of vehicles that have crossed the line

    # Create a placeholder for the live video feed
    video_placeholder = st.empty()

    # Display vehicle count dynamically
    count_placeholder = st.empty()

    # Start a loop to process the video in real-time
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        # Apply mask
        if mask is not None:
            imgRegion = cv2.bitwise_and(img, mask)
        else:
            imgRegion = img

        # Process YOLO results
        results = model(imgRegion, stream=True)

        detections = np.empty((0, 5))
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])

                if cls < len(classNames) and classNames[cls] == "car" and conf > 0.3:
                    detections = np.vstack((detections, [x1, y1, x2, y2, conf]))

        # Update tracker
        resultsTracker = tracker.update(detections)

        # Initialize a flag for line color reset
        line_color = (0, 0, 255)  # Reset line color to red at the start of each frame

        # Check if any vehicle has crossed the line
        for result in resultsTracker:
            x1, y1, x2, y2, id = map(int, result)
            cx, cy = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2
            
            if limits[0] < cx < limits[2] and abs(cy - limits[1]) < 5:
                if id not in totalcount and id not in crossed_ids:
                    totalcount.append(id)
                    crossed_ids.add(id)
                    line_color = (0, 255, 0)  # Change line color to green when a vehicle crosses

            # Draw the rectangle and the vehicle ID
            cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), l=9, rt=5, colorR=(255, 0, 255))
            cvzone.putTextRect(img, f'{id}', (max(0, x1), max(20, y1)), scale=2, thickness=3, offset=10)

            # Draw a circle at the center of the bounding box
            cv2.circle(img, (cx, cy), 3, (0, 0, 255), cv2.FILLED)

            # Collect vehicle data (ID and timestamp)
            vehicle_data.append({"Vehicle ID": id, "Timestamp": cap.get(cv2.CAP_PROP_POS_MSEC)})

        # Draw the line (green when a vehicle crosses, red otherwise)
        cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), line_color, 5)

        # Overlay the logo at the top-left corner
        img[10:160, 10:160] = logo  # Place logo on the top-left corner (adjust size as needed)

        # Add the counter text next to the logo
        counter_text = f"Count: {len(totalcount)}"
        cv2.putText(img, counter_text, (170, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Update the live video feed in Streamlit
        video_placeholder.image(frame_rgb, caption="Live Video Feed", use_container_width=True)

    # Display final count after processing
    st.success(f"Total Vehicle Count: {len(totalcount)}")

    # Release resources
    cap.release()

    # Provide a button to download the vehicle data
    st.write("### Download Vehicle Data")
    data_df = pd.DataFrame(vehicle_data)  # Convert vehicle data to DataFrame

    # Export to CSV
    csv_file = data_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV",
        data=csv_file,
        file_name="vehicle_data.csv",
        mime="text/csv",
    )

    # Export to Excel
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        data_df.to_excel(writer, index=False)
    excel_buffer.seek(0)

    st.download_button(
        label="Download Excel",
        data=excel_buffer,
        file_name="vehicle_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
