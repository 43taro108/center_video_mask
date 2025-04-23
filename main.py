# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 01:47:29 2025

@author: ktrpt
"""
import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO

st.title("Center Person Extractor using YOLOv5 (Videos up to 30 seconds)")
st.write("Upload a video (max 30 seconds). The app will detect and extract the person closest to the center.")

# Load YOLOv5 model
@st.cache_resource
def load_yolo_model():
    model = YOLO('yolov5s.pt')  # Use the small version for speed
    return model

model = load_yolo_model()

video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_file.read())
        video_path = temp_video.name

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_sec = total_frames / fps

    if duration_sec > 30:
        st.error(f"❌ This video is {duration_sec:.1f} seconds long. Please upload a video shorter than 30 seconds.")
        cap.release()
    else:
        st.success(f"Video accepted! Duration: {duration_sec:.1f} seconds")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cx, cy = w / 2, h / 2

        output_path = os.path.join(tempfile.gettempdir(), "output_masked.mp4")
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        progress_bar = st.progress(0)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB and pass as list
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model([rgb_frame])

            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()

            person_idxs = np.where(classes == 0)[0]  # Class 0 is person

            if len(person_idxs) == 0:
                out.write(np.zeros_like(frame))
            else:
                centers = []
                for idx in person_idxs:
                    x1, y1, x2, y2 = boxes[idx]
                    centers.append(((x1 + x2) / 2, (y1 + y2) / 2))
                centers = np.array(centers)
                ds = np.sqrt((centers[:,0] - cx)**2 + (centers[:,1] - cy)**2)
                target_idx = person_idxs[np.argmin(ds)]

                x1, y1, x2, y2 = boxes[target_idx].astype(int)
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                mask[y1:y2, x1:x2] = 255

                masked = cv2.bitwise_and(frame, frame, mask=mask)
                out.write(masked)

            frame_idx += 1
            progress_bar.progress(frame_idx / total_frames)

        cap.release()
        out.release()

        st.success("✅ Processing complete!")
        st.video(output_path)

        with open(output_path, "rb") as f:
            st.download_button("Download processed video", f, file_name="center_person_masked.mp4")
