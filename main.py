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
import mediapipe as mp

st.title("Center Person Extractor using MediaPipe (Videos up to 30 seconds)")
st.write("Upload a video (max 30 seconds). The app will detect the person closest to the center and mask other areas.")

# Initialize MediaPipe Pose
@st.cache_resource
def load_pose_model():
    mp_pose = mp.solutions.pose
    return mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

pose = load_pose_model()

@st.cache_data
def process_video(video_path, w, h, fps, total_frames):
    output_path = os.path.join(tempfile.gettempdir(), "output_masked.mp4")
    cap = cv2.VideoCapture(video_path)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    frame_idx = 0
    progress_bar = st.progress(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            xs = [lm.x for lm in landmarks]
            ys = [lm.y for lm in landmarks]

            x_min = int(min(xs) * w)
            x_max = int(max(xs) * w)
            y_min = int(min(ys) * h)
            y_max = int(max(ys) * h)

            padding_x = int((x_max - x_min) * 0.1)
            padding_y = int((y_max - y_min) * 0.1)

            x_min_exp = max(0, x_min - padding_x)
            x_max_exp = min(w, x_max + padding_x)
            y_min_exp = max(0, y_min - padding_y)
            y_max_exp = min(h, y_max + padding_y)

            mask[y_min_exp:y_max_exp, x_min_exp:x_max_exp] = 255

        masked = cv2.bitwise_and(frame, frame, mask=mask)
        out.write(masked)

        frame_idx += 1
        progress_bar.progress(frame_idx / total_frames)

    cap.release()
    out.release()

    return output_path

video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_file.read())
        video_path = temp_video.name

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_sec = total_frames / fps
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if duration_sec > 30:
        st.error(f"❌ This video is {duration_sec:.1f} seconds long. Please upload a video shorter than 30 seconds.")
    else:
        st.success(f"Video accepted! Duration: {duration_sec:.1f} seconds")

        output_path = process_video(video_path, w, h, fps, total_frames)

        st.success("✅ Processing complete!")

        with open(output_path, "rb") as f:
            st.download_button("Download processed video", f, file_name="center_person_masked.mp4")
