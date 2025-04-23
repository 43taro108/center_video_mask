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
import subprocess

@st.cache_resource
def install_and_load_detectron2():
    try:
        import detectron2
    except ImportError:
        st.warning("Installing Detectron2... (this may take a few minutes)")
        subprocess.run(
            ["pip", "install", "git+https://github.com/facebookresearch/detectron2.git"],
            check=True
        )
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2 import model_zoo

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    return DefaultPredictor(cfg)

predictor = install_and_load_detectron2()

st.title("Center Person Extractor (Videos up to 30 seconds)")
st.write("Upload a video (max 30 seconds). The app will detect and extract the person closest to the center.")

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

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        output_path = os.path.join(tempfile.gettempdir(), "output_masked.mp4")
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        progress_bar = st.progress(0)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            outputs = predictor(frame)
            instances = outputs["instances"].to("cpu")
            is_person = (instances.pred_classes == 0).numpy()
            idxs = np.nonzero(is_person)[0].tolist()

            if len(idxs) == 0:
                out.write(np.zeros_like(frame))
            else:
                if len(idxs) > 1:
                    boxes = instances.pred_boxes.tensor.numpy()[idxs]
                    xs = (boxes[:, 0] + boxes[:, 2]) / 2
                    ys = (boxes[:, 1] + boxes[:, 3]) / 2
                    ds = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
                    target = idxs[int(np.argmin(ds))]
                else:
                    target = idxs[0]

                mask = instances.pred_masks[target].numpy().astype(np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=2)

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
