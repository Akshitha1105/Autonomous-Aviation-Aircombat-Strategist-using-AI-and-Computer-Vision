import streamlit as st
import tempfile
import os
import random
import cv2
import numpy as np
from collections import defaultdict
from PIL import Image
from ultralytics import YOLO
import time
import shutil

from aircraft_data import aircraft_data
from strategy_engine import generate_strategy

MODEL_PATH = os.path.join("Weights", "final_best.pt")
model = YOLO(MODEL_PATH)


# Paths
RANDOM_IMG_DIR = r"D:\AVIATION_COMBAT_STRTERGIST\random_data\images"
RANDOM_VID_DIR = r"D:\AVIATION_COMBAT_STRTERGIST\random_data\videos"

# --- STREAMLIT SETUP ---
st.set_page_config(page_title="Combat Strategy Module", layout="centered")
st.title(" Combat Strategy Module")

st.markdown("""
1. *Select your aircraft*  
2. *Upload or randomly select an image/video* of an enemy aircraft  
3. *Click "Run Detection" to identify the threat and get your strategy*
""")

# --- SESSION STATE ---
if "selected_file" not in st.session_state:
    st.session_state["selected_file"] = None
if "file_type" not in st.session_state:
    st.session_state["file_type"] = None
if "file_name" not in st.session_state:
    st.session_state["file_name"] = None

# --- AIRCRAFT SELECTION ---
user_plane = st.selectbox("Select Your Aircraft", list(aircraft_data.keys()))

# --- RANDOM IMAGE / VIDEO ---
col1, col2 = st.columns(2)
with col1:
    if st.button(" Use Random Image"):
        images = [f for f in os.listdir(RANDOM_IMG_DIR) if f.lower().endswith(("jpg", "jpeg", "png"))]
        if images:
            selected_image = random.choice(images)
            st.session_state["selected_file"] = os.path.join(RANDOM_IMG_DIR, selected_image)
            st.session_state["file_type"] = "image"
            st.session_state["file_name"] = selected_image
with col2:
    if st.button(" Use Random Video"):
        videos = [f for f in os.listdir(RANDOM_VID_DIR) if f.lower().endswith(("mp4", "avi", "mov"))]
        if videos:
            selected_video = random.choice(videos)
            st.session_state["selected_file"] = os.path.join(RANDOM_VID_DIR, selected_video)
            st.session_state["file_type"] = "video"
            st.session_state["file_name"] = selected_video

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader("Or upload your own enemy aircraft image/video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])
if uploaded_file:
    ext = uploaded_file.name.split('.')[-1].lower()
    temp_dir = tempfile.mkdtemp()
    input_path = os.path.join(temp_dir, f"uploaded.{ext}")
    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())
    st.session_state["selected_file"] = input_path
    st.session_state["file_type"] = "image" if ext in ["jpg", "jpeg", "png"] else "video"
    st.session_state["file_name"] = uploaded_file.name

# --- DISPLAY SELECTED FILE ---
if st.session_state["selected_file"]:
    st.markdown(f"###  Selected File: {st.session_state['file_name']}")
    if st.session_state["file_type"] == "image":
        st.image(st.session_state["selected_file"], caption="Selected Image", use_column_width=True)
    elif st.session_state["file_type"] == "video":
        st.video(st.session_state["selected_file"])

# --- STRATEGY DISPLAY ---
def display_strategy(class_name):
    if class_name in aircraft_data:
        strategy = generate_strategy(user_plane, class_name, aircraft_data)
        st.markdown(f"**Win Probability:** {strategy['win_probability'] * 100:.1f}%")
        st.markdown("**Advantages:**")
        for adv in strategy["advantages"]:
            st.markdown(f"- {adv}")
        st.markdown("**Disadvantages:**")
        for dis in strategy["disadvantages"]:
            st.markdown(f"- {dis}")
        if strategy["counter_strategy"]:
            st.markdown(f"**Counter Strategy:** {strategy['counter_strategy']}")
        if strategy["escape_plan"]:
            st.markdown(f"**Escape Plan:** {strategy['escape_plan']}")
    else:
        st.warning(f"{class_name} not found in knowledge base.")

# --- DETECTION ---
if st.button(" Run Detection"):
    selected_file = st.session_state["selected_file"]
    file_type = st.session_state["file_type"]

    if not selected_file:
        st.warning("No file selected.")
    else:
        st.info("Running detection... Please wait...")

        if file_type == "image":
            results = model.predict(source=selected_file, conf=0.25)
            r = results[0]

            if len(r.boxes) > 0:
                img = r.plot()
                st.image(img, caption="Detected Aircraft", use_column_width=True)
                class_ids = [int(x) for x in r.boxes.cls.tolist()]
                detected_classes = list(set([r.names[cid] for cid in class_ids]))
                st.markdown("###  Detected Aircraft:")
                for class_name in detected_classes:
                    st.subheader(class_name)
                    display_strategy(class_name)
            else:
                st.warning("No aircraft detected.")

        elif file_type == "video":
            cap = cv2.VideoCapture(selected_file)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            temp_dir = tempfile.mkdtemp()
            out_video_path = os.path.join(temp_dir, "detected_output.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

            progress_bar = st.progress(0)
            status_text = st.empty()
            live_frame_placeholder = st.empty()

            frame_count = 0
            processed_frames = 0
            start_time = time.time()

            class_counter = defaultdict(int)
            class_confidence = defaultdict(list)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model.predict(source=frame, conf=0.25, verbose=False)
                r = results[0]

                for box in r.boxes:
                    class_id = int(box.cls.item())
                    conf = float(box.conf.item())
                    class_name = r.names[class_id]
                    class_counter[class_name] += 1
                    class_confidence[class_name].append(conf)

                annotated_frame = r.plot()
                out.write(annotated_frame)

                live_frame_placeholder.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_column_width=True)

                processed_frames += 1
                elapsed = time.time() - start_time
                eta = (total_frames - processed_frames) * (elapsed / processed_frames) if processed_frames > 0 else 0
                progress_bar.progress(min(processed_frames / total_frames, 1.0))
                status_text.text(f"Processed: {processed_frames}/{total_frames} | ETA: {int(eta)}s")

            cap.release()
            out.release()
            cv2.destroyAllWindows()

            st.success("Detection complete!")

            #  Fix: Read video as bytes to ensure playback
            with open(out_video_path, 'rb') as video_file:
                video_bytes = video_file.read()
                st.markdown("###  Detected Video Output")
                st.video(video_bytes)

            if class_counter:
                top_class = max(class_counter.items(), key=lambda x: x[1])
                class_name, count = top_class
                avg_conf = np.mean(class_confidence[class_name])
                st.markdown("###  Top Detected Aircraft:")
                st.subheader(f"{class_name} — {count} times (Avg Conf: {avg_conf:.2f})")
                display_strategy(class_name)
            else:
                st.warning("No aircraft detected in the video.")