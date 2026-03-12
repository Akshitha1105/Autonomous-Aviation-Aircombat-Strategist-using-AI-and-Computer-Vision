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
import gdown

from aircraft_data import aircraft_data
from strategy_engine import generate_strategy


# -----------------------------
# MODEL DOWNLOAD
# -----------------------------
MODEL_PATH = os.path.join("Weights", "final_best.pt")


def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs("Weights", exist_ok=True)

        url = "https://drive.google.com/uc?id=1ILDTIAeLyfQQRDDp2uEXadXG-wBHqJgE"
        st.info("Downloading model weights... (first run only)")
        gdown.download(url, MODEL_PATH, quiet=False)


download_model()


# -----------------------------
# LOAD YOLO MODEL (CACHED)
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


model = load_model()


# -----------------------------
# PATHS
# -----------------------------
RANDOM_IMG_DIR = "random_data/images"
RANDOM_VID_DIR = "random_data/videos"


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Aviation Combat Strategist", layout="centered")

st.title("✈️ Aviation Combat Strategist (ACS)")
st.caption("AI-powered aircraft detection and combat strategy recommendation system")

st.markdown("""
### How to Use
1️⃣ Select **your aircraft**  
2️⃣ Upload or randomly select an **enemy aircraft image/video**  
3️⃣ Click **Run Detection** to generate combat strategy
""")


# -----------------------------
# SESSION STATE
# -----------------------------
if "selected_file" not in st.session_state:
    st.session_state["selected_file"] = None

if "file_type" not in st.session_state:
    st.session_state["file_type"] = None

if "file_name" not in st.session_state:
    st.session_state["file_name"] = None


# -----------------------------
# AIRCRAFT SELECTION
# -----------------------------
user_plane = st.selectbox("Select Your Aircraft", list(aircraft_data.keys()))


# -----------------------------
# RANDOM INPUT
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    if st.button("Use Random Image"):

        images = [
            f for f in os.listdir(RANDOM_IMG_DIR)
            if f.lower().endswith(("jpg", "jpeg", "png"))
        ]

        if images:
            selected_image = random.choice(images)

            st.session_state["selected_file"] = os.path.join(
                RANDOM_IMG_DIR, selected_image
            )
            st.session_state["file_type"] = "image"
            st.session_state["file_name"] = selected_image


with col2:
    if st.button("Use Random Video"):

        videos = [
            f for f in os.listdir(RANDOM_VID_DIR)
            if f.lower().endswith(("mp4", "avi", "mov"))
        ]

        if videos:
            selected_video = random.choice(videos)

            st.session_state["selected_file"] = os.path.join(
                RANDOM_VID_DIR, selected_video
            )
            st.session_state["file_type"] = "video"
            st.session_state["file_name"] = selected_video


# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "Or upload enemy aircraft image/video",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
)

if uploaded_file:

    ext = uploaded_file.name.split(".")[-1].lower()

    temp_dir = tempfile.mkdtemp()
    input_path = os.path.join(temp_dir, f"uploaded.{ext}")

    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())

    st.session_state["selected_file"] = input_path
    st.session_state["file_type"] = "image" if ext in ["jpg", "jpeg", "png"] else "video"
    st.session_state["file_name"] = uploaded_file.name


# -----------------------------
# DISPLAY SELECTED FILE
# -----------------------------
if st.session_state["selected_file"]:

    st.markdown(f"### Selected File: {st.session_state['file_name']}")

    if st.session_state["file_type"] == "image":
        st.image(st.session_state["selected_file"], use_column_width=True)

    elif st.session_state["file_type"] == "video":
        st.video(st.session_state["selected_file"])


# -----------------------------
# STRATEGY DISPLAY
# -----------------------------
def display_strategy(class_name):

    if class_name in aircraft_data:

        strategy = generate_strategy(user_plane, class_name, aircraft_data)

        st.markdown(f"### Strategy vs {class_name}")

        st.markdown(f"**Win Probability:** {strategy['win_probability']*100:.1f}%")

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
        st.warning(f"{class_name} not in knowledge base.")


# -----------------------------
# DETECTION
# -----------------------------
if st.button("Run Detection"):

    selected_file = st.session_state["selected_file"]
    file_type = st.session_state["file_type"]

    if not selected_file:
        st.warning("No file selected.")
        st.stop()

    st.info("Running detection...")


# -----------------------------
# IMAGE DETECTION
# -----------------------------
    if file_type == "image":

        results = model.predict(source=selected_file, conf=0.25)

        r = results[0]

        if len(r.boxes) > 0:

            img = r.plot()

            st.image(img, caption="Detected Aircraft", use_column_width=True)

            class_ids = [int(x) for x in r.boxes.cls.tolist()]
            detected_classes = list(set([r.names[cid] for cid in class_ids]))

            st.markdown("### Detected Aircraft")

            for class_name in detected_classes:
                st.subheader(class_name)
                display_strategy(class_name)

        else:
            st.warning("No aircraft detected.")


# -----------------------------
# VIDEO DETECTION
# -----------------------------
    elif file_type == "video":

        cap = cv2.VideoCapture(selected_file)

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        temp_dir = tempfile.mkdtemp()

        out_video_path = os.path.join(temp_dir, "output.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        out = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

        progress_bar = st.progress(0)

        frame_count = 0

        class_counter = defaultdict(int)
        class_confidence = defaultdict(list)

        frame_skip = 3

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            if frame_count % frame_skip != 0:
                continue

            results = model.predict(frame, conf=0.25, verbose=False)

            r = results[0]

            for box in r.boxes:
                class_id = int(box.cls.item())
                conf = float(box.conf.item())
                class_name = r.names[class_id]

                class_counter[class_name] += 1
                class_confidence[class_name].append(conf)

            annotated = r.plot()

            out.write(annotated)

            progress_bar.progress(frame_count / total_frames)

        cap.release()
        out.release()

        st.success("Detection complete")

        with open(out_video_path, "rb") as f:
            st.video(f.read())

        if class_counter:

            top_class = max(class_counter.items(), key=lambda x: x[1])

            class_name, count = top_class

            avg_conf = np.mean(class_confidence[class_name])

            st.markdown("### Top Detected Aircraft")

            st.subheader(f"{class_name} ({count} detections | avg conf {avg_conf:.2f})")

            display_strategy(class_name)

        else:
            st.warning("No aircraft detected in video.")
