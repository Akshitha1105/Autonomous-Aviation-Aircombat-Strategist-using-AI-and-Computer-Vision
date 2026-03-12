import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

import streamlit as st
import tempfile
import random
import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import gdown

from aircraft_data import aircraft_data
from strategy_engine import generate_strategy


# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Aviation Combat Strategist", layout="centered")

MODEL_PATH = "Weights/final_best.pt"
MODEL_URL = "https://drive.google.com/uc?id=1ILDTIAeLyfQQRDDp2uEXadXG-wBHqJgE"

RANDOM_IMG_DIR = "random_data/images"
RANDOM_VID_DIR = "random_data/videos"

MIN_MODEL_SIZE_MB = 200


# -----------------------------
# ENSURE DIRECTORIES
# -----------------------------
os.makedirs("Weights", exist_ok=True)
os.makedirs(RANDOM_IMG_DIR, exist_ok=True)
os.makedirs(RANDOM_VID_DIR, exist_ok=True)


# -----------------------------
# SAFE MODEL DOWNLOAD
# -----------------------------
def ensure_model():

    download_required = False

    if not os.path.exists(MODEL_PATH):
        download_required = True
    else:
        size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        if size < MIN_MODEL_SIZE_MB:
            os.remove(MODEL_PATH)
            download_required = True

    if download_required:

        with st.spinner("Downloading YOLO model (~300MB)... first run only"):

            gdown.download(
                MODEL_URL,
                MODEL_PATH,
                quiet=False
            )

        size = os.path.getsize(MODEL_PATH) / (1024 * 1024)

        if size < MIN_MODEL_SIZE_MB:
            raise RuntimeError(
                "Model download incomplete. Please reboot the Streamlit app."
            )


ensure_model()


# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


model = load_model()


# -----------------------------
# UI HEADER
# -----------------------------
st.title("✈️ Aviation Combat Strategist (ACS)")
st.caption("AI-powered aircraft detection and combat strategy recommendation system")

st.markdown("""
### How to Use
1️⃣ Select **your aircraft**  
2️⃣ Upload or randomly choose an **enemy aircraft image/video**  
3️⃣ Click **Run Detection** to generate combat strategy
""")


# -----------------------------
# SESSION STATE
# -----------------------------
if "selected_file" not in st.session_state:
    st.session_state.selected_file = None

if "file_type" not in st.session_state:
    st.session_state.file_type = None

if "file_name" not in st.session_state:
    st.session_state.file_name = None


# -----------------------------
# AIRCRAFT SELECTION
# -----------------------------
user_plane = st.selectbox(
    "Select Your Aircraft",
    list(aircraft_data.keys())
)


# -----------------------------
# RANDOM FILE SELECTION
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    if st.button("Use Random Image"):

        images = [
            f for f in os.listdir(RANDOM_IMG_DIR)
            if f.lower().endswith(("jpg", "jpeg", "png"))
        ]

        if images:

            selected = random.choice(images)

            st.session_state.selected_file = os.path.join(
                RANDOM_IMG_DIR,
                selected
            )

            st.session_state.file_type = "image"
            st.session_state.file_name = selected

        else:
            st.warning("No sample images found.")


with col2:
    if st.button("Use Random Video"):

        videos = [
            f for f in os.listdir(RANDOM_VID_DIR)
            if f.lower().endswith(("mp4", "avi", "mov"))
        ]

        if videos:

            selected = random.choice(videos)

            st.session_state.selected_file = os.path.join(
                RANDOM_VID_DIR,
                selected
            )

            st.session_state.file_type = "video"
            st.session_state.file_name = selected

        else:
            st.warning("No sample videos found.")


# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "Or upload enemy aircraft image/video",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
)

if uploaded_file:

    ext = uploaded_file.name.split(".")[-1].lower()

    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, f"uploaded.{ext}")

    with open(path, "wb") as f:
        f.write(uploaded_file.read())

    st.session_state.selected_file = path
    st.session_state.file_type = "image" if ext in ["jpg","jpeg","png"] else "video"
    st.session_state.file_name = uploaded_file.name


# -----------------------------
# DISPLAY INPUT
# -----------------------------
if st.session_state.selected_file:

    st.markdown(f"### Selected File: {st.session_state.file_name}")

    if st.session_state.file_type == "image":
        st.image(
            st.session_state.selected_file,
            use_column_width=True
        )

    else:
        st.video(
            st.session_state.selected_file
        )


# -----------------------------
# STRATEGY DISPLAY
# -----------------------------
def display_strategy(class_name):

    if class_name not in aircraft_data:

        st.warning(
            f"{class_name} not in knowledge base"
        )

        return

    strategy = generate_strategy(
        user_plane,
        class_name,
        aircraft_data
    )

    st.markdown(f"### Strategy vs {class_name}")

    st.write(
        f"**Win Probability:** {strategy['win_probability']*100:.1f}%"
    )

    st.write("**Advantages:**")

    for adv in strategy["advantages"]:
        st.write(f"- {adv}")

    st.write("**Disadvantages:**")

    for dis in strategy["disadvantages"]:
        st.write(f"- {dis}")

    if strategy["counter_strategy"]:
        st.write(
            f"**Counter Strategy:** {strategy['counter_strategy']}"
        )

    if strategy["escape_plan"]:
        st.write(
            f"**Escape Plan:** {strategy['escape_plan']}"
        )


# -----------------------------
# RUN DETECTION
# -----------------------------
if st.button("Run Detection"):

    file = st.session_state.selected_file
    file_type = st.session_state.file_type

    if not file:
        st.warning("Please select a file first.")
        st.stop()

    st.info("Running detection...")


    # -------------------------
    # IMAGE DETECTION
    # -------------------------
    if file_type == "image":

        results = model.predict(
            source=file,
            conf=0.25
        )

        r = results[0]

        if len(r.boxes) == 0:
            st.warning("No aircraft detected.")
            st.stop()

        img = r.plot()

        st.image(
            img,
            caption="Detected Aircraft",
            use_column_width=True
        )

        class_ids = [int(x) for x in r.boxes.cls.tolist()]

        detected = list(
            set([r.names[c] for c in class_ids])
        )

        st.markdown("### Detected Aircraft")

        for name in detected:

            st.subheader(name)

            display_strategy(name)


    # -------------------------
    # VIDEO DETECTION
    # -------------------------
    else:

        cap = cv2.VideoCapture(file)

        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        ) or 1

        temp_dir = tempfile.mkdtemp()

        output = os.path.join(
            temp_dir,
            "output.mp4"
        )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        out = cv2.VideoWriter(
            output,
            fourcc,
            fps,
            (width, height)
        )

        progress = st.progress(0)

        frame_count = 0
        frame_skip = 3

        class_counter = defaultdict(int)
        class_conf = defaultdict(list)

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            if frame_count % frame_skip != 0:
                continue

            results = model.predict(
                frame,
                conf=0.25,
                verbose=False
            )

            r = results[0]

            for box in r.boxes:

                cid = int(box.cls.item())
                conf = float(box.conf.item())

                name = r.names[cid]

                class_counter[name] += 1
                class_conf[name].append(conf)

            annotated = r.plot()

            out.write(annotated)

            progress.progress(
                min(frame_count / total_frames, 1.0)
            )

        cap.release()
        out.release()

        st.success("Detection complete")

        with open(output, "rb") as f:
            st.video(f.read())

        if not class_counter:
            st.warning("No aircraft detected.")
            st.stop()

        top = max(
            class_counter.items(),
            key=lambda x: x[1]
        )

        name, count = top

        avg_conf = np.mean(class_conf[name])

        st.markdown("### Top Detected Aircraft")

        st.subheader(
            f"{name} ({count} detections | avg conf {avg_conf:.2f})"
        )

        display_strategy(name)
