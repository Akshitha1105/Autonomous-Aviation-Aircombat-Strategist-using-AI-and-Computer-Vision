# Aviation Combat Strategist (ACS)

Aviation Combat Strategist (ACS) is an AI-powered system that detects enemy aircraft in real-time and suggests tactical combat strategies based on the aircraft type and attributes. It uses deep learning for object detection and a rule-based engine for strategic decision-making.

---

## Overview

This project integrates a YOLOv8 object detection pipeline with a Streamlit-based interface to analyze aircraft data and generate effective combat strategies. It supports both image and video inputs and provides insights such as win probability, advantages, disadvantages, and counter/escape plans.

---

## Features

- Aircraft detection using YOLOv8
- Handles both images and videos without FFmpeg
- Strategy recommendation based on aircraft capabilities
- User-friendly Streamlit interface
- Real-time visual feedback with annotated detections

---

## Dataset

The model was trained on the following publicly available dataset:

**Military Aircraft Detection Dataset**  
Kaggle: [https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset](https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset)

---
### Download YOLOv8 Model Weights

Due to GitHub file size limits, the trained model weights (`final_best.pt`) are hosted externally.

[ Download final_best.pt from Google Drive](https://drive.google.com/file/d/1ILDTIAeLyfQQRDDp2uEXadXG-wBHqJgE/view?usp=sharing)


## Technologies Used

| Component        | Library/Tool           |
|------------------|------------------------|
| Detection Model  | YOLOv8 (Ultralytics)   |
| User Interface   | Streamlit              |
| Image Processing | OpenCV, PIL            |
| Logic Layer      | Python, NumPy, Pandas  |

---

## Folder Structure

```bash
AVIATION_COMBAT_STRATEGIST/
├── app.py                          # Streamlit frontend app
├── aircraft_data.py                # Aircraft knowledge base
├── strategy_engine.py              # Rule-based strategy logic
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── .gitignore                      # Git ignore rules

├── Weights/
│   └── final_best.pt               # Trained YOLOv8 model weights

├── random_data/
│   ├── images/                     # Sample aircraft images
│   └── videos/                     # Sample aircraft videos

├── notebooks/
│   └── aviation_combat_training.ipynb  # YOLOv8 Colab training notebook

├── __pycache__/                    # Python bytecode cache (ignored)
```

---

## Setup Instructions

### 1. Clone the Repository

git clone https://github.com/Madukalmanoj/Autonomous_Aviation_Combat_Strategist.git

cd Autonomous_Aviation_Combat_Strategist

### 2. Install requirements:

pip install -r requirements.txt

### 3. Launch the app:

streamlit run app.py

## Demo Media

Example input files are located in the random_data/images/ and random_data/videos/ directories. You can use them for testing the system’s performance and strategy output.

## Team Members

| Name | Role |
|------|------|
| [Madukal Manoj](https://github.com/Madukalmanoj) | Lead ML Engineer – YOLOv8 Training & Testing |
| [Benoorkar Akshitha](https://github.com/Akshitha1105) | Strategy Engine Developer |
| [Thudimalla Vaishnavi](https://github.com/thudimillavaishnavi26) | Data Specialist – Cleaning & Manual Annotation |
| [P. Ananth](https://github.com/Ananthx66) | Knowledge Base Developer – Aircraft Data Module |
| [A. Varun](https://github.com/Appalavarun) | UI Developer – Streamlit App Creator |



Each teammate has hosted this project independently on their GitHub profile as part of the academic submission process.
