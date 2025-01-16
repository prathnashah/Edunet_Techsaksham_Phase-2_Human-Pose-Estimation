import streamlit as st
from PIL import Image
import numpy as np
import cv2

# Constants
DEMO_IMAGE = "C:/Users/LENOVO/Desktop/Edunet Internship/Human Pose Estimation/images & videos/man_standing (3).png"
BODY_PARTS = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
}
POSE_PAIRS = [
    ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"],
    ["Neck", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"],
    ["Neck", "Nose"], ["Nose", "REye"], ["REye", "REar"],
    ["Nose", "LEye"], ["LEye", "LEar"]
]

INPUT_WIDTH, INPUT_HEIGHT = 368, 368

# Load pre-trained model
@st.cache_resource
def load_model():
    try:
        return cv2.dnn.readNetFromTensorflow("graph_opt.pb")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

net = load_model()

# Function to detect poses
@st.cache_data
def pose_detector(frame, threshold):
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (INPUT_WIDTH, INPUT_HEIGHT),
                                       (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()[:, :19, :, :]

    points = []
    for i, (part, idx) in enumerate(BODY_PARTS.items()):
        heat_map = out[0, idx, :, :]
        _, conf, _, point = cv2.minMaxLoc(heat_map)
        x = int(frame_width * point[0] / out.shape[3])
        y = int(frame_height * point[1] / out.shape[2])
        points.append((x, y) if conf > threshold else None)

    for part_from, part_to in POSE_PAIRS:
        id_from, id_to = BODY_PARTS[part_from], BODY_PARTS[part_to]
        if points[id_from] and points[id_to]:
            cv2.line(frame, points[id_from], points[id_to], (0, 255, 0), 3)
            for point in [points[id_from], points[id_to]]:
                cv2.ellipse(frame, point, (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

    return frame

# Streamlit UI
st.title("Human Pose Estimation with OpenCV")
st.text("Upload a clear image for accurate pose estimation.")

# File upload
img_file_buffer = st.file_uploader("Upload an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
image = np.array(Image.open(img_file_buffer)) if img_file_buffer else np.array(Image.open(DEMO_IMAGE))

# Display original image
st.subheader("Original Image")
st.image(image, caption="Original Image", use_column_width=True)

# Threshold slider
threshold = st.slider("Key Point Detection Threshold (%)", min_value=0, max_value=100, value=20, step=5) / 100

# Run pose detection
output = pose_detector(image, threshold)

# Display output
st.subheader("Pose Estimation Result")
st.image(output, caption="Estimated Pose", use_column_width=True)

st.markdown("Developed with ❤️ using OpenCV and Streamlit.")