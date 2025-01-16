# Real-Time Human Pose Estimation
This project demonstrates a real-time human pose estimation system using the MediaPipe Pose framework and OpenCV, developed as part of the AICTE Internship on AI: Transformative Learning (2024). It focuses on detecting and tracking human body keypoints in real-time, addressing challenges like dynamic movements, occlusions, and environmental variations.

## Features
* **33 Body Landmarks Detection:** Efficient and accurate keypoint detection using MediaPipe Pose.
* **Real-Time Visualization:** Annotated pose landmarks on live video feed via OpenCV.
* **Lightweight Framework:** Low computational overhead, suitable for desktop environments.
* **Applications:** Gesture recognition, fitness monitoring, and rehabilitation analysis.

## Installation
__Clone the Repository:__
```bash
git clone https://github.com/prathnashah/Edunet_Techsaksham_Phase-2_Human-Pose-Estimation.git
cd Edunet_Techsaksham_Phase-2_Human-Pose-Estimation
```

## Requirements
### Hardware
* A webcam or equivalent video capture device.
* A system with at least 4 GB RAM and an Intel i5 processor (or equivalent). A GPU is recommended for better performance.
### Software
* Python 3.7+
* Libraries: OpenCV, MediaPipe, Streamlit (optional for UI).

## Usage
1. Connect your webcam or camera device.
2. Run the program to start live pose estimation.
3. The detected body landmarks will be displayed in real-time on the video feed.

## Snapshots of Results
### Football Video Example:

### Yoga Video Example:

## Future Work
1. Enhance robustness for dynamic conditions (e.g., poor lighting, occlusions).
2. Implement real-time feedback for fitness and rehabilitation.
3. Extend for mobile devices and edge computing.
4. Integrate gesture recognition for expanded applications.

## References
1. Ming-Hsuan Yang, D. J. Kriegman, and N. Ahuja, “Detecting Faces in Images: A Survey,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 24, no. 1, 2002.
2. C. Lugaresi et al., “MediaPipe: A framework for building perception pipelines,” arXiv preprint arXiv:1906.08172, 2019.
3. Z. Cao et al., “OpenPose: Realtime multi-person 2D pose estimation using part affinity fields,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 43, no. 1, 2021.
4. B. Xiao, H. Wu, and Y. Wei, “Simple baselines for human pose estimation and tracking,” Proceedings of the ECCV, 2018.
