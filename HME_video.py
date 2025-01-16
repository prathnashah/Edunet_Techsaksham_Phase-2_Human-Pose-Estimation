import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Capture Video: Replace with "path_to_video.mp4" for a file or 0 for webcam
video_source = "C:/Users/LENOVO/Desktop/Edunet Internship/Human Pose Estimation/images & videos/yoga_video.mp4"
cap = cv2.VideoCapture(video_source)

# Set Video Resolution for webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize MediaPipe Pose
with mp_pose.Pose(static_image_mode=False, model_complexity=0, enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video capture ended.")
            break
        
        # Flip the frame for a laterally mirrored display
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and get the pose landmarks
        results = pose.process(rgb_frame)

        # Draw pose landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display the frame
        cv2.imshow("Pose Detection", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
