import os
import cv2
import dlib

# Load dlib's face detector
detector = dlib.get_frontal_face_detector()

def extract_faces_from_videos(source_folder, target_folder, frames_per_video=5):
    os.makedirs(target_folder, exist_ok=True)
    videos = os.listdir(source_folder)

    for video in videos:
        if not video.endswith(".mp4"):
            continue

        video_path = os.path.join(source_folder, video)
        cap = cv2.VideoCapture(video_path)

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(frame_count // frames_per_video, 1)

        frame_id = 0
        saved = 0
        while cap.isOpened() and saved < frames_per_video:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_id % interval == 0:
                # Detect faces
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector(gray)

                for i, face in enumerate(faces):
                    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()

                    # Clamp coordinates to valid region
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                    cropped_face = frame[y1:y2, x1:x2]
                    resized_face = cv2.resize(cropped_face, (224, 224))

                    # Save face image
                    filename = f"{video[:-4]}_frame{frame_id}_face{i}.png"
                    output_path = os.path.join(target_folder, filename)
                    cv2.imwrite(output_path, resized_face)
                    saved += 1

                    if saved >= frames_per_video:
                        break
            frame_id += 1

        cap.release()
# Extract to new folders
extract_faces_from_videos("dataset/real", "frames/real", frames_per_video=5)
extract_faces_from_videos("dataset/fake", "frames/fake", frames_per_video=5)
