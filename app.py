import os
import torch
import shutil
import logging
import uuid
import tempfile
from PIL import Image
from flask import Flask, request, render_template, redirect, url_for, flash, session
from flask_cors import CORS
from torchvision import transforms
from torchvision.models import efficientnet_b0
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
# Change OpenCV import to use headless version
try:
    import cv2
except ImportError:
    # If standard cv2 fails, try installing headless version
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless"])
    import cv2
load_dotenv()

# ========== CONFIG ==========
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "deepfake_detection_secret")
UPLOAD_FOLDER = tempfile.mkdtemp()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB limit

origins = os.getenv("ALLOWED_ORIGINS", "").split(",")
CORS(app, resources={r"/*": {"origins": origins}})

# ========== LOGGING ==========
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== FACE DETECTION ==========
# Add the missing face_cascade definition
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# ========== MODEL LOAD ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Recreate model architecture to match training
model = efficientnet_b0(weights=None)
for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.classifier[1].in_features
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.5, inplace=True),
    torch.nn.Linear(num_ftrs, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(p=0.5),
    torch.nn.Linear(512, 1),
    torch.nn.Sigmoid()
)

model.load_state_dict(torch.load("deepfake_model_final.pth", map_location=device))
model.eval()
model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ========== HELPERS ==========
def allowed_file(filename):
    return filename.lower().endswith(('.mp4', '.avi', '.mov'))

def extract_faces_from_video(video_path, max_frames=10):
    faces = []
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, frame_count // max_frames)
    frame_id = 0

    while cap.isOpened() and len(faces) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            if len(detected) > 0:
                x, y, w, h = detected[0]
                face_crop = frame[y:y+h, x:x+w]
                if face_crop.size == 0:
                    continue
                image = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                image = transform(image).to(device)
                faces.append(image)
        frame_id += 1
    cap.release()
    return faces


def predict_deepfake(frames):
    if not frames:
        return {"prediction": "Unknown", "probability": 0.0}

    batch = torch.stack(frames)
    with torch.no_grad():
        outputs = model(batch)
        probs = outputs.squeeze()  # sigmoid already applied
        avg_fake_prob = probs.mean().item()

    label = "Fake" if avg_fake_prob > 0.5 else "Real"
    probability = avg_fake_prob * 100 if label == "Fake" else (1 - avg_fake_prob) * 100

    return {"prediction": label, "probability": round(probability, 2)}

# ========== ROUTES ==========
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        flash('No video uploaded', 'error')
        return redirect(url_for('index'))

    file = request.files['video']
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        try:
            filename = str(uuid.uuid4()) + "_" + secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            logger.info(f"Processing file: {filepath}")
            frames = extract_faces_from_video(filepath)
            result = predict_deepfake(frames)

            session['prediction'] = result['prediction']
            session['probability'] = result['probability']

            os.remove(filepath)
            return redirect(url_for('result'))

        except Exception as e:
            logger.exception("Error during video processing")
            flash(f"Server error: {str(e)}", 'error')
            return redirect(url_for('index'))

    flash('Invalid file type. Please upload a video.', 'error')
    return redirect(url_for('index'))

@app.route('/result')
def result():
    prediction = session.get('prediction')
    probability = session.get('probability')

    if prediction is None or probability is None:
        flash("No result available. Please upload a video.", 'error')
        return redirect(url_for('index'))

    return render_template('result.html', prediction=prediction, probability=probability)

@app.errorhandler(413)
def too_large(e):
    flash("File too large. Maximum allowed is 100MB.", 'error')
    return redirect(url_for('index'))

@app.errorhandler(500)
def internal_error(e):
    flash("Internal server error.", 'error')
    return redirect(url_for('index'))

# ========== CLEANUP ==========
import atexit
@atexit.register
def cleanup():
    shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)

if __name__ == '__main__':
   port = int(os.environ.get('PORT', 8080))
   app.run(host='0.0.0.0', port=port,debug=False)