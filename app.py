import os
import cv2
import torch
import numpy as np
import csv
from PIL import Image
from flask import Flask, render_template, Response, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from transformers import DetrImageProcessor, DetrForObjectDetection
import logging
import time

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Setup logging to file for additional debug info
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s',
                    filename="detection.log", filemode="a")

# Load DETR model for pedestrian detection
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
model.eval()

PERSON_CLASS_ID = 1  # COCO ID for "person"
video_path = None
output_video_path = os.path.join(app.config["UPLOAD_FOLDER"], "processed_video.mp4")
use_camera = False

# Initialize CSV logging for detection stats
CSV_LOG_FILE = "detection_stats.csv"
if not os.path.exists(CSV_LOG_FILE):
    with open(CSV_LOG_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_number", "num_detections"])

def log_detections(frame_number, num_detections):
    """Logs the number of detections for each frame into a CSV file."""
    with open(CSV_LOG_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([frame_number, num_detections])
    logging.debug(f"Frame {frame_number}: {num_detections} detections logged.")

def send_alert_if_overcrowded(num_detections, threshold=5):
    """Sends an alert if the number of detected persons exceeds a threshold."""
    if num_detections >= threshold:
        # For now, we log the alert. In a real system, this could trigger an email or SMS.
        alert_msg = f"Alert: High pedestrian density detected ({num_detections} persons)!"
        logging.warning(alert_msg)
        print(alert_msg)

def generate_video_frames(source):
    cap = cv2.VideoCapture(source)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0

    if not use_camera:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        start_time = time.time()

        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(images=pil_image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([pil_image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.8)[0]

        detection_count = 0
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if label == PERSON_CLASS_ID:
                detection_count += 1
                x, y, x2, y2 = map(int, box.tolist())
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Pedestrian {score:.2f}", (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Log detections for this frame
        log_detections(frame_count, detection_count)
        send_alert_if_overcrowded(detection_count)

        if not use_camera:
            out.write(frame)

        end_time = time.time()
        logging.debug(f"Processed frame {frame_count} in {end_time - start_time:.3f} seconds.")
        frame_count += 1

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    cap.release()
    if not use_camera:
        out.release()

@app.route("/", methods=["GET", "POST"])
def index():
    global video_path, use_camera
    if request.method == "POST":
        use_camera = False
        if "file" in request.files:
            file = request.files["file"]
            if file.filename != "":
                filename = secure_filename(file.filename)
                video_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(video_path)
                logging.debug(f"Video file saved: {video_path}")
                return redirect(url_for("result"))
        if "use_camera" in request.form:
            use_camera = True
            return redirect(url_for("camera"))
    return render_template("index.html")

@app.route("/result")
def result():
    return render_template("result.html")

@app.route("/camera")
def camera():
    return render_template("camera.html")

@app.route("/video_feed")
def video_feed():
    if video_path and not use_camera:
        return Response(generate_video_frames(video_path), mimetype="multipart/x-mixed-replace; boundary=frame")
    return "No video uploaded."

@app.route("/camera_feed")
def camera_feed():
    return Response(generate_video_frames(0), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/download")
def download():
    return render_template("download.html", video_url=url_for("get_video", filename="processed_video.mp4"))

@app.route("/get_video/<filename>")
def get_video(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename, as_attachment=False)

if __name__ == "__main__":
    app.run(debug=True,port=8081)
