import os
import json
import uuid
import threading
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import cv2
from ultralytics import YOLO
from gps_reader import read_gps_from_frame
from math import radians, sin, cos, sqrt, atan2
from collections import defaultdict
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

CONFIDENCE_THRESHOLD = 0.80
DEDUPE_DISTANCE_M = 8.0
FRAME_SKIP = 10
ROAD_CROP_TOP = 0.40
ROAD_CROP_BOTTOM = 0.90
GENERAL_CLASSES = {"person", "car", "motorcycle", "bus", "truck", "bicycle"}

general_model = None
pothole_model = None

jobs = {}

def get_models():
    global general_model, pothole_model
    if general_model is None:
        general_model = YOLO("yolov8n.pt")
    if pothole_model is None:
        pothole_model = YOLO("pothole_v2.pt")
    return general_model, pothole_model

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1-a))

def is_duplicate(gps, logged_list):
    if not gps:
        return False
    for (plat, plon) in logged_list:
        if haversine_m(gps[0], gps[1], plat, plon) < DEDUPE_DISTANCE_M:
            return True
    return False

def process_video(job_id, video_path, video_name, manual_location, user_id):
    jobs[job_id]['status'] = 'processing'
    try:
        result_dir = os.path.join(app.config['RESULTS_FOLDER'], job_id)
        os.makedirs(result_dir, exist_ok=True)
        images_dir = os.path.join(result_dir, 'pothole_images')
        os.makedirs(images_dir, exist_ok=True)

        general_model, pothole_model = get_models()

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        last_good_gps = manual_location
        logged_by_class = defaultdict(list)
        detections = []
        pothole_count = 0
        class_counts = defaultdict(int)
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % FRAME_SKIP != 0:
                frame_idx += 1
                continue

            progress = int((frame_idx / max(total_frames, 1)) * 100)
            jobs[job_id]['progress'] = progress

            timestamp_s = frame_idx / fps
            gps = read_gps_from_frame(frame)
            if gps is None:
                gps = last_good_gps
            else:
                if 47.5 <= gps[0] <= 48.5 and 106.0 <= gps[1] <= 108.0:
                    last_good_gps = gps
                else:
                    gps = last_good_gps

            h, w = frame.shape[:2]

            gen_results = general_model.predict(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)[0]
            for box in gen_results.boxes:
                cls_name = general_model.names[int(box.cls[0])]
                if cls_name not in GENERAL_CLASSES:
                    continue
                if is_duplicate(gps, logged_by_class[cls_name]):
                    continue
                if gps:
                    logged_by_class[cls_name].append(gps)
                class_counts[cls_name] += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    "class": cls_name,
                    "confidence": round(float(box.conf[0]), 3),
                    "lat": gps[0] if gps else None,
                    "lon": gps[1] if gps else None,
                    "timestamp_s": round(timestamp_s, 2),
                    "image": None,
                })

            road_crop = frame[int(h*ROAD_CROP_TOP):int(h*ROAD_CROP_BOTTOM), :]
            pot_results = pothole_model.predict(road_crop, verbose=False, conf=CONFIDENCE_THRESHOLD)[0]
            for box in pot_results.boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                y1 = int(y1 + h * ROAD_CROP_TOP)
                y2 = int(y2 + h * ROAD_CROP_TOP)

                if is_duplicate(gps, logged_by_class["pothole"]):
                    continue
                if gps:
                    logged_by_class["pothole"].append(gps)

                pothole_count += 1
                crop_img = frame[max(0,y1):y2, max(0,x1):x2]
                img_filename = f"pothole_{pothole_count:04d}.jpg"
                img_path = os.path.join(images_dir, img_filename)
                cv2.imwrite(img_path, crop_img)

                with open(img_path, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode()

                detections.append({
                    "class": "pothole",
                    "confidence": round(conf, 3),
                    "lat": gps[0] if gps else None,
                    "lon": gps[1] if gps else None,
                    "timestamp_s": round(timestamp_s, 2),
                    "image": img_b64,
                })

            frame_idx += 1

        cap.release()
        if os.path.exists(video_path):
            os.remove(video_path)

        summary = {"pothole": pothole_count}
        summary.update(class_counts)

        result = {
            "job_id": job_id,
            "video_name": video_name,
            "user_id": user_id,
            "summary": summary,
            "detections": detections,
        }
        with open(os.path.join(result_dir, "result.json"), "w") as f:
            json.dump(result, f)

        jobs[job_id]['status'] = 'done'
        jobs[job_id]['progress'] = 100

    except Exception as e:
        print(f"ERROR processing job {job_id}: {e}")
        import traceback
        traceback.print_exc()
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['error'] = str(e)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("video")
    manual_lat = request.form.get("lat", "")
    manual_lon = request.form.get("lon", "")
    video_name = request.form.get("name", file.filename if file else "Untitled")
    user_id = request.form.get("user_id", "anonymous")

    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    job_id = str(uuid.uuid4())
    filename = secure_filename(file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_{filename}")
    file.save(video_path)

    manual_location = None
    if manual_lat and manual_lon:
        try:
            manual_location = (float(manual_lat), float(manual_lon))
        except ValueError:
            pass

    jobs[job_id] = {"status": "queued", "progress": 0}
    thread = threading.Thread(target=process_video, args=(job_id, video_path, video_name, manual_location, user_id))
    thread.daemon = True
    thread.start()

    return jsonify({"job_id": job_id})

@app.route("/status/<job_id>")
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)

@app.route("/result/<job_id>")
def result(job_id):
    result_path = os.path.join(app.config['RESULTS_FOLDER'], job_id, "result.json")
    if not os.path.exists(result_path):
        return jsonify({"error": "Result not found"}), 404
    with open(result_path) as f:
        return jsonify(json.load(f))

@app.route("/sessions")
def sessions():
    user_id = request.args.get('user_id', 'anonymous')
    sessions_list = []
    for job_id in os.listdir(app.config['RESULTS_FOLDER']):
        result_path = os.path.join(app.config['RESULTS_FOLDER'], job_id, "result.json")
        if os.path.exists(result_path):
            with open(result_path) as f:
                data = json.load(f)
                if data.get('user_id') == user_id:
                    sessions_list.append({
                        "job_id": job_id,
                        "video_name": data.get("video_name", "Untitled"),
                        "pothole_count": data.get("summary", {}).get("pothole", 0),
                    })
    return jsonify(sessions_list)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)