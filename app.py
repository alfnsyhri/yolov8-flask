import io, os, time
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import requests

# ================================================
# KONFIGURASI
# ================================================
MODEL_PATH = "best.pt"
SAVE_FOLDER = "static/capture/"
PHP_SAVE_URL = "http://kel3.myiot.fun/public_html/save_prediction.php"

# === IP ESP32 DEV UNTUK KENDALI MOTOR ============
ESP32_DEV_URL = "http://10.98.0.32/motor"

CLASS_NAMES = ["busuk", "matang", "tidak_matang"]
CONF_THRESHOLD = 0.25

# ================================================
# FLASK APP
# ================================================
app = Flask(__name__)
CORS(app)

# Route untuk Render health check
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "YOLO Flask API Running"})

# ================================================
# LOAD YOLO
# ================================================
print("Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("Model Loaded!")

# ================================================
# VAR GLOBAL
# ================================================
last_output = {
    "label": "none",
    "confidence": 0,
    "image_path": "",
    "timestamp": ""
}

# ================================================
# BACA RAW BYTE → PIL
# ================================================
def read_img_bytes(b):
    try:
        return Image.open(io.BytesIO(b)).convert("RGB")
    except:
        return None

# ================================================
# ROUTE KENDALI MOTOR
# ================================================
@app.route("/motor", methods=["POST"])
def motor_control():
    data = request.get_json()
    state = data.get("status", "")

    if state not in ["on", "off"]:
        return jsonify({"message": "invalid command"}), 400

    try:
        esp_url = f"{ESP32_DEV_URL}?state={state}"
        requests.get(esp_url, timeout=2)
        return jsonify({"message": f"motor {state}"}), 200
    except Exception as e:
        return jsonify({"message": "ESP32 unreachable", "error": str(e)}), 500

# ================================================
# ROUTE PREDICT
# ================================================
@app.route("/predict", methods=["POST"])
def predict():
    global last_output

    raw = request.files['file'].read() if 'file' in request.files else request.get_data()

    if not raw:
        return jsonify({"error": "no_image"}), 400

    img = read_img_bytes(raw)
    if img is None:
        return jsonify({"error": "decode_failed"}), 400

    os.makedirs(SAVE_FOLDER, exist_ok=True)
    filename = f"capture_{int(time.time())}.jpg"
    filepath = os.path.join(SAVE_FOLDER, filename)
    img.save(filepath)

    np_img = np.array(img)[:, :, ::-1]
    result = model(np_img, imgsz=224, conf=CONF_THRESHOLD)[0]

    cls_id = int(result.probs.top1)
    conf = float(result.probs.top1conf)
    label = CLASS_NAMES[cls_id]

    output = {
        "label": label,
        "confidence": conf,
        "image_path": filename,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    last_output = output

    try:
        with open(filepath, "rb") as f:
            requests.post(
                PHP_SAVE_URL,
                data=output,
                files={"image_file": f},
                timeout=2
            )
    except Exception as e:
        print("PHP unreachable:", e)

    return jsonify(output)

# ================================================
# LAST PREDICTION ROUTE
# ================================================
@app.route("/last", methods=["GET"])
def last():
    return jsonify(last_output)

# ================================================
# RUN SERVER
# ================================================
if __name__ == "__main__":
    os.makedirs(SAVE_FOLDER, exist_ok=True)
    port = int(os.environ.get("PORT", 5000))   # FIX penting untuk Render
    app.run(host="0.0.0.0", port=port, debug=False)
