from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
import base64
import traceback
import time
import os
import onnxruntime as ort

app = Flask(__name__)

MODEL_PATH = "helmet.onnx"
DETECTION_CONF = 0.5
DETECTION_IOU = 0.45
INPUT_SIZE = 640

CLASS_NAMES = {0: "Helmet", 1: "No Helmet"}

try:
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
except Exception as e:
    print(e)
    session = None
    input_name = None


def preprocess(img):
    h, w = img.shape[:2]
    scale = min(INPUT_SIZE / h, INPUT_SIZE / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh))

    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    canvas[0:nh, 0:nw] = resized

    blob = canvas[:, :, ::-1].astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis, ...]
    return blob, scale, (h, w)


def nms(boxes, scores, iou_threshold):
    idxs = cv2.dnn.NMSBoxes(
        boxes.tolist(), scores.tolist(), DETECTION_CONF, iou_threshold
    )
    if len(idxs) == 0:
        return []
    return idxs.flatten().tolist()


def postprocess(output, scale, orig_shape):
    predictions = np.squeeze(output[0]).T
    scores = np.max(predictions[:, 4:], axis=1)
    keep = scores > DETECTION_CONF
    predictions = predictions[keep]
    scores = scores[keep]
    class_ids = np.argmax(predictions[:, 4:], axis=1)

    if len(predictions) == 0:
        return []

    boxes = predictions[:, :4].copy()
    boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2

    boxes /= scale

    keep_idxs = nms(boxes, scores, DETECTION_IOU)

    detections = []
    h, w = orig_shape
    for i in keep_idxs:
        x, y, bw, bh = boxes[i]
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(w, int(x + bw))
        y2 = min(h, int(y + bh))
        detections.append({
            "class_id": int(class_ids[i]),
            "class_name": CLASS_NAMES.get(int(class_ids[i]), str(class_ids[i])),
            "confidence": float(scores[i]),
            "box": (x1, y1, x2, y2)
        })
    return detections


def draw_detections(img, detections):
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = det["class_name"]
        conf = det["confidence"]
        color = (0, 200, 0) if label.lower() == "helmet" else (0, 0, 230)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.2f}"
        cv2.putText(img, text, (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img


def classify_detections(detections):
    wearing = 0
    not_wearing = 0
    for det in detections:
        name = det["class_name"].lower().strip()
        if name == "helmet":
            wearing += 1
        elif name == "no helmet":
            not_wearing += 1
    return wearing, not_wearing, len(detections)


def detect_objects(image_data):
    try:
        start = time.perf_counter()

        if ',' in image_data:
            image_data = image_data.split(',')[1]
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")

        blob, scale, orig_shape = preprocess(img)
        output = session.run(None, {input_name: blob})
        detections = postprocess(output, scale, orig_shape)

        result_img = draw_detections(img.copy(), detections)
        wearing, not_wearing, total = classify_detections(detections)

        if wearing > 0:
            final_helmet, final_no_helmet = wearing, 0
        elif not_wearing > 0:
            final_helmet, final_no_helmet = 0, not_wearing
        else:
            final_helmet, final_no_helmet = 0, 0

        processing_time = round((time.perf_counter() - start) * 1000, 2)

        _, buffer = cv2.imencode('.jpg', result_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        result_base64 = base64.b64encode(buffer).decode('utf-8')

        return (result_base64, processing_time,
                final_helmet, final_no_helmet, total)

    except Exception:
        print(traceback.format_exc())
        raise


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    try:
        if not request.is_json:
            return jsonify({'success': False, 'error': 'Request must be JSON'}), 400

        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image data provided'}), 400

        if session is None:
            return jsonify({'success': False, 'error': 'Model not loaded'}), 500

        result_image, proc_time, wearing, not_wearing, total = detect_objects(data['image'])

        return jsonify({
            'success': True,
            'result_image': f"data:image/jpeg;base64,{result_image}",
            'processing_time': proc_time,
            'wearing_helmet': wearing,
            'not_wearing_helmet': not_wearing,
            'total_detections': total
        })

    except Exception as e:
        return jsonify({'success': False, 'error': f"Server error: {str(e)}"}), 500


@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model': 'loaded' if session else 'failed',
        'model_classes': list(CLASS_NAMES.values())
    })


@app.route('/test_images/<path:filename>')
def serve_test_images(filename):
    return send_from_directory('test_images', filename)


if __name__ == '__main__':
    print("Starting Helmet Detection System on http://localhost:5000")
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )