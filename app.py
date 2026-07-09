from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import traceback
from ultralytics import YOLO
import time

app = Flask(__name__)


try:
    model = YOLO("helmet.pt")
     
except Exception as e:
    print(e)
    model = None


DETECTION_CONF = 0.25   # increased from 0.30 to reduce false positives
DETECTION_IOU  = 0.45


HELMET_KEYWORDS = ['helmet', 'hardhat', 'with_helmet', 'wearing_helmet']


NO_HELMET_KEYWORDS = ['no_helmet', 'no-helmet', 'without_helmet',
                       'without-helmet', 'head', 'no helmet', 'nohelmet']



PERSON_KEYWORDS = ['person']


def classify_detections(results):
   
    wearing = 0
    not_wearing = 0
    person_count = 0
    total = 0

    for r in results:
        for box in r.boxes:
            total += 1
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            class_lower = class_name.lower().strip()


            if class_lower in NO_HELMET_KEYWORDS:
             not_wearing += 1
             print(f"NO_HELMET -> {class_name}")
            elif any(k in class_lower for k in HELMET_KEYWORDS):
              wearing += 1
              print(f"HELMET -> {class_name}")
            elif any(k in class_lower for k in PERSON_KEYWORDS):
                person_count += 1
                print(f"  PERSON   -> {class_name} ({confidence:.2f})")
            else:

                print(f"  OTHER    -> {class_name} ({confidence:.2f}) - ignored")

    return wearing, not_wearing, person_count, total


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


        results = model(img, imgsz=640, conf=DETECTION_CONF, iou=DETECTION_IOU)


        result_img = results[0].plot(line_width=2, font_size=0.8, conf=True)

        wearing, not_wearing, person_count, total = classify_detections(results)







        if wearing > 0:
            final_helmet = wearing
            final_no_helmet = 0
        elif not_wearing > 0:
            final_helmet = 0
            final_no_helmet = not_wearing
        elif person_count > 0:
            final_helmet = 0
            final_no_helmet = person_count
        else:
            final_helmet = 0
            final_no_helmet = 0

        processing_time = round((time.time() - start) * 1000, 2)


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

        if model is None:
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
        'model': 'loaded' if model else 'failed',
        'model_classes': list(model.names.values()) if model else []
    })


if __name__ == '__main__':
    print("Starting Helmet Detection System on http://localhost:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)