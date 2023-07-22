from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np

app = Flask(__name__, static_folder='static')

LINE_START = sv.Point(600, 0)
LINE_END = sv.Point(600, 700)

line_counter = sv.LineZone(start=LINE_START, end=LINE_END)
line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=2, text_scale=1)
box_annotator = sv.BoxAnnotator(
    thickness=2,
    text_thickness=1,
    text_scale=0.5
)

model = YOLO("yolov8n.pt")
def generate_frames():
    for result in model.track(source='live.mp4', show=False, stream=True, agnostic_nms=True):
        frame = result.orig_img
        detections = sv.Detections.from_yolov8(result)

        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

        detections = detections[(detections.class_id == 0)]
        print(detections)

        if len(detections) > 0:  # Check if there are any detections before processing
            labels = [
                f"{detections.tracker_id[0]} {model.model.names[detections.class_id[0]]} {detections.confidence[0]:0.2f}"
            ]

            frame = box_annotator.annotate(
                scene=frame,
                detections=detections,
                labels=labels
            )

            line_counter.trigger(detections=detections)

            line_annotator.annotate(frame=frame, line_counter=line_counter)

        in_count = line_counter.in_count
        out_count = line_counter.out_count

        # Convert the frame to JSON format
        frame_json = cv2.imencode('.jpg', frame)[1].tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_json + b'\r\n')
        
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/count', methods=['GET'])
def count():
    in_count = line_counter.in_count
    out_count = line_counter.out_count
    return jsonify(in_count=in_count, out_count=out_count)

if __name__ == "__main__":
    app.run(debug=True)
