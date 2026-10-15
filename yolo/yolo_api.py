from flask import Flask, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO("yolov8n.pt")

@app.route("/detect", methods=["POST"])
def detect():
    file = request.files["image"]
    file.save("temp.jpg")
    results = model("temp.jpg")
    objects = []
    for result in results:
        for box in result.boxes:
            obj_name = result.names[int(box.cls)]
            conf = float(box.conf)
            objects.append(f"{obj_name}（置信度{conf:.0%}）")
    return jsonify({"detected_objects": objects})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)
