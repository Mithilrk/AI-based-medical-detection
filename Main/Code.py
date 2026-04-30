from flask import Flask, render_template_string, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import pytesseract

# ---- TESSERACT PATH (IMPORTANT) ----
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = Flask(__name__)

model = YOLO("model/best.pt")  

# -------- HTML inside Python --------
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Medicine & Object Detector</title>
</head>
<body style="text-align:center; background:black; color:white;">

<h2>AI Medicine and Object Detection</h2>

<video id="video" width="400" autoplay></video><br><br>
<button onclick="capture()"> Capture & Detect</button>

<p id="result">Waiting...</p>

<style>
button {
    padding: 20px 50px;
    font-size: 22px;
    margin-top: 25px;
    cursor: pointer;
    border-radius: 12px;
    border: none;
    background: #00adb5;
    color: white;
    font-weight: bold;
    display: block;
    margin-left: auto;
    margin-right: auto;
    transition: 0.3s;
}
button:hover {
    background: #007b80;
    transform: scale(1.05);
}
</style>

<script>
const video = document.getElementById('video');

navigator.mediaDevices.getUserMedia({ video: true })
.then(stream => video.srcObject = stream);

function speak(text) {
    const speech = new SpeechSynthesisUtterance(text);
    speechSynthesis.speak(speech);
}

function capture() {
    const canvas = document.createElement('canvas');
    canvas.width = 400;
    canvas.height = 300;
    const ctx = canvas.getContext('2d');

    ctx.drawImage(video, 0, 0, 400, 300);

    canvas.toBlob(function(blob) {
        let formData = new FormData();
        formData.append('image', blob);

        document.getElementById('result').innerText = "Detecting...";

        fetch('/detect', {
            method: 'POST',
            body: formData
        })
        .then(res => res.json())
        .then(data => {
            let output = "";

            if (data.objects.length > 0) {
                output = "Detected: " + data.objects.join(", ");

                if (data.text) {
                    output += ". Text: " + data.text;
                }

                speak(output);
            } else {
                output = "No objects detected";
                speak(output);
            }

            document.getElementById('result').innerText = output;
        });
    }, 'image/jpeg');
}
</script>

</body>
</html>
"""

# -------- ROUTES --------
@app.route('/')
def home():
    return render_template_string(HTML_PAGE)

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    results = model(frame)
    detected = []

    for r in results:
        for cls_id, conf in zip(r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            if conf > 0.4:
                detected.append(r.names[int(cls_id)])

    # Remove duplicates
    detected = list(set(detected))

    # -------- OCR --------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)

    # Clean OCR text
    text = text.strip().replace("\n", " ")
    if len(text) > 100:
        text = text[:100]  

    return jsonify({
        "objects": detected,
        "text": text
    })

# -------- RUN --------
if __name__ == "__main__":
    app.run(debug=True)
