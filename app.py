# from flask import Flask, request, jsonify, render_template
# import numpy as np
# from PIL import Image
# import io, base64

# app = Flask(__name__)

# # Load your model (Keras example)
# from tensorflow.keras.models import load_model
# model = load_model('mnist_cnn.h5')

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.json['image']  # base64 image from canvas

#     # Decode base64 → PIL image → numpy array
#     img_data = base64.b64decode(data.split(',')[1])
#     img = Image.open(io.BytesIO(img_data)).convert('L')  # grayscale
#     img = img.resize((28, 28))

#     arr = np.array(img) / 255.0
#     arr = arr.reshape(1, 28, 28, 1)  # batch + channel dims

#     prediction = model.predict(arr)
#     digit = int(np.argmax(prediction))
#     confidence = float(np.max(prediction))

#     return jsonify({'digit': digit, 'confidence': round(confidence * 100, 2)})

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io, base64
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model('mnist_cnn.h5', compile=False)

HTML = """
<!DOCTYPE html>
<html>
<head><title>MNIST Digit Recognizer</title></head>
<body style="background:#1a1a2e; color:white; text-align:center; font-family:Arial;">
  <h1>Draw a Digit</h1>
  <canvas id="canvas" width="280" height="280"
    style="border:3px solid white; cursor:crosshair;"></canvas>
  <br><br>
  <button onclick="predict()" style="padding:10px 30px; font-size:16px; margin:5px;">Guess!</button>
  <button onclick="clearCanvas()" style="padding:10px 30px; font-size:16px; margin:5px;">Clear</button>
  <h2 id="result"></h2>

  <script>
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, 280, 280);
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 18;
    ctx.lineCap = 'round';

    let drawing = false;
    canvas.onmousedown = e => { drawing = true; ctx.beginPath(); move(e); };
    canvas.onmouseup   = () => drawing = false;
    canvas.onmouseleave = () => drawing = false;
    canvas.onmousemove = e => { if (!drawing) return; move(e); };

    function move(e) {
      const r = canvas.getBoundingClientRect();
      ctx.lineTo(e.clientX - r.left, e.clientY - r.top);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(e.clientX - r.left, e.clientY - r.top);
    }

    async function predict() {
      const imageData = canvas.toDataURL('image/png');
      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: imageData })
      });
      const data = await res.json();
      document.getElementById('result').innerText =
        'Prediction: ' + data.digit + '  (' + data.confidence + '% confident)';
    }

    function clearCanvas() {
      ctx.fillStyle = 'black';
      ctx.fillRect(0, 0, 280, 280);
      document.getElementById('result').innerText = '';
    }
  </script>
</body>
</html>
"""

@app.route('/')
def index():
    return HTML

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['image']
    img_data = base64.b64decode(data.split(',')[1])
    img = Image.open(io.BytesIO(img_data)).convert('L')
    img = img.resize((28, 28))
    arr = np.array(img) / 255.0
    arr = arr.reshape(1, 28, 28, 1)
    prediction = model.predict(arr)
    digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    return jsonify({'digit': digit, 'confidence': round(confidence * 100, 2)})

if __name__ == '__main__':
    app.run(debug=True)