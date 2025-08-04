from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # ✅ Enables CORS for all origins

@app.route('/process', methods=['POST'])
def process_image():
    try:
        file = request.files['image']
        num_layers = int(request.form.get('layers', 3))

        # Convert to grayscale
        img = Image.open(file.stream).convert('L')
        img = img.resize((600, 600))
        img_np = np.array(img)

        thresholds = np.linspace(0, 255, num_layers + 1)[1:]
        layers = []

        for thresh in thresholds:
            _, bin_img = cv2.threshold(img_np, thresh, 255, cv2.THRESH_BINARY)
            pil_img = Image.fromarray(bin_img)
            buffered = BytesIO()
            pil_img.save(buffered, format="PNG")
            layer_base64 = base64.b64encode(buffered.getvalue()).decode()
            layers.append(layer_base64)

        return jsonify({"layers": layers})

    except Exception as e:
        print("Error in /process:", e)
        return jsonify({"error": str(e), "layers": []}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
