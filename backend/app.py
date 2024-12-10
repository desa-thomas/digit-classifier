import base64
from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import model
from waitress import serve

app = Flask(__name__)

#page index
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def upload_image():
    """
    API endpoint for getting model predictions from uploaded image
    """
    data = request.get_json()
    img_data = data.get('image')

    #if image present
    if img_data:
        #remove 'data:image/png;base64' prefix
        img_data = img_data.split(',')[1]
        img_bytes = base64.b64decode(img_data)
       
        img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        image = cv2.imdecode(img_arr, cv2.IMREAD_GRAYSCALE)

        #get prediction from model, then server prediction
        prediction = model.predict(image)
        
        return jsonify({'message': "Image uploaded successfully", "prediction": prediction}), 200
    else:
        return jsonify({'error': "No image data found"}), 400

if __name__ =='__main__':
    serve(app, host = "0.0.0.0", port=8080)

    #Development server
    #app.run(debug=True)