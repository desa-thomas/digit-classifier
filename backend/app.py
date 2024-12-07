import base64
from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import io

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def upload_image():

    data = request.get_json()
    img_data = data.get('image')
    
    #if image present
    if img_data: 
        #remove 'data:image/png;base64' prefix
        img_data = img_data.split(',')[1]
        img_bytes = base64.b64decode(img_data)
        with open("debug_image.png", "wb") as f:
            f.write(img_bytes)

        img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        image = cv2.imdecode(img_arr, cv2.IMREAD_GRAYSCALE)
        
        #Debug, check that the image uploaded
        # if image is not None:
        #     cv2.imshow('Image', image)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

        return jsonify({'message': "Image uploaded successfully"}), 200
    else:
        return jsonify({'error': "No image data found"}), 400

if __name__ =='__main__':
    app.run(debug=True)

