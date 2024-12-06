from flask import Flask, request, jsonify
import cv2

app = Flask(__name__)

@app.route('/')
def home():
    return '<p>root/</p>' 

@app.route('/upload', methods=['POST'])
def upload_image():
    #Check if image file was uploaded
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    #get image from request
    image_file = request.files['image']

    try: 
        image = cv2.imread(image_file)

    except Exception as e: 
        return jsonify({'error': str(e)}), 500
    

if __name__ =='__main__':
    app.run(debug=True)

