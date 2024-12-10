import base64
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

import cv2
import numpy as np
import model

from waitress import serve
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
import requests
import atexit
import threading

app = Flask(__name__)
CORS(app)

first_start = True

scheduler = BackgroundScheduler()

def poke_server():
    """function to poke server"""
    try:
        # API_URL = 'https://digit-classifier-fw6f.onrender.com/poke'
        API_URL = 'http://localhost:8080/poke'
        response = requests.get(API_URL)
        print(response.status_code, response.json()['message'])
    
    except Exception as e:
        print(f'Error while poking server: {e}')
    return 
    

def start_scheduler():
    """Create a cron job that pokes the server every 14 minutes to avoid server spin down"""

    scheduler.add_job(
        func = poke_server,
        trigger=IntervalTrigger(minutes = 14),
        id = 'poke_server',
        replace_existing=True
    )
    scheduler.start()
    print("Started poking server in background thread...")

    return

@app.before_request
def initialize_scheduler_thread():
    """Initialize background thread to poke server upon first request"""
    global first_start
    if first_start:
        scheduler_thread = threading.Thread(target=start_scheduler, daemon=True)
        scheduler_thread.start()

        first_start = False
    
    return

@atexit.register
def shutdown_scheduler():
    """shut down cron job at exit"""
    print("Shutting down")
    if scheduler.running:
        scheduler.shutdown()

#API routes---------------------------------------------------------------
#page index (for local debuging)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/poke')
def poke():
    return jsonify({'message': 'poked'})

@app.route('/classify', methods=['POST'])
def upload_image():
    """
    API endpoint for getting model predictions from uploaded image
    """
    request.headers
    print("Classification request")

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