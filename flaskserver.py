from flask import Flask, request, jsonify
import pickle
import cv2
from prediction import estimate_robot_motion
from datetime import datetime
import os

app = Flask(__name__)
DEBUG = True
UPLOAD_FOLDER = 'images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        data = request.data
        image = pickle.loads(data)
        
        current_time = ''
        if DEBUG:
            current_time = datetime.now().strftime("%H:%M:%S")
            print("Tiempo:", current_time)
            img_path = os.path.join(UPLOAD_FOLDER, f"img-{current_time}.png")
            cv2.imwrite(img_path, image)
        
        v, w, d = estimate_robot_motion(image, _time=current_time, _debug=DEBUG)
        
        response = {
            'vel': v,
            'vel_ang': w,
            'duration': d
        }
        return pickle.dumps(response)
    
    except pickle.PickleError:
        return jsonify({'error': 'Error al cargar la imagen'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8765, debug=True)
