import argparse
import base64

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import load_model

import cv2


sio = socketio.Server()
app = Flask(__name__)
model = None
steering_history = np.zeros(2)

# lenet (32, 32)
def preprocess(img, resize_shape=(32, 32)):
    # Crop the road area (no sky)
    l_inf = int(0.3125 * img.shape[0])
    l_sup = int(0.75 * img.shape[0])
    img = img[l_inf:l_sup,:]
    # Resize the image
    img = cv2.resize(img, resize_shape)
    # Transform to YUV to maitain croma
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img

def smooth_steering_angle(steering_angle):
    '''
    Apply smoothin to steering_angle and reject disturbance
    '''
    global steering_history
    w = [9.0, 1.0]

    steering_history[0], steering_history[1] = steering_angle, steering_history[0]

    # Weight the steering angle with the previous to smooth it
    weighted = np.sum(steering_history * w) / np.sum(w)

    # Discard little perturbations (less than 2 grads)
    if abs(weighted) < 0.08:
        weighted = 0.0 

    return weighted

def adapt_throttle(steering_angle):
    '''
    Adapt throttle to slow down on curves 
    '''
    throttle = 0.2
    if abs(steering_angle) > 0.12:
        throttle = 0.1

    return throttle

@sio.on('telemetry')
def telemetry(sid, data):
    start_time = time.time()

    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)
    image_array = preprocess(image_array)
    #transformed_image_array = image_array[None, :, :, :]
    transformed_image_array = np.reshape(image_array, (1, 32, 32, 1))
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    steering_angle = float(model.predict(transformed_image_array, batch_size=1))
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    # throttle = 0.2

    steering_angle = smooth_steering_angle(steering_angle)
    throttle = adapt_throttle(steering_angle)

    if steering_angle > 0.1:
        side = '/'
    elif steering_angle < -0.1:
        side = '\\'
    else:
        side = '|'

    end_time = time.time()
    elapsed = end_time - start_time

    print('Elapsed: {:.3f} - Turn: {} - Steering angle: {:.3f} - Throttle: {}'.format(elapsed, side, abs(steering_angle), throttle))
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model.')
    args = parser.parse_args()
    model = load_model(args.model)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
