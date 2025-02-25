import cv2
import torch
import urllib.request
import matplotlib.pyplot as plt
import time
import numpy as np
from random import randint
#turtlebot
from transformers import pipeline
from PIL import Image
import json

pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

def forcast_x_min_depth(image, _time=''):
    # Convert numpy array to PIL
    image = Image.fromarray(image)

    start = time.time()

    prediction = pipe(image)["depth"]

    end = time.time()
    print("Time of inference:", end - start)

    # Convert back PIL image to numpy array
    prediction = torch.from_numpy(np.array(prediction))

    # Save predicted image
    cv2.imwrite(f"predictions/pred-{_time}.png", prediction.cpu().numpy())
    
    # Normalize image
    pmin = torch.min(prediction)
    pmax = torch.max(prediction)
    prediction = (prediction - pmin) / (pmax - pmin)

    # Average pooling to archive a 1x11 vector
    prediction = torch.nn.functional.adaptive_avg_pool2d(prediction.unsqueeze(0), (1,11)).squeeze()

    # We divide image into 11 parts and 
    # extract the index of the part with minimun depth
    # The maximun angular velocitiy of the robot is 5 rad/s
    # the robot will rotate as much as the index is far the center (index 5 in this case because our array has lenght 11)
    # The indexes look like this:
    # [-5 -4 -3 -2 -1 0 1 2 3 4 5]
    # The index is the estimate angular velocity of the robot for its next step.
    i = 5 - torch.argmin(prediction)

    # Save prediction in file
    with open(f'out_vector/vector_and_direction.txt', 'a')  as f:
        formatted_prediction = [f"{x:.2f}" for x in prediction.tolist()]  # Formatea cada número con 2 decimales
        f.write(f'({_time}): ' + json.dumps(i.tolist()) + " -> " + json.dumps(formatted_prediction, indent=2) + "\n")
        
    return i.item(), prediction


def estimate_robot_motion(image, _time=''):
    w, pred_vec = forcast_x_min_depth(image,  _time=_time)

    # Si hay numeros en el vector con 
    # un valor mayor a 0.95 puede 
    # significar un posible choque, 
    # por lo tanto retroceder.

    # Return velocity and normalized angular velocity
    w = w / 3.45
    print("w:", w)
    return 0.2, w / 3.45

# q