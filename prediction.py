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


'''
OLD MODEL
# if model == 'large':
#     model_type = "DPT_Large"   # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
# elif model == 'hybrid':
#     model_type = "DPT_Hybrid"  # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
# elif model == 'small':
#     model_type = "MiDaS_small" # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
# else:
#     raise ValueError('Select (large/hybrid/small) model to predict the depth map.')
# model_type = 'DPT_Large'
# midas = torch.hub.load("intel-isl/MiDaS", model_type)

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# print(f'Using {device} for prediction...')

# midas.to(device)
# midas.eval()

# midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

# if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
#     transform = midas_transforms.dpt_transform
# else:
#     transform = midas_transforms.small_transform
'''

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
        f.write(json.dumps(i.tolist())  + json.dumps(prediction.tolist()) + '\n')
        
    return i.item()


def estimate_robot_motion(image, _time=''):
    w = forcast_x_min_depth(image,  _time=_time)

    # Return velocity and normalized angular velocity
    return 0.2, w

# q