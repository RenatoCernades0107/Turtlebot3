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
    start = time.time()
    image = Image.fromarray(image)
    prediction = pipe(image)["depth"]
    prediction = torch.from_numpy(np.array(prediction))
    end = time.time()
    print("Time of inference:", end - start)

    # # Send image to device    
    # input_batch = transform(image).to(device)

    # with torch.no_grad():
    #     start = time.time()
    #     prediction = midas(input_batch)

    #     prediction = torch.nn.functional.interpolate(
    #         prediction.unsqueeze(1),
    #         size=image.shape[:2],
    #         mode="bicubic",
    #         align_corners=False,
    #     ).squeeze()

    #     end = time.time()
    #     print("Time of inference:", end - start)

    cv2.imwrite(f"predictions/pred-{_time}.png", prediction.cpu().numpy())
    pmin = torch.min(prediction)
    pmax = torch.max(prediction)
    prediction = (prediction - pmin) / (pmax - pmin)
    #cv2.imwrite(f"out/out2222.png", 255*prediction.cpu().numpy())

    prediction = torch.nn.functional.adaptive_avg_pool2d(prediction.unsqueeze(0), (1,11)).squeeze()
    # cv2.imwrite(f"out/outchan.png", 255*prediction.cpu().numpy())
    print(prediction)
    
    i = 5 - torch.argmin(prediction)

    # Save prediction in file
    with open(f'out_vector/vector_and_direction.txt', 'a')  as f:
        f.write(json.dumps(i.tolist())  + json.dumps(prediction.tolist()) + '\n')
        
    return i.item()


# def forcast_x_min_depth(depth_map):
#     z = torch.mean(depth_map, dim=0)
#     z = z.unsqueeze(0)
#     nsz = 50
#     z = torch.nn.functional.adaptive_avg_pool1d(z, nsz).squeeze()
#     i = torch.argmin(z)
#     return i, z

def estimate_robot_motion(image, _time=''):
    w = forcast_x_min_depth(image,  _time=_time)
    return 0.2, w / 3.14
    # distancia focal f (mm)
    f = 45
    cx = image.shape[1] // 2
    v = np.array([f, cx-xmin])
    u = np.array([f, 0])
    
    dot_product = np.dot(u, v)
    unorm = np.linalg.norm(u)
    vnorm = np.linalg.norm(v)
    
    # Asegurar que el valor esté en el rango [-1, 1] para evitar errores numéricos
    cos_theta = dot_product / (unorm * vnorm)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # Obtener el ángulo en radianes y convertir a grados
    angle = np.arccos(cos_theta)
    print(f'Angulo: -60/{angle}/60')
    # Calculamos la velocidad angular
    # angle/max_angle * max_vel
    w = 5*angle/120
    
    return 0.2, w
    

