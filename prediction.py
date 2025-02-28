import cv2
import torch
import time
import numpy as np
from random import randint
import os

from transformers import pipeline
from PIL import Image
import json

pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

os.makedirs('out_vector', exist_ok=True)
os.makedirs('predictions', exist_ok=True)

def dividir_imagen_con_sombra(img, barra_index):
    # Convertir imagen a numpy si no lo es
    if not isinstance(img, np.ndarray):
        img = np.array(img, dtype=np.uint8)

    # Convertir imagen a RGB si es en blanco y negro
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    alto, ancho, _ = img.shape
    region_ancho = ancho // 11  # Dividir en 11 regiones iguales

    for i in range(11):
        x_inicio = i * region_ancho
        x_fin = (i + 1) * region_ancho if i != 10 else ancho  # La última región puede no ser exacta
        if i == 5 - barra_index:
            # Crear sombra verde con transparencia
            sombra = np.zeros_like(img[:, x_inicio:x_fin], dtype=np.uint8)
            sombra[:, :, 1] = 150  # Canal verde (G) con intensidad mediad
            sombra = cv2.addWeighted(img[:, x_inicio:x_fin], 0.7, sombra, 0.3, 0)
            img[:, x_inicio:x_fin] = sombra
            
        img = cv2.putText(img, f'{5 - i}', (x_inicio+20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        # Dibujar líneas divisorias para visualizar las regiones
        cv2.line(img, (x_inicio, 0), (x_inicio, alto), (255, 255, 255), 1)
    
    return img


def forcast_x_min_depth(image, _time='', _debug=False):
    # Convert numpy array to PIL
    image = Image.fromarray(image)

    if _debug:
        start = time.time()

    prediction = pipe(image)["depth"]
    
    if _debug:
        end = time.time()
        print("Time of inference:", end - start)

    # Convert back PIL image to tensor
    complete_prediction = torch.from_numpy(np.array(prediction))

    # Normalize image
    pmin = torch.min(complete_prediction)
    pmax = torch.max(complete_prediction)
    prediction = (complete_prediction - pmin) / (pmax - pmin)

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

    if _debug:
        # Save prediction in file
        with open(f'out_vector/vector_and_direction.txt', 'a')  as f:
            formatted_prediction = [f"{x:.2f}" for x in prediction.tolist()]  # Formatea cada número con 2 decimales
            f.write(f'({_time}): ' + json.dumps(i.tolist()) + " -> " + json.dumps(formatted_prediction, indent=2) + "\n")
        
        # Save predicted image divided in 11 horizontal regions.
        complete_prediction = dividir_imagen_con_sombra(complete_prediction.cpu(), i)
        cv2.imwrite(f"predictions/pred-{_time}.png", complete_prediction)

    return i.item(), prediction


def estimate_robot_motion(image, _time='', _debug=False):
    w, prediction = forcast_x_min_depth(image,  _time=_time, _debug=_debug)

    # Si hay numeros en el vector con 
    # un valor mayor a 0.95 puede 
    # significar un posible choque, 
    # por lo tanto retroceder.
    v = 0.2
    for i in range(3, 10):
        if prediction[i] > 0.85:
            v = -0.2
    
    # Return velocity and normalized angular velocity
    if _debug:
        print("Velocidad Angular:", w)
        print("Velocidad:", v)
        # print('Duracion (t):', t.item())

    # assert(t > 0, 'Duration must be positive')
    return v, w, 1

# q