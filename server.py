import asyncio
import websockets
import pickle
import cv2
from random import randint
from prediction import estimate_robot_motion
from datetime import datetime

# q
DEBUG = True

async def handler(websocket):
    try:
        async for msg in websocket:
            try:
                image = pickle.loads(msg)
            except Exception as e:
                raise pickle.PickleError("Error al cargar la imagen")

            current_time = '' 
            if DEBUG:
                current_time = datetime.now().strftime("%H:%M:%S")
                print("Tiempo:", current_time)
                cv2.imwrite(f"images/img-{current_time}.png", image)
            v, w, d = estimate_robot_motion(image, _time=current_time, _debug=DEBUG)

            res = {
                'vel': v,
                'vel_ang': w,
                'duration': d
            }
            res = pickle.dumps(res)
            await websocket.send(res)  # Respuesta al cliente
            
    except websockets.exceptions.ConnectionClosed:
        print("🚫 Conexión cerrada")


# Función para iniciar el servidor correctamente en Windows
async def start_server():
    server = await websockets.serve(handler, "0.0.0.0", 8765, max_size=2765018)
    print("🚀 Servidor WebSocket corriendo en ws://localhost:8765")
    await server.wait_closed()

# Ejecutar con asyncio.run() para evitar errores en Windows
asyncio.run(start_server())
# q