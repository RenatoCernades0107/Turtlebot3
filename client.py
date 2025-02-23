import asyncio
import websockets

# Cambia la IP por la de tu servidor
SERVER_IP = "localhost"  # Cambia por la IP del servidor
PORT = 8765

async def send_message():
    uri = f"ws://{SERVER_IP}:{PORT}"
    async with websockets.connect(uri) as websocket:
        message = "Hola desde el cliente!"
        print(f"📤 Enviando mensaje: {message}")
        await websocket.send(message)

        response = await websocket.recv()
        print(f"📩 Respuesta del servidor: {response}")

# Ejecutar el cliente WebSocket
asyncio.run(send_message())
