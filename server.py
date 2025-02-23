import asyncio
import websockets

# Función para manejar la conexión WebSocket
async def handler(websocket):
    try:
        async for message in websocket:
            print(f"📩 Mensaje recibido: {message}")
            response = f"✅ Recibí tu mensaje: {message}"
            await websocket.send(response)  # Responder al cliente
    except websockets.exceptions.ConnectionClosed:
        print("🚫 Conexión cerrada")

# Función para iniciar el servidor correctamente en Windows
async def start_server():
    server = await websockets.serve(handler, "0.0.0.0", 8765)
    print("🚀 Servidor WebSocket corriendo en ws://localhost:8765")
    await server.wait_closed()

# Ejecutar con asyncio.run() para evitar errores en Windows
asyncio.run(start_server())
