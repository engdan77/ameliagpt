import ngrok
from loguru import logger


async def create_tunnel(port=8000):
    session = await ngrok.NgrokSessionBuilder().authtoken_from_env().connect()
    tunnel = await session.http_endpoint().listen()
    logger.info(f"Ingress established at {tunnel.url()}")
    tunnel.forward_tcp(f"localhost:{port}")
    return tunnel
