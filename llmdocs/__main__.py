import ngrok
from api import start
from loguru import logger
from tunnel import create_tunnel
import asyncio


def main(port=8000):
    # tunnel = ngrok.connect(8000, authtoken_from_env=True)
    # asyncio.create_task(create_tunnel(port=port))
    start(port=port)


if __name__ == '__main__':
    main()

