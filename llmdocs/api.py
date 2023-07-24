import asyncio
from loguru import logger
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from tunnel import create_tunnel

from llm import MyLLM

app = FastAPI()
my_llm = MyLLM()


class Question(BaseModel):
    question: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/ask")
def read_item(q: Question):
    logger.info(f'{q=}')
    response = my_llm.ask(q.question)
    logger.info(f'{response=}')
    return {"question": q.question, "response": response}


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(create_tunnel())


def start(port=8000):
    loop = asyncio.get_event_loop()

    config = uvicorn.Config(app, host="0.0.0.0", port=port)
    server = uvicorn.Server(config)

    loop.run_until_complete(server.serve())