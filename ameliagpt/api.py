import asyncio
from pathlib import Path

from loguru import logger
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

from .textutils import get_filename
from .tunnel import create_tunnel
from .llm import MyLLM
from pywebio.platform.fastapi import asgi_app
from .web import conversation
from .shared import shared_obj
from ameliagpt import __version__

app = FastAPI(docs_url='/api',
              title="AmeliaGPT",
              version=__version__,
              description='A service supply answers chained with documents',
              contact={
                  "name": "Daniel Engvall",
                  "email": "daniel.engvall@ipsoft.com",
              },
              )


class Question(BaseModel):
    question: str


@app.get("/")
def read_root():
    return "Nothing here"


@app.post("/ask")
def read_item(q: Question):
    logger.info(f'Ask API: {q}')
    llm = shared_obj.llm
    response = llm.ask(q.question)
    logger.info(f'Response API: {response}')
    return {"question": q.question, "answer": response['answer'], "sources": get_filename(response['sources'])}


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(create_tunnel())


def start(src_docs: Path, port=8000):
    loop = asyncio.get_event_loop()
    my_llm = MyLLM(src_docs)
    shared_obj.llm = my_llm
    app.mount("/conversation", asgi_app(conversation))
    config = uvicorn.Config(app, host="0.0.0.0", port=port)
    server = uvicorn.Server(config)
    loop.run_until_complete(server.serve())