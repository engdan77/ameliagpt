from ameliagpt.llm import MyLLM
from ameliagpt.textutils import get_subclass_by_name
from pathlib import Path
from loguru import logger
import typer
from .api import start
import rich.table
import rich.console
from humanize import number
from .llm import AbstractEngineFactory
from enum import StrEnum
from shared import shared_obj
from ameliagpt import __version__

app = typer.Typer()
engine_names = [_.__name__ for _ in AbstractEngineFactory.__subclasses__()]
Engine = StrEnum('Engine', {_:_ for _ in engine_names})

logger.add("ameliagpt.log", rotation="100 MB")


@app.command("start")
def start_server(
        name: str = typer.Argument('llm', help='Name of database'),
        port: int = typer.Option(8000, help="Server port"),
        engine: Engine = typer.Argument(..., help='Which LLM engine to use'),
        tunnel_enabled: bool = typer.Option(False, help='NGROK tunnel enabled')
):
    logger.info(f"Starting {__name__} {__version__}")
    shared_obj.tunnel = tunnel_enabled
    start(name=name, port=port, engine=get_subclass_by_name(engine.value, AbstractEngineFactory))


@app.command()
def count_tokens(docs_path: Path = typer.Argument(..., resolve_path=True, help="Path to docs to be used with LLM")):
    my_llm = MyLLM()
    token_count = my_llm.count_tokens(docs_path)
    t = rich.table.Table()
    t.add_column('Document')
    t.add_column('Count')
    t.add_column('%')
    top_ones = dict(token_count.most_common(20))
    the_rest = sum(list(dict(token_count).values())[20:])
    tot = token_count.total()
    for document_name, count in top_ones.items():
        t.add_row(document_name, str(count), f'{count / tot * 100:.2f}%')
    t.add_row('Rest', str(the_rest), f'{the_rest / tot * 100:.2f}%')
    rich.console.Console().print(t)
    logger.info(f'Total: {tot} ({number.intword(tot)})')
    logger.info(f'Cost using text-embedding-ada-002: ${tot * 0.0001:.2f}')


@app.command()
def add_documents(
    docs_path: Path = typer.Argument(..., resolve_path=True, help="Path to docs to be used with LLM"),
        name: str = typer.Option('llm', help='Name of database'),
        engine: Engine = typer.Argument(..., help='Which LLM engine to use')):
    my_llm = MyLLM(name=name, engine=get_subclass_by_name(engine.value, AbstractEngineFactory))
    my_llm.init_engine()
    data, sources = my_llm.get_docs_by_path(docs_path)
    logger.info('Adding word embeddings')
    store = my_llm.append_data_to_vector_store(data=data, metadatas=sources)
    my_llm.save_faiss_vectorstore(store)


if __name__ == "__main__":
    app()
