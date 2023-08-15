from ameliagpt.llm import MyLLM
from . import settings
from pathlib import Path
from loguru import logger
import typer
from .api import start
import rich.table
import rich.console
from humanize import number

app = typer.Typer()

logger.add("ameliagpt.log", rotation="100 MB")

@app.command("start")
def start_server(
    docs_path: Path = typer.Argument(
        ..., resolve_path=True, help="Path to docs to be used with LLM"
    ),
        name: str = typer.Option('llm', help='Name of database'),
        port: int = typer.Option(8000, help="Server port"),
):
    logger.info(f"Starting {__name__}")
    logger.info(f"Set docs path to {docs_path.as_posix()}")
    start(docs_path, name=name, port=port)


@app.command()
def count_tokens(docs_path: Path = typer.Argument(..., resolve_path=True, help="Path to docs to be used with LLM")):
    my_llm = MyLLM(docs_path)
    token_count = my_llm.count_tokens()
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
        name: str = typer.Option('llm', help='Name of database')):
    my_llm = MyLLM(name=name)
    data, sources = my_llm.get_docs_by_path(docs_path)
    # store = my_llm.get_vector_store()
    logger.info('Adding word embeddings')
    store = my_llm.append_data_to_vector_store(data=data, metadatas=sources)
    my_llm.store_faiss_vectorstore(store)


if __name__ == "__main__":
    app()
