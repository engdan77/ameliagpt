from . import settings
from pathlib import Path
from loguru import logger
import typer
from .api import start

app = typer.Typer()


@app.command("start")
def start_server(
    docs_path: Path = typer.Argument(
        ..., resolve_path=True, help="Path to docs to be used with LLM"
    ),
    port: int = typer.Option(8000, help="Server port"),
):
    logger.info(f"Starting {__name__}")
    logger.info(f"Set docs path to {docs_path.as_posix()}")
    start(docs_path, port=port)


if __name__ == "__main__":
    app()
