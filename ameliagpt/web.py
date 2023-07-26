from pywebio.output import put_table, put_loading, put_text, put_markdown
from pywebio.input import input
from .shared import shared_obj
from ameliagpt import __version__
from .textutils import get_filename


def conversation():
    put_markdown(f'# AmeliaGPT {__version__} Test interface')
    put_text(f'Currently loaded documents')
    for doc in shared_obj.loaded_docs:
        put_text(doc)
    while True:
        llm = shared_obj.llm
        question = input('Ask something')
        with put_loading():
            response = llm.ask(question)
        put_table([
            ['Q:', question],
            ['A:', [put_markdown(f'Sources being from **{get_filename(response["sources"])}**  \n\n\n{response["answer"]}')]
             ]])
