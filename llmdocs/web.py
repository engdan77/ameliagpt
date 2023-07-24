from pywebio.output import put_table
from pywebio.input import input


def conversation():
    while True:
        question = input('Ask something')
        put_table([
            ['Q:', question],
            ['A:', my_llm(question)]
        ])