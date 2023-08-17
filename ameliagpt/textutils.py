from typing import Type

from ameliagpt.llm import AbstractEngineFactory, OpenAiEngine


def get_filename(input_string: str) -> str:
    return input_string.split('/')[-1]


def get_subclass_by_name(name: str | None, abstract_class=AbstractEngineFactory):
    """Allow to get the first subclass whose name is name"""
    if name is None:
        name = OpenAiEngine
    return [_ for _ in abstract_class.__subclasses__() if _.__name__ == name][0]
