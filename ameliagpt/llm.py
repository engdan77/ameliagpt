import abc
import subprocess
import sys
import tempfile
from abc import ABC
from collections import Counter
from enum import StrEnum, auto
from pathlib import Path
from typing import Type
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain import VectorDBQAWithSourcesChain
from langchain.document_loaders import TextLoader
from langchain.embeddings import LlamaCppEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from loguru import logger
from .shared import shared_obj
import time
from pyrate_limiter import Limiter, Duration, RequestRate


class ChainType(StrEnum):
    stuff = auto()   # Stuff everything (may cause size over LLM context)
    map_reduce = auto()   # Iterate over docs, and finally ask question on combined
    refine = auto()   # Iterate get answer on doc1, but ask additionally doc1 and doc2
    map_rerank = auto()  # As map_reduce but supply scores


class AbstractEngineFactory(ABC):
    """For any new engine profile create subclass for your project that will adapt to the service"""
    def __int__(self):
        self.model_object = None
        self.embedding_object = None
        self.splitter_chunk_size = 1000
        self.chain_type = ChainType.stuff
        logger.info(f"Initializing LLM engine {self.__name__}")

    @abc.abstractmethod
    def initialize(self, **kwargs):
        """Import required modules and set properties"""
        self.name = NotImplementedError
        self.model_object = NotImplementedError
        self.embedding_object = NotImplementedError
        self.chain_type = ChainType.stuff


class Gpt4AllEngine(AbstractEngineFactory):
    """This was sucessfully run either using the mini model or the larger one, and using GPT4All default embedding"""
    def initialize(self, max_tokens=1000):
        from langchain.embeddings import GPT4AllEmbeddings
        from langchain.llms import GPT4All
        self.name = "gpt4all"
        if (Path.home() / '.cache/gpt4all/gpt4all-converted.bin').exists():
            logger.info('Using the larger model for LLM')
            model_name = 'gpt4all-converted.bin'
        else:
            logger.info('Using the default model')
            model_name = 'orca-mini-3b.ggmlv3.q4_0.bin'
        self.model_object = GPT4All(
            model=model_name, max_tokens=max_tokens,
        )
        self.embedding_object = GPT4AllEmbeddings()
        self.splitter_chunk_size = 900
        self.chain_type = ChainType.map_reduce  # Avoid get over context size of LLM


class Gpt4AllAndLLamaEmbeddingEngine(AbstractEngineFactory):
    """Still experimenting and appear to fail loading the Llama model - not sure due to lack of memory"""
    def initialize(self, max_tokens=1000):
        from langchain.llms import GPT4All
        base_path = Path.home() / '.cache/gpt4all'
        model_name = base_path / 'gpt4all-converted.bin'
        embedding_model_name = base_path / 'ggml-model-q4_0.bin'
        assert model_name.exists() and embedding_model_name.exists(), "Missing model files for this profile, please check README"
        self.model_object = GPT4All(model=model_name.as_posix(), max_tokens=max_tokens)
        self.embedding_object = LlamaCppEmbeddings(model_path=embedding_model_name.as_posix())
        self.splitter_chunk_size = 1000
        self.chain_type = ChainType.map_reduce  # Avoid get over context size of LLM


class OpenAiEngine(AbstractEngineFactory):
    def initialize(
        self,
        temperature=0.7,
        max_tokens=1000,
        model_name="gpt-3.5-turbo",
        chunk_size=20,
    ):
        # chunk_size helps reduce number of tokens passed to OpenAI, helps rate-limits
        from langchain import OpenAI
        from langchain.embeddings import OpenAIEmbeddings

        self.name = "openai"
        self.model_object = OpenAI(
            temperature=temperature, max_tokens=max_tokens, model_name=model_name
        )
        self.embedding_object = OpenAIEmbeddings()
        self.splitter_chunk_size = 1000
        self.chain_type = ChainType.stuff


def limited(until):
    duration = int(round(until - time.time()))
    logger.info("Rate limited, sleeping for {:d} seconds".format(duration))


class MyLLM:
    def __init__(
        self, name: str = "llm", engine: Type[AbstractEngineFactory] = OpenAiEngine
    ):
        load_dotenv()
        self.name = name
        self.engine = engine()
        self.vector_store = None
        self.docs_source_path = None
        self.docs_sources = []
        self.source_processors = {
            "pdf": self.get_pdf_content,
            "doc": self.get_doc_content,
            "txt": self.get_txt_content,
        }
        self.fn_index = f"{name}.index"
        self.fn_vector_store = f"{name}.pkl"
        self.qa_chain = None
        logger.warning(
            f"Remember to call {self.__class__.__name__}.init_engine() before processing"
        )

    def init_engine(self):
        self.engine.initialize()

    def run(self):
        self.init_engine()  # Create all engine attributes before using those
        self.vector_store = self.get_vector_store()
        self.qa_chain = self.get_faiss_qa_chain()

    def get_current_docs_loaded(self) -> list:
        if not self.vector_store:
            return []
        return list(
            set(
                [
                    _.metadata["source"].name
                    for _ in self.vector_store.docstore._dict.values()
                ]
            )
        )

    def count_tokens(self, docs_path: Path) -> Counter:
        data, sources = self.get_texts_including_sources_by_path(docs_path)
        token_count = Counter()
        for i, doc in enumerate(data):
            token_count[sources[i].stem] = len(doc.split())
        return token_count

    def get_docs_by_path(self, docs_source_path: Path) -> tuple[list, list]:
        chunk_size = self.engine.splitter_chunk_size
        data, sources = self.get_texts_including_sources_by_path(
            docs_source_path, records_processed_files=shared_obj.loaded_docs
        )
        data, sources = self.get_chunks_including_metadata(
            data, sources, chunk_size=chunk_size
        )
        data, sources = self.remove_over_sized_chunks_inline(data, sources, chunk_size)
        return data, sources

    def get_vector_store(self) -> FAISS | object:
        # if Path(self.fn_index).exists() and Path(self.fn_vector_store).exists():
        if Path(f"{self.name}_index").exists():
            logger.info(f"Loading {self.fn_vector_store} and {self.fn_index} as DB")
            store = self.load_vectorstore()
        else:
            logger.info(f"Creating new vector DB")
            store = self.create_faiss_vectorstore([], [])  # TODO: should be improved

        return store

    def append_data_to_vector_store(
        self, data: list, metadatas: list, vector_store: FAISS | None = None
    ) -> FAISS | object:
        """This apparently not working so might be removed"""
        self.engine.initialize()
        if vector_store:
            store = vector_store
            store.from_texts(
                data, self.engine.embedding_object, metadatas=metadatas
            )  # TODO: this does not seem to work
        else:
            store = FAISS.from_texts(
                data, self.engine.embedding_object, metadatas=metadatas
            )
        self.vector_store = store
        return store

    def append_data_to_vector_store_throttled(
        self, vector_store: FAISS, data: list, metadatas: list, rate_limit_tpm=300_000
    ) -> FAISS | object:
        """This apparently not working so might be removed"""
        store = vector_store
        tokens_per_chunk = len(data[0].split())
        calls_per_minute = rate_limit_tpm / tokens_per_chunk
        tot_chunks = len(data)
        rate = RequestRate(calls_per_minute, Duration.MINUTE)
        limiter = Limiter(rate)
        for i, (doc, metadata) in enumerate(zip(data, metadatas)):
            with limiter.ratelimit("add_to_vector", delay=True):
                logger.info(
                    f"Adding vector for chunk {i}/{tot_chunks} {i/tot_chunks:.1f}%"
                )
                store.from_texts(
                    [doc], self.engine.embedding_object, metadatas=[metadata]
                )
        self.vector_store = store
        return store

    @staticmethod
    def get_pdf_content(input_file: Path) -> str:
        text = ""
        pdf_reader = PdfReader(input_file.as_posix())
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

    @staticmethod
    def get_doc_content(input_file: Path) -> str:
        fn = input_file.as_posix()
        if sys.platform != "darwin":
            logger.warning(
                f"Converting DOC {input_file.name} currently only supported on macOS"
            )
        cmd = "textutil -stdout -strip -cat txt".split()
        cmd.append(fn)
        result = subprocess.run(cmd, capture_output=True)
        soup = BeautifulSoup(result.stdout, "html.parser")
        return soup.get_text()

    @staticmethod
    def get_txt_content(input_file: Path) -> str:
        return input_file.read_text()

    def get_texts_including_sources_by_path(
        self, source_path: Path, records_processed_files: list[str] = []
    ) -> tuple[list, list]:
        """Build a list of all text, while also tracking the source of each text chunk."""
        data = []
        sources = []
        for ext, func in self.source_processors.items():
            for p in source_path.rglob(f"*.{ext}"):
                logger.info(f"Processing {p.as_posix()}")
                data.append(func(p))
                sources.append(p)
                records_processed_files.append(p.name)
        self.docs_source_path = sources
        return data, sources

    @staticmethod
    def get_chunks_including_metadata(
        data: list, sources: list, chunk_size=1000
    ) -> tuple[list, list]:
        """Split the text into character chunks and generate metadata"""
        logger.info(f"Splitting data based on size of {chunk_size}")
        text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=50, separator="\n"
        )
        docs = []
        metadatas = []
        for i, d in enumerate(data):
            splits = text_splitter.split_text(d)
            docs.extend(splits)
            metadatas.extend([{"source": sources[i]}] * len(splits))
        return docs, metadatas

    @staticmethod
    def remove_over_sized_chunks_inline(
        docs: list, metadatas: list, chunk_size=1000
    ) -> tuple[list, list]:
        """Remove oversized chunks, could be noise inline"""
        bad_docs = [i for i, d in enumerate(docs) if len(d) > chunk_size + 100]
        for i in sorted(bad_docs, reverse=True):
            logger.debug(
                "Discarding chunk due to size:{len(docs[i])} doc-index: {docs[i]}"
            )
            logger.debug(f"Data such as {docs[i][:40]}...{docs[i][-40:]}")
            del docs[i]
            del metadatas[i]
        return docs, metadatas

    def create_faiss_vectorstore(self, docs: list, metadatas: list) -> FAISS:
        """Create a vector store"""
        if docs:
            store = FAISS.from_texts(
                docs, self.engine.embedding_object, metadatas=metadatas
            )
        else:
            f = tempfile.NamedTemporaryFile("w", prefix="tmp_", delete=False)
            f.write("foo")
            f.close()
            loader = TextLoader(f.name)
            documents = loader.load()
            text_splitter = CharacterTextSplitter(
                chunk_size=1000, chunk_overlap=0
            )  # TODO: How the heck to create an empty FAISS..?
            docs = text_splitter.split_documents(documents)
            store = FAISS.from_documents(docs, self.engine.embedding_object)
        self.vector_store = store
        return store

    def save_faiss_vectorstore(self, store: FAISS):
        name = f"{self.name}_index"
        logger.info(f"Saving index to {name}")
        store.save_local(name)

    def load_vectorstore(self) -> FAISS | object:
        """Load the FAISS index from disk."""
        name = f"{self.name}_index"
        logger.info(f"Loading index from {name}")
        store = FAISS.load_local(name, self.engine.embedding_object)
        return store

    def get_faiss_qa_chain(self):
        if not self.vector_store:
            raise RuntimeError("vector_store needs to be supplied as argument")
        """Build the question answering chain."""
        return VectorDBQAWithSourcesChain.from_chain_type(
            llm=self.engine.model_object,
            chain_type=self.engine.chain_type,
            vectorstore=self.vector_store,
        )

    def ask(self, question: str):
        logger.info(f"Asking question: {question}")
        response = self.qa_chain({"question": question})
        logger.info(f"Got answer: {response}")
        return response
