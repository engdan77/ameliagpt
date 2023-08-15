import pickle
import subprocess
import sys
from collections import Counter
from pathlib import Path

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain import VectorDBQAWithSourcesChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
import faiss
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFaceHub
from loguru import logger
from .shared import shared_obj
from langchain import OpenAI
import time
from pyrate_limiter import Limiter, Duration, RequestRate


def limited(until):
    duration = int(round(until - time.time()))
    logger.info('Rate limited, sleeping for {:d} seconds'.format(duration))


class MyLLM:
    def __init__(self, docs_source_path: Path | None = None, name: str = 'llm'):
        load_dotenv()
        self.vector_store = None
        self.docs_source_path = docs_source_path
        self.docs_sources = []
        self.source_processors = {'pdf': self.get_pdf_content,
                                  'doc': self.get_doc_content,
                                  'txt': self.get_txt_content}
        self.fn_index = f'{name}.index'
        self.fn_vector_store = f'{name}.pkl'
        self.qa_chain = None

    def run(self):
        store = self.get_vector_store(self.docs_source_path)
        data, sources = self.get_docs_by_path(self.docs_source_path)
        store = self.append_data_to_vector_store(vector_store=store, data=data, metadatas=sources)
        llm = self.get_openai_llm()
        self.qa_chain = self.get_faiss_qa_chain(store, llm)

    def count_tokens(self) -> Counter:
        data, sources = self.get_texts_including_sources_by_path(self.docs_source_path)
        token_count = Counter()
        for i, doc in enumerate(data):
            token_count[sources[i].stem] = len(doc.split())
        return token_count

    def get_docs_by_path(self, docs_source_path: Path) -> tuple[list, list]:
        data, sources = self.get_texts_including_sources_by_path(docs_source_path, records_processed_files=shared_obj.loaded_docs)
        data, sources = self.get_chunks_including_metadata(data, sources)
        data, sources = self.remove_over_sized_chunks_inline(data, sources)
        return data, sources

    def get_vector_store(self) -> FAISS | object:
        # data, sources = self.get_docs_by_path(docs_source_path)
        if Path(self.fn_index).exists() and Path(self.fn_vector_store).exists():
            logger.info(f'Loading {self.fn_vector_store} and {self.fn_index} as DB')
        else:
            logger.info(f'Creating new vector DB')
            store = self.create_faiss_vectorstore([], [])
            # store = self.append_data_to_vector_store(vector_store=store, data=data, metadatas=sources)
            # store = self.create_faiss_vectorstore(docs, metadatas)
            self.store_faiss_vectorstore(store)
        store = self.load_vectorstore()
        return store

    def append_data_to_vector_store(self, vector_store: FAISS, data: list, metadatas: list, rate_limit_tpm=300_000) -> FAISS | object:
        store = vector_store
        tokens_per_chunk = len(data[0].split())
        calls_per_minute = rate_limit_tpm / tokens_per_chunk
        tot_chunks = len(data)
        rate = RequestRate(calls_per_minute, Duration.MINUTE)
        limiter = Limiter(rate)
        for i, (doc, metadata) in enumerate(zip(data, metadatas)):
            with limiter.ratelimit('add_to_vector', delay=True):
                logger.info(f'Adding vector for chunk {i}/{tot_chunks} {i/tot_chunks:.1f}%')
                store.add_texts([doc], metadatas=[metadata])
        self.vector_store = store
        return store


    @staticmethod
    def get_pdf_content(input_file: Path) -> str:
        text = ''
        pdf_reader = PdfReader(input_file.as_posix())
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

    @staticmethod
    def get_doc_content(input_file: Path) -> str:
        fn = input_file.as_posix()
        if sys.platform != 'darwin':
            logger.warning(f'Converting DOC {input_file.name} currently only supported on macOS')
        cmd = 'textutil -stdout -strip -cat txt'.split()
        cmd.append(fn)
        result = subprocess.run(cmd, capture_output=True)
        soup = BeautifulSoup(result.stdout, 'html.parser')
        return soup.get_text()

    @staticmethod
    def get_txt_content(input_file: Path) -> str:
        return input_file.read_text()

    def get_texts_including_sources_by_path(self, source_path: Path, records_processed_files: list[str] = []) -> tuple[list, list]:
        """Build a list of all text, while also tracking the source of each text chunk."""
        data = []
        sources = []
        for ext, func in self.source_processors.items():
            for p in source_path.rglob(f'*.{ext}'):
                logger.info(f'Processing {p.as_posix()}')
                data.append(func(p))
                sources.append(p)
                records_processed_files.append(p.name)
        self.docs_source_path = sources
        return data, sources

    @staticmethod
    def get_chunks_including_metadata(data: list, sources: list, chunk_size=1000) -> tuple[list, list]:
        """Split the text into character chunks and generate metadata"""
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, separator="\n")
        docs = []
        metadatas = []
        for i, d in enumerate(data):
            splits = text_splitter.split_text(d)
            docs.extend(splits)
            metadatas.extend([{"source": sources[i]}] * len(splits))
        return docs, metadatas

    @staticmethod
    def remove_over_sized_chunks_inline(docs: list, metadatas: list, chunk_size=1000) -> tuple[list, list]:
        """Remove oversized chunks, could be noise inline"""
        bad_docs = [i for i, d in enumerate(docs) if len(d) > chunk_size + 100]
        for i in sorted(bad_docs, reverse=True):
            logger.debug('Deleting doc due to size', f'size:{len(docs[i])} doc: {docs[i]}')
            del docs[i]
            del metadatas[i]
        return docs, metadatas

    def create_faiss_vectorstore(self, docs: list, metadatas: list) -> FAISS:
        """Create a vector store"""
        store = FAISS.from_texts(docs, OpenAIEmbeddings(), metadatas=metadatas)
        self.vector_store = store
        return store

    def store_faiss_vectorstore(self, store: FAISS):
        faiss.write_index(store.index, self.fn_index)
        store.index = None
        with open(self.fn_vector_store, "wb") as f:
            pickle.dump(store, f)

    def load_vectorstore(self) -> FAISS:
        """Load the FAISS index from disk."""
        index = faiss.read_index(self.fn_index)
        with open(self.fn_vector_store, "rb") as f:
            store: object = pickle.load(f)
        store.index = index
        return store

    @staticmethod
    def get_openai_llm(model_name="gpt-3.5-turbo", temperature=0.7, max_tokens=1000) -> OpenAI:
        return OpenAI(temperature=temperature, max_tokens=max_tokens, model_name=model_name)

    @staticmethod
    def get_faiss_qa_chain(vector_store: FAISS | None, llm: OpenAI):
        if not vector_store:
            raise RuntimeError('vector_store needs to be supplied as argument')
        """Build the question answering chain."""
        # return VectorDBQAWithSourcesChain.from_llm(llm=llm, vectorstore=vectorstore)
        return VectorDBQAWithSourcesChain.from_chain_type(
            llm=llm,
            chain_type="stuff",
            vectorstore=vector_store
        )

    def ask(self, question: str):
        logger.info(f'Asking question: {question}')
        # response = self.conversation({'question': question})
        # answer = response['answer']
        response = self.qa_chain({'question': question})
        logger.info(f'Got answer: {response}')
        return response
