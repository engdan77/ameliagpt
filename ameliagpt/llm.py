from pathlib import Path
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from loguru import logger
from .shared import shared_obj


class MyLLM:
    def __init__(self, docs_sources: Path):
        load_dotenv()
        raw_text = self.get_pdf_text(docs_sources)
        text_chunks = self.get_text_chunks(raw_text)
        vectorstore = self.get_vectorstore(text_chunks)
        self.conversation = self.get_conversation_chain(vectorstore)

    @staticmethod
    def get_pdf_text(src_path: Path, extensions=('pdf',)):
        text = ""
        if not src_path.is_dir():
            raise RuntimeError(f'Docs dir {src_path.as_posix()} needs to be a directory')
        for ext in extensions:
            for file in src_path.rglob(f'*.{ext}'):
                logger.info(f'Processing {file.as_posix()}')
                pdf_reader = PdfReader(file.as_posix())
                for page in pdf_reader.pages:
                    text += page.extract_text()
                shared_obj.loaded_docs.append(file.name)
        return text

    @staticmethod
    def get_text_chunks(text):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        return chunks

    @staticmethod
    def get_vectorstore(text_chunks):
        embeddings = OpenAIEmbeddings()
        # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vectorstore

    @staticmethod
    def get_conversation_chain(vectorstore):
        llm = ChatOpenAI()
        # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

        memory = ConversationBufferMemory(
            memory_key='chat_history', return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
        return conversation_chain

    def ask(self, question: str):
        logger.info(f'Asking question: {question}')
        response = self.conversation({'question': question})
        answer = response['answer']
        logger.info(f'Got answer: {answer}')
        return answer