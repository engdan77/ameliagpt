import os
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFDirectoryLoader

# Step 1
loader = PyPDFDirectoryLoader('/Users/edo/git/my/llm_docs/docs')
docs = loader.load()

# Step 2
embeddings = OpenAIEmbeddings()
# vectordb = Chroma.from_documents(docs, embedding=embeddings, persist_directory="./cache")
vectordb = Chroma.from_documents(docs, embedding=embeddings)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
pdf_qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.8), vectordb.as_retriever(), memory=memory)


# Step 3
# query = "What is Milky way?"
query = "What is the e-mail to Neato Robotics?"
result = pdf_qa({"question": query})
print("Answer:")
print(result["answer"])

