import os
import sys

#import openai
#from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain

import constants

os.environ["OPENAI_API_KEY"] = constants.APIKEY

PERSIST = False
query = None

if len(sys.argv) > 1:
  query = sys.argv[1]

embedding = OpenAIEmbeddings()

if PERSIST and os.path.exists("persist"):
  print("Reusing index...\n")
  vectorstore = Chroma(persist_directory="persist", embedding_function=embedding)
else:
  loader = DirectoryLoader("data/")
  documents = loader.load()
  vectorstore = Chroma.from_documents(documents, embedding, persist_directory="persist" if PERSIST else None)

retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

chain = ConversationalRetrievalChain.from_llm(
  llm=ChatOpenAI(model="gpt-3.5-turbo"),
  retriever=retriever,
)

chat_history = []
while True:
  if not query:
    query = input("Prompt: ")
  if query in ['quit', 'q', 'exit']:
    sys.exit()
  result = chain({"question": query, "chat_history": chat_history})
  print(result['answer'])

  chat_history.append((query, result['answer']))
  query = None
