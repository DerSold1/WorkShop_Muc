# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 14:09:12 2025

@author: msoldner
"""
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
import os
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import embeddings
from langchain_openai import OpenAIEmbeddings

from  langchain_core.embeddings import Embeddings
import langchain_openai.embeddings.base
#from langchain_openai.embeddings.base.OpenAIEmbeddings import embed_query
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain import hub
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langgraph.graph import START, StateGraph
from IPython.display import Image, display
import getpass

#OPENAI_API_KEY = ""
OPENAI_API_KEY = ""

SPLIT_CHUNK_SIZE = 100
SPLIT_CHUNK_OVERLAP = 2

os.environ["LANGCHAIN_PROJECT"] = "ProofOfConcept"

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

os.environ["NVIDIA_API_KEY"] = OPENAI_API_KEY



embeddings = NVIDIAEmbeddings(model="NV-Embed-QA")
vector_store = InMemoryVectorStore(embeddings)


llm = ChatNVIDIA(model="meta/llama3-70b-instruct")



'''Load the PDF Document. Adjust the link to your file'''
loader = PyPDFLoader("Taycan-Porsche-Connect-Gut-zu-wissen-Die-Anleitung.pdf")
pages = []
for page in loader.lazy_load():
    pages.append(page)
text = ""

#Extract document to string
for document in loader.lazy_load():
    #print(document)
    text += document.page_content
docs = pages    
print(f"Total characters: {len(pages[0].page_content)}")




#Textsplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=SPLIT_CHUNK_SIZE,  # chunk size (characters)
    chunk_overlap=SPLIT_CHUNK_OVERLAP,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)
print(f"Split blog post into {len(all_splits)} sub-documents.")


#Store documents
document_ids = vector_store.add_documents(documents=all_splits)
print(document_ids[:3])

#docs = retriever.invoke(query)

prompt = hub.pull("rlm/rag-prompt")



# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()



while True:
    print("Chatbot: Wie kann ich helfen?")
    question = Input("Bitte Frage eingeben: ")
    
    response = graph.invoke({"question": question})
    answer = response["answer"]
    print(answer)








