from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, WebBaseLoader, UnstructuredMarkdownLoader
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

import os
os.environ['USER_AGENT'] = 'myagent'

#### Load documents from a directory (there is problem here, will look again.)
#loader = DirectoryLoader("./docs", glob="**/*.txt")
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
#loader = UnstructuredMarkdownLoader("./docs")
documents = loader.load()
print(f"{len(documents)} file(s) loaded.")

#### Create text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    add_start_index=True,
)

#### Split documents into chunks
texts = text_splitter.split_documents(documents)
print("Text splitting success.")

#### Create embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

#### Create vector store
vectorstore = Chroma.from_documents(
    documents=texts, 
    embedding= embeddings,
    persist_directory="./vectorstore")
print("Vectorstore created.")