from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
#### from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader, TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

import os
os.environ['USER_AGENT'] = 'myagent'

#### Load documents from a directory (there is problem here, will look again.)
#loader = DirectoryLoader("./docs", glob="**/*.txt", loader_cls=TextLoader)
#loader = WebBaseLoader("https://openstax.org/books/organic-chemistry/pages/23-1-carbonyl-condensations-the-aldol-reaction")
#loader = UnstructuredMarkdownLoader("./docs")
loader = WebBaseLoader("https://pandas.pydata.org/docs/getting_started/intro_tutorials/01_table_oriented.html#min-tut-01-tableoriented")
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