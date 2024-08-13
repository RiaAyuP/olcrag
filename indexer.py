from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

#### Load documents from a directory
loader = DirectoryLoader("./docs", glob="*.md")
documents = loader.load()
print(f"{len(documents)} MD file(s) loaded.")

#### Create text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300,
    add_start_index=True,
)

#### Split documents into chunks
texts = text_splitter.split_documents(documents)

#### Create embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)

#### Create vector store
vectorstore = Chroma.from_documents(
    documents=texts, 
    embedding= embeddings,
    persist_directory="./vectorstore")
print("Vectorstore created.")