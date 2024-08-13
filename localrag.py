from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

#### Create embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

db = Chroma(persist_directory="./vectorstore",
            embedding_function=embeddings)

#### Create retriever
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs= {"k": 5}
)

#### Create Ollama language model - Gemma 2
local_llm = 'llama3.1'

llm = ChatOllama(model=local_llm,
                 keep_alive="3h", 
                 max_tokens=512,  
                 temperature=0)

#### Convert loaded documents into strings by concatenating their content and ignoring metadata
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

RAG_TEMPLATE = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

<context>
{context}
</context>

Answer the following question:

{question}"""

rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

#### Create the RAG chain using LCEL with prompt printing and streaming output
qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

question = "What are the approaches to Task Decomposition?"

print(qa_chain.invoke(question))