from flask import Flask, render_template, request
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
#### from langchain.memory import ConversationBufferMemory

app = Flask(__name__)

#### Create embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

db = Chroma(persist_directory="./vectorstore",
            embedding_function=embeddings)

#### Create retriever
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs= {"k": 5}
)

#### Create Ollama language model
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

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    # clears memory at the start of the chat
    # memory.clear()
    
    msg = request.form["msg"]
    input = msg
    print(input)
    
    # fetch chat history from memory
    # chat_history = memory.load_memory_variables({})['chat_history']

    # Ensure the correct keys are passed to the QA chain
    # result=qa_chain({"question": input, "chat_history": chat_history})
    result = qa_chain.invoke(input)

    print("Response : ", result)
    return str(result)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)