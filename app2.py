from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import os
from flask_cors import CORS
import joblib

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.tools.retriever import create_retriever_tool
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore

def PrintHello(n):
    print(f"Hello world {n}")
PrintHello(1)
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
PrintHello(2)
def load_or_generate_embeddings():
    PrintHello("Inside docs")
    pdfs=["IPCC_AR6.pdf","tectonic.pdf"]
    documents=[]
    for i in pdfs:
        loader=PyPDFLoader(i)
        documents.extend(loader.load())


    text_splitter= RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=50)
    splitted_docs=text_splitter.split_documents(documents)

    # Setting up Chroma db to store embeddings
    db=FAISS.from_documents(splitted_docs,OpenAIEmbeddings(api_key=api_key))
    get_docs=db.as_retriever()
    return get_docs
PrintHello(3)
def initialize_chatbot_tools(api_key):
    PrintHello("inside agents")
    # Load or generate embeddings
    get_docs = load_or_generate_embeddings()

    # Initialize tools for chatbot
    api_wrapperwiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    wiki = WikipediaQueryRun(api_wrapper=api_wrapperwiki)
    webloader = WebBaseLoader("https://www.ipcc.ch/")
    web_docs = webloader.load()
    web_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    splitted_webdocs = web_splitter.split_documents(web_docs)
    web_db = FAISS.from_documents(splitted_webdocs, OpenAIEmbeddings(api_key=api_key))
    get_web = web_db.as_retriever()
    docs_retriever = create_retriever_tool(
        get_docs,
        "IPCC_and_tectonic_aspects_of_climate_change",
        "Search anything from IPCC sixth assessment long report and from the review paper tectonic aspects of climate change."
    )
    web_retriever = create_retriever_tool(get_web, "IPCC_website", "Search any information from the IPCC website.")

    tools = [wiki, web_retriever, docs_retriever]
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    prompt = hub.pull("hwchase17/openai-functions-agent")
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executer = AgentExecutor(agent=agent, tools=tools, verbose=True)

    return agent_executer
PrintHello(4)
agent_executer = initialize_chatbot_tools(api_key)
# Initialize Flask app
app = Flask(__name__)
CORS(app)
# In-memory chat history storage (for demonstration purposes)
# Route to handle user prompts
@app.route('/answer', methods=['POST'])
def answer():
    PrintHello(5)
    data = request.get_json()
    user_prompt = data.get("user_prompt")
    chat_history = data.get("chat_history", [])
    PrintHello(user_prompt)
    # Retrieve chat history for this user
    if chat_history:
        chat_history = chat_history
    else:
        chat_history = []
    PrintHello(6)
    # Invoke agent to process the user prompt
    response = agent_executer.invoke({"input": user_prompt})
    PrintHello(7)
    chat_history.append({"role": "user", "content": user_prompt})
    chat_history.append({"role": "assistant", "content": response})
    PrintHello(response)
    return jsonify({"response": response["output"], "chat_history": chat_history})

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000)
