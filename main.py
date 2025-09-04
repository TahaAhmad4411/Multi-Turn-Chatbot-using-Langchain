# Loaders
from langchain_community.document_loaders import TextLoader

# Embeddings
from langchain_openai import OpenAIEmbeddings

# Vector stores
from langchain_community.vectorstores import FAISS

# Memory & Chains (still in core langchain)
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# LLMs (OpenAI Chat model)
from langchain_openai import ChatOpenAI



files = ["about.txt", "features.txt", "faq.txt"]
docs = []
for f in files:
    loader = TextLoader(f)
    docs.extend(loader.load())   # extend instead of append

embeddings = OpenAIEmbeddings(api_key="Your generated API Key")

vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local("visa_chatbot_index")
new_vectorstore = FAISS.load_local(
    "visa_chatbot_index",
    embeddings,
    allow_dangerous_deserialization=True
)
retriever = new_vectorstore.as_retriever()
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

llm = ChatOpenAI(model_name="gpt-3.5-turbo",
                 api_key="Your generated API Key"
)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm, retriever, memory=memory
)
import streamlit as st

st.title("VisaBridge Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask something:")

if user_input:
    response = qa_chain({"question": user_input})
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", response["answer"]))

for speaker, msg in st.session_state.chat_history:
    st.write(f"**{speaker}:** {msg}")
import json

while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break
    
    result = qa_chain({"question": query})
    print("Bot:", result["answer"])
    
    # Log inside the loop
    with open("chat_log.json", "a") as f:
        json.dump({"query": query, "response": result["answer"]}, f)
        f.write("\n")

