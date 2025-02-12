import streamlit as st
st.set_page_config(page_title="MEDIBOT - AI Health Assistant", layout="centered")
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt  # ✅ Import the fixed prompt

# Load environment variables
load_dotenv()

# Load model
@st.cache_resource
def load_llm():
    return ChatGroq(
        temperature=0.4,
        max_tokens=500,
        model_name="llama-3.3-70b-versatile"
    )

# Load embeddings
@st.cache_resource
def load_embeddings():
    return download_hugging_face_embeddings()

# Load retriever
@st.cache_resource
def load_retriever():
    embeddings = load_embeddings()
    index_name = "medibot"
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name, 
        embedding=embeddings
    )
    return docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Initialize session state
if "initialized" not in st.session_state:
    st.session_state["retriever"] = load_retriever()
    st.session_state["llm"] = load_llm()
    st.session_state["messages"] = []  #  Store chat history
    st.session_state["initialized"] = True

# Create the document question-answering chain
question_answer_chain = create_stuff_documents_chain(
    llm=st.session_state["llm"], 
    prompt=system_prompt  # 🔹 Ensure system_prompt explicitly accepts `context`
)

# Create RAG chain
rag_chain = create_retrieval_chain(
    retriever=st.session_state["retriever"], 
    combine_docs_chain=question_answer_chain
)
# Styling
st.markdown(
    """
    <style>
    body { background-color: #F5F5F5; } /* Light Gray */
    .stChatMessage { border-radius: 10px; padding: 10px; margin: 5px 0; }
    .userMessage { background-color: #1B5E20; color: white; text-align: right; padding: 10px; border-radius: 10px; width: fit-content; max-width: 80%; }
    .botMessage { background-color: #E0E0E0; color: black; text-align: left; padding: 10px; border-radius: 10px; width: fit-content; max-width: 80%; }
    .header { font-size: 30px; font-weight: bold; color: #2E7D32; text-align: center; }
    .subheader { font-size: 18px; color: #444; text-align: center; }
    .chat-container { display: flex; flex-direction: column; gap: 10px; padding: 20px; }
    .send-container { display: flex; align-items: center; }
    .send-button { background-color: #388E3C; color: white; padding: 10px 15px; border-radius: 50%; border: none; cursor: pointer; margin-left: 10px; }
    </style>
    """,
    unsafe_allow_html=True
)

# MEDIBOT Intro
st.markdown("<div class='header'>🤖 Hi! I'm <b>MEDIBOT</b> </div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>Your AI-powered medical assistant. Ask me anything health-related! 🩺</div>", unsafe_allow_html=True)
st.write("___")

# Display chat history inside a styled container
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
for msg in st.session_state["messages"]:
    role, text = msg["role"], msg["content"]
    if role == "user":
        st.markdown(f"<div class='stChatMessage userMessage'>{text}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='stChatMessage botMessage'>{text}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Chat Input (Press Enter to Submit)
user_query = st.chat_input("💬 Tell me how I can help today!")

if user_query and user_query.strip():  # Prevent empty spaces from being submitted
    st.session_state["messages"].append({"role": "user", "content": user_query})

    with st.spinner("🤖 MEDIBOT is thinking..."):
        # 🔹 Fetch relevant medical context from documents
        context_docs = st.session_state["retriever"].invoke(user_query)
        context = " ".join([doc.page_content for doc in context_docs])

        # Make sure `context` is passed properly
        response = rag_chain.invoke({"context": context, "input": user_query})
        bot_response = response["answer"]

    # Store response in session state
    st.session_state["messages"].append({"role": "assistant", "content": bot_response})

    # Display user query (ensures it's visible)
    st.markdown(f"<div class='stChatMessage userMessage'>{user_query}</div>", unsafe_allow_html=True)

    # Display bot response
    st.markdown(f"<div class='stChatMessage botMessage'>{bot_response}</div>", unsafe_allow_html=True)
