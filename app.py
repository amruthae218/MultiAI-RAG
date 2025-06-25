import os
import streamlit as st
from typing import List, Literal
from typing_extensions import TypedDict
import re
import time

import cassio
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Cassandra
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langgraph.graph import END, StateGraph, START
from langchain.schema import Document   
from pydantic import BaseModel, Field

# --- Disable tokenizer warning ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("USER_AGENT", "StreamlitRAGApp/1.0")

# --- Streamlit Configuration ---
st.set_page_config(
    page_title="RAG Chat Assistant",
    page_icon="🤖",
    layout="centered"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hi! I'm your RAG Assistant. Ask a question, and I’ll search your sources and Wikipedia to find the best possible answer."
    }]

# --- UI Components ---
st.title("🤖 RAG Chat Assistant")
st.caption("Powered by LangChain, Groq, and Astra DB")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")

    user_urls = st.text_input(
        "Custom Knowledge Sources (comma-separated URLs):",
        placeholder="https://example.com, https://another.com",
        help="Leave this empty to use the default knowledge base."
    )
    st.caption("**Note:** Press ⏎ Enter after typing URLs to apply them.")

    st.caption("Default sources include Lilian Weng's blog posts on agents, prompt engineering, and adversarial attacks.")

    st.divider()

    st.subheader("How It Works")
    st.markdown("""
    - Ask a question below in the chat interface.
    - The assistant first searches your configured knowledge sources.
    - If no relevant answer is found, it falls back to Wikipedia.
    - A final answer is generated using the most relevant context.
    """)

    # st.divider()

    # st.subheader("Recommended Topics")
    # st.markdown("""
    # This assistant works best with questions related to:
    # - Prompt engineering
    # - Adversarial attacks on LLMs
    # - General AI/ML topics
    # """)

    st.caption("Example: Explain Adversarial attacks on LLMs?")

# --- RAG Pipeline Setup ---
@st.cache_resource(show_spinner=False)
def setup_pipeline(custom_urls=None):
    load_dotenv()
    
    # Initialize AstraDB
    # ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    # ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
    ASTRA_DB_APPLICATION_TOKEN = st.secrets["ASTRA_DB_APPLICATION_TOKEN"]
    ASTRA_DB_ID = st.secrets["ASTRA_DB_ID"]
    GROQ_KEY = st.secrets["GROQ_KEY"]
    HF_TOKEN = st.secrets["HF_TOKEN"]

    cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

    # Document loading
    urls = custom_urls or [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    # Text splitting
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=700, 
        chunk_overlap=30
    )
    splits = splitter.split_documents(docs_list)

    # Embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    astra_vector_store = Cassandra(
        embedding=embeddings, 
        table_name="qa_mini_demo", 
        session=None, 
        keyspace=None
    )
    astra_vector_store.add_documents(splits)
    retriever = astra_vector_store.as_retriever()

    # LLMs initialization
    #GROQ_KEY = os.getenv("GROQ_KEY")
    llm_router = ChatGroq(groq_api_key=GROQ_KEY, model_name="llama3-8b-8192")
    llm = ChatGroq(groq_api_key=GROQ_KEY, model_name="llama3-8b-8192")

    # Routing configuration
    class RouteQuery(BaseModel):
        datasource: Literal["vectorstore", "wiki_search"] = Field(...)

    system = (
        "You are an expert at routing user questions to appropriate sources.\n"
        "Use the vectorstore for: LLM agents, prompt engineering, adversarial attacks.\n"
        "For all other topics, use Wikipedia search."
    )
    route_prompt = ChatPromptTemplate.from_messages([
        ("system", system), 
        ("human", "{question}")
    ])
    structured_router = llm_router.with_structured_output(RouteQuery)
    question_router = route_prompt | structured_router

    # Wikipedia tool
    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
    wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

    # LangGraph workflow
    class GraphState(TypedDict):
        question: str
        generation: str
        documents: List[Document]

    def retrieve(state):
        """Retrieve documents from vector store"""
        question = state["question"]
        documents = retriever.invoke(question)
        return {"documents": documents, "question": question}

    def wiki_search(state):
        """Search Wikipedia for answers"""
        question = state["question"]
        docs = wiki.invoke({"query": question})
        return {"documents": [Document(page_content=docs)], "question": question}

    def route_question(state):
        """Determine data source for query"""
        question = state["question"]
        source = question_router.invoke({"question": question})
        return source.datasource

    # Build workflow
    workflow = StateGraph(GraphState)
    workflow.add_node("wiki_search", wiki_search)
    workflow.add_node("retrieve", retrieve)
    workflow.add_conditional_edges(
        START, 
        route_question, 
        {"wiki_search": "wiki_search", "vectorstore": "retrieve"}
    )
    workflow.add_edge("wiki_search", END)
    workflow.add_edge("retrieve", END)
    
    return workflow.compile(), llm

# --- Answer Processing Functions ---
def extract_answer(docs: List[Document]) -> str:
    """Process and clean document content for answer generation"""
    seen = set()
    chunks = []
    for doc in docs[:3]:  # Consider top 3 documents
        lines = doc.page_content.split("\n")
        for line in lines:
            clean = line.strip()
            if (not clean or 
                clean.lower() in seen or
                any(clean.lower().startswith(prefix) 
                    for prefix in ("table of contents", "posts", "archive", "faq", "tags", "|"))):
                continue
            seen.add(clean.lower())
            chunks.append(clean)
    
    # Merge and truncate at last complete sentence
    merged = "\n\n".join(chunks).strip()
    last_period = merged.rfind(".")
    return merged[:last_period + 1] if last_period != -1 else merged

def get_answer(app, question, llm):
    """Execute RAG workflow and generate final answer with simple source attribution."""
    inputs = {"question": question}
    final_value = None

    # Execute workflow
    for output in app.stream(inputs):
        for key, value in output.items():
            final_value = value

    # Process results
    if final_value and "documents" in final_value:
        context = extract_answer(final_value["documents"])
        prompt = f"""
        You are a helpful assistant. Answer the following question using ONLY the provided context.
        Be concise, accurate, and include key details. If the answer is not directly in the context, try to infer related insights based on it. Do not make up facts.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
        result = llm.invoke(prompt)
        # Determine source information
        source_type = "Wikipedia" if "wikipedia" in context.lower() else "Knowledge Base"
        return result.content.strip(), source_type

    return "I couldn't find an answer to that question.", "None"


# --- Main Chat Interface ---
def main():
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "source" in message:
                st.caption(f"Source: {message['source']}")
            if "source_info" in message:
                st.caption(f"Document: {message['source_info']}")

    # Process custom URLs
    url_list = [url.strip() for url in user_urls.split(",") if url.strip()] if user_urls.strip() else None
    
    # Initialize RAG pipeline
    app, llm = setup_pipeline(custom_urls=url_list)
    
    # Chat input
    if prompt := st.chat_input("Ask anything ..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                answer, source = get_answer(app, prompt, llm)
            
            # Display response
            message_placeholder.markdown(answer)
            
            # Display source info
            # if source != "None":
            #     st.caption(f"Source: {source}")
        
        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
            # ,
            # "source": source
        })

if __name__ == "__main__":
    main()
