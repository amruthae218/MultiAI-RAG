import os
import streamlit as st
from typing import List, Any
from typing_extensions import TypedDict

import cassio
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Cassandra
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langgraph.graph import END, StateGraph, START
from langchain.schema import Document

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
    st.caption("Example: Explain Adversarial attacks on LLMs?")

from langchain_community.document_loaders import BSHTMLLoader
import requests
from bs4 import BeautifulSoup

# --- RAG Pipeline Setup ---
@st.cache_resource(show_spinner=False)
def setup_pipeline(custom_urls=None):
    load_dotenv()

    # ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    # ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
    # GROQ_KEY = os.getenv("GROQ_KEY")
    # HF_TOKEN = os.getenv("HF_TOKEN")
    ASTRA_DB_APPLICATION_TOKEN = st.secrets["ASTRA_DB_APPLICATION_TOKEN"]
    ASTRA_DB_ID = st.secrets["ASTRA_DB_ID"]
    GROQ_KEY = st.secrets["GROQ_KEY"]
    HF_TOKEN = st.secrets["HF_TOKEN"]

    if not (ASTRA_DB_APPLICATION_TOKEN and ASTRA_DB_ID and GROQ_KEY):
        st.error("One or more environment variables (ASTRA_DB_APPLICATION_TOKEN, ASTRA_DB_ID, GROQ_KEY) are missing!")
        st.stop()

    cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

    urls = custom_urls or [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]
    docs = []
    for url in urls:
        st.write(f"Loading {url} ...")
        try:
            loaded = WebBaseLoader(url).load()
            for doc in loaded:
                doc.metadata["source"] = url
            docs.extend(loaded)
        except Exception as e:
            st.warning(f"Failed to load {url}: {e}")

    st.write(f"Loaded {len(docs)} documents from URLs")
    if len(docs) > 0:
        st.write(docs[:1])  # Show a sample document for debugging

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=700, chunk_overlap=30)
    splits = splitter.split_documents(docs)
    st.write(f"Split into {len(splits)} document chunks")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # IMPORTANT FIX: Provide session and keyspace for Cassandra connection here
    # Replace these placeholders with your actual Astra DB session and keyspace objects or connection
    # For example (pseudo code):
    # from cassandra.cluster import Cluster
    # cluster = Cluster([...])
    # session = cluster.connect('your_keyspace')
    # astra_vector_store = Cassandra(..., session=session, keyspace='your_keyspace')

    # For demonstration, we'll initialize without session/keyspace but this needs to be fixed by user
    session = None
    keyspace = None

    astra_vector_store = Cassandra(
        embedding=embeddings,
        table_name="qa_mini_demo",
        session=session,
        keyspace=keyspace
    )

    try:
        astra_vector_store.add_documents(splits)
        st.write("Documents added to vectorstore.")
    except Exception as e:
        st.error(f"Error adding docs to vector store: {e}")

    retriever = astra_vector_store.as_retriever()

    llm = ChatGroq(groq_api_key=GROQ_KEY, model_name="llama3-8b-8192")

    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
    wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

    return retriever, wiki, llm

# --- Fixed GraphState with retriever/wiki/llm ---
class GraphState(TypedDict):
    question: str
    documents: List[Document]
    vector_confident: bool
    generation: str
    retriever: Any
    wiki: Any
    llm: Any

# --- Extract Answer Helper ---
def extract_answer(docs: List[Document]) -> str:
    seen = set()
    chunks = []
    for doc in docs[:3]:
        lines = doc.page_content.split("\n")
        for line in lines:
            clean = line.strip()
            if not clean or clean.lower() in seen or any(clean.lower().startswith(prefix) for prefix in (
                "table of contents", "posts", "archive", "faq", "tags", "|")):
                continue
            seen.add(clean.lower())
            chunks.append(clean)
    merged = "\n\n".join(chunks).strip()
    last_period = merged.rfind(".")
    return merged[:last_period + 1] if last_period != -1 else merged

# --- Vectorstore Retrieval Node ---
def retrieve_vectorstore(state):
    question = state["question"]
    threshold = 0.6  # Set to 0 during debugging to avoid filtering out docs
    retriever = state["retriever"]

    results = []
    if hasattr(retriever, "similarity_search_with_score"):
        try:
            results = retriever.similarity_search_with_score(question, k=3)
        except Exception as e:
            st.warning(f"Error during similarity search: {e}")
    elif hasattr(retriever, "vectorstore"):
        try:
            results = retriever.vectorstore.similarity_search_with_score(question, k=3)
        except Exception as e:
            st.warning(f"Error during vectorstore similarity search: {e}")
    else:
        try:
            docs = retriever.invoke(question)
            return {**state, "documents": docs, "vector_confident": False}
        except Exception as e:
            st.warning(f"Error invoking retriever: {e}")
            return {**state, "documents": [], "vector_confident": False}

    st.write(f"Similarity results (score/source): {[ (score, doc.metadata.get('source', '')) for doc, score in results ]}")

    confident = False
    docs = []
    for doc, score in results:
        if score >= threshold:
            confident = True
            docs.append(doc)

    st.write(f"Docs passed to generation (count): {len(docs)}")

    return {**state, "documents": docs if confident else [], "vector_confident": confident}

# --- Wikipedia Retrieval Node ---
def retrieve_wikipedia(state):
    question = state["question"]
    wiki = state["wiki"]
    try:
        wiki_text = wiki.invoke({"query": question})
    except Exception as e:
        st.warning(f"Wikipedia API error: {e}")
        wiki_text = ""
    docs = [Document(page_content=wiki_text, metadata={"source": "Wikipedia"})]
    return {**state, "documents": docs, "vector_confident": False}

# --- Final Answer Generation Node ---
def generate_final_answer(state):
    question = state["question"]
    docs = state["documents"]
    llm = state["llm"]
    context = extract_answer(docs)
    st.write(f"Context used for generation:\n{context}")

    prompt = f"""
You are an expert assistant helping answer user questions using the provided context.

Your job is to provide an accurate, informative, and well-structured answer using only the context below.
If the context doesn't cover the full answer, you can still infer reasonable insights — but do not hallucinate or make up data.

Make the answer clear and helpful, even for beginners.

Context:
{context}

Question:
{question}

Answer:
"""
    try:
        result = llm.invoke(prompt)
        st.write(f"LLM raw result: {result}")
        
        # Fix: Better handling of LLM response
        if hasattr(result, "content") and result.content and result.content.strip():
            return {**state, "generation": result.content.strip()}
        elif isinstance(result, str) and result.strip():
            return {**state, "generation": result.strip()}
        else:
            return {**state, "generation": "Sorry, I couldn't generate a proper answer from the available context."}
            
    except Exception as e:
        st.error(f"LLM invocation error: {e}")
        return {**state, "generation": "Sorry, I couldn't generate an answer due to an LLM error."}

# --- Build LangGraph Workflow ---
def build_workflow(retriever, wiki, llm):
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve_vectorstore", retrieve_vectorstore)
    workflow.add_node("retrieve_wikipedia", retrieve_wikipedia)
    workflow.add_node("generate_final_answer", generate_final_answer)

    def route_after_vectorstore(state):
        return "generate_final_answer" if state["vector_confident"] else "retrieve_wikipedia"

    workflow.add_edge(START, "retrieve_vectorstore")
    workflow.add_conditional_edges("retrieve_vectorstore", route_after_vectorstore, {
        "generate_final_answer": "generate_final_answer",
        "retrieve_wikipedia": "retrieve_wikipedia"
    })
    workflow.add_edge("retrieve_wikipedia", "generate_final_answer")
    workflow.add_edge("generate_final_answer", END)

    compiled = workflow.compile()

    def wrapper(question: str):
        initial_state: GraphState = {
            "question": question,
            "documents": [],
            "vector_confident": False,
            "generation": "",
            "retriever": retriever,
            "wiki": wiki,
            "llm": llm
        }
        
        # Fix: Properly extract the final state from stream
        final_state = None
        for state_update in compiled.stream(initial_state):
            for node_name, node_state in state_update.items():
                final_state = node_state
        
        return final_state if final_state else initial_state

    return wrapper

# --- Main Streamlit Chat Interface ---
def main():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "source" in message and message["source"]:
                if isinstance(message["source"], list):
                    for src in message["source"]:
                        st.caption(f"Source: {src}")
                else:
                    st.caption(f"Source: {message['source']}")
            if "source_info" in message:
                st.caption(f"Document: {message['source_info']}")

    url_list = [url.strip() for url in user_urls.split(",") if url.strip()] if user_urls.strip() else None
    retriever, wiki, llm = setup_pipeline(custom_urls=url_list)
    rag_workflow = build_workflow(retriever, wiki, llm)

    if prompt := st.chat_input("Ask anything ..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                output_state = rag_workflow(prompt)
                answer = output_state.get("generation", "").strip()
                docs = output_state.get("documents", [])
                sources = list({doc.metadata.get("source", "Wikipedia") for doc in docs})

            if answer:
                message_placeholder.markdown(answer)
            else:
                message_placeholder.markdown("❌ Sorry, I couldn't generate an answer.")

            for src in sources:
                st.caption(f"Source: {src}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "source": sources
        })

if __name__ == "__main__":
    main()
