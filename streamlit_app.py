import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import os

# --- CONFIG ---
DOCS_DIR = "docs"
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    st.error("Please set your OpenAI API key in Streamlit secrets.")
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# --- UI ---
st.title("🏭 Facility Document Assistant")
st.write("Ask questions about your facility based on preloaded documents.")

# --- Load and Index Documents ---
@st.cache_resource(show_spinner=True)
def load_index():
    if not os.path.exists(DOCS_DIR) or not os.listdir(DOCS_DIR):
        st.warning("No documents found in the 'docs/' folder.")
        return None
    documents = SimpleDirectoryReader(DOCS_DIR).load_data()
    return VectorStoreIndex.from_documents(documents)

index = load_index()

# --- Ask Questions ---
if index:
    chat_engine = index.as_chat_engine()
    question = st.text_input("❓ Ask a question about Service Plan")
    if question:
        with st.spinner("Thinking..."):
            response = chat_engine.chat(question)
            st.markdown(f"**Answer:** {response.response}")
else:
    st.stop()
