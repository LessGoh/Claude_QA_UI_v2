import streamlit as st
import os

def get_api_keys():
    """Get API keys from Streamlit secrets or environment variables"""
    openai_key = None
    pinecone_key = None
    
    # Try to get from secrets first
    try:
        openai_key = st.secrets["OPENAI_API_KEY"]
        pinecone_key = st.secrets["PINECONE_API_KEY"]
        return openai_key, pinecone_key, True
    except (KeyError, FileNotFoundError):
        # Fall back to environment variables
        openai_key = os.environ.get("OPENAI_API_KEY")
        pinecone_key = os.environ.get("PINECONE_API_KEY")
        return openai_key, pinecone_key, False