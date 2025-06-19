import streamlit as st
import time
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from utils.auth import get_user_index_id

class IndexManager:
    def __init__(self, pinecone_key):
        self.pinecone_key = pinecone_key
        self.pc = Pinecone(api_key=pinecone_key)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        self.spec = ServerlessSpec(cloud='aws', region='us-east-1')
        
        # Initialize indices
        self._ensure_shared_index()
        if st.session_state.username:
            self._ensure_personal_index(st.session_state.username)
    
    def _ensure_shared_index(self):
        """Ensure shared index exists"""
        index_name = "pdf-qa-shared"
        existing_indexes = [item['name'] for item in self.pc.list_indexes().indexes]
        
        if index_name not in existing_indexes:
            try:
                self.pc.create_index(
                    index_name,
                    dimension=1536,
                    metric='cosine',
                    spec=self.spec
                )
                # Wait for index to be ready
                while not self.pc.describe_index(index_name).status['ready']:
                    time.sleep(1)
            except Exception as e:
                st.error(f"Ошибка создания общего индекса: {str(e)}")
    
    def _ensure_personal_index(self, username):
        """Ensure personal index exists for user"""
        user_id = get_user_index_id(username)
        index_name = f"pdf-qa-personal-{user_id}"
        existing_indexes = [item['name'] for item in self.pc.list_indexes().indexes]
        
        if index_name not in existing_indexes:
            try:
                self.pc.create_index(
                    index_name,
                    dimension=1536,
                    metric='cosine',
                    spec=self.spec
                )
                # Wait for index to be ready
                while not self.pc.describe_index(index_name).status['ready']:
                    time.sleep(1)
            except Exception as e:
                st.error(f"Ошибка создания личного индекса: {str(e)}")
    
    def get_vectorstore(self, index_type="shared", username=None):
        """Get vectorstore for specified index type"""
        if index_type == "personal" and username:
            user_id = get_user_index_id(username)
            index_name = f"pdf-qa-personal-{user_id}"
        else:
            index_name = "pdf-qa-shared"
        
        try:
            index = self.pc.Index(index_name)
            return PineconeVectorStore(index, self.embeddings, "text")
        except Exception as e:
            st.error(f"Ошибка подключения к индексу {index_name}: {str(e)}")
            return None
    
    def get_current_vectorstore(self):
        """Get vectorstore for current user's selected index"""
        if not st.session_state.current_index:
            # Default to personal index
            user_id = get_user_index_id(st.session_state.username)
            st.session_state.current_index = f"pdf-qa-personal-{user_id}"
        
        try:
            index = self.pc.Index(st.session_state.current_index)
            return PineconeVectorStore(index, self.embeddings, "text")
        except Exception as e:
            st.error(f"Ошибка подключения к индексу: {str(e)}")
            return None
    
    def get_index_stats(self, index_name):
        """Get statistics for specified index"""
        try:
            index = self.pc.Index(index_name)
            stats = index.describe_index_stats()
            return {
                "status": "success",
                "total_vectors": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def clear_index(self, index_name):
        """Clear all documents from specified index"""
        try:
            index = self.pc.Index(index_name)
            index.delete(delete_all=True)
            return {"status": "success", "message": "Индекс очищен успешно"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def list_all_indexes(self):
        """List all indexes in account"""
        try:
            indexes = [item['name'] for item in self.pc.list_indexes().indexes]
            return {"status": "success", "indexes": indexes}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def switch_index(self, index_type, username=None):
        """Switch to specified index type"""
        if index_type == "personal" and username:
            user_id = get_user_index_id(username)
            st.session_state.current_index = f"pdf-qa-personal-{user_id}"
        else:
            st.session_state.current_index = "pdf-qa-shared"