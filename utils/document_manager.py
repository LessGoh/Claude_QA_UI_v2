import streamlit as st
import json
from datetime import datetime
from typing import List, Dict, Any
from utils.auth import get_user_index_id

class DocumentManager:
    """Simple document metadata manager using session state"""
    
    def __init__(self):
        # document_metadata is initialized in main app.py
        pass
    
    def add_document(self, doc_metadata: Dict[str, Any]):
        """Add document metadata"""
        doc_metadata['id'] = len(st.session_state.document_metadata) + 1
        st.session_state.document_metadata.append(doc_metadata)
    
    def get_user_documents(self, username: str) -> List[Dict[str, Any]]:
        """Get documents for a specific user"""
        return [doc for doc in st.session_state.document_metadata if doc.get('upload_user') == username]
    
    def get_shared_documents(self) -> List[Dict[str, Any]]:
        """Get documents in shared index"""
        return [doc for doc in st.session_state.document_metadata if doc.get('index_name') == 'pdf-qa-shared']
    
    def get_personal_documents(self, username: str) -> List[Dict[str, Any]]:
        """Get documents in personal index"""
        user_id = get_user_index_id(username)
        return [doc for doc in st.session_state.document_metadata if doc.get('index_name') == f'pdf-qa-personal-{user_id}']
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents"""
        return st.session_state.document_metadata
    
    def delete_document(self, doc_id: int):
        """Delete document by ID"""
        st.session_state.document_metadata = [doc for doc in st.session_state.document_metadata if doc.get('id') != doc_id]
    
    def get_document_by_id(self, doc_id: int) -> Dict[str, Any]:
        """Get document by ID"""
        for doc in st.session_state.document_metadata:
            if doc.get('id') == doc_id:
                return doc
        return None
    
    def get_documents_by_index(self, index_name: str) -> List[Dict[str, Any]]:
        """Get documents by index name"""
        return [doc for doc in st.session_state.document_metadata if doc.get('index_name') == index_name]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get document statistics"""
        docs = st.session_state.document_metadata
        
        total_docs = len(docs)
        total_size = sum(doc.get('file_size', 0) for doc in docs)
        total_chunks = sum(doc.get('chunk_count', 0) for doc in docs)
        
        # By index
        shared_docs = [doc for doc in docs if doc.get('index_name') == 'pdf-qa-shared']
        personal_docs = [doc for doc in docs if 'pdf-qa-personal-' in doc.get('index_name', '')]
        
        return {
            'total_documents': total_docs,
            'total_size_mb': total_size / (1024 * 1024),
            'total_chunks': total_chunks,
            'shared_documents': len(shared_docs),
            'personal_documents': len(personal_docs)
        }