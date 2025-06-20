import streamlit as st
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any

class ChatHistoryManager:
    """Simple chat history manager using session state for persistence"""
    
    def __init__(self):
        # chat_histories is initialized in main app.py
        pass
    
    def get_conversation_history(self, conversation_id: str, username: str) -> List[Dict[str, Any]]:
        """Get conversation history for a specific conversation and user"""
        key = f"{username}_{conversation_id}"
        return st.session_state.chat_histories.get(key, [])
    
    def add_message(self, conversation_id: str, username: str, message: Dict[str, Any]):
        """Add a message to conversation history"""
        key = f"{username}_{conversation_id}"
        
        if key not in st.session_state.chat_histories:
            st.session_state.chat_histories[key] = []
        
        # Add timestamp if not present
        if 'timestamp' not in message:
            message['timestamp'] = datetime.now().isoformat()
        
        st.session_state.chat_histories[key].append(message)
        
        # Keep only last 50 messages per conversation
        if len(st.session_state.chat_histories[key]) > 50:
            st.session_state.chat_histories[key] = st.session_state.chat_histories[key][-50:]
    
    def clear_conversation(self, conversation_id: str, username: str):
        """Clear conversation history"""
        key = f"{username}_{conversation_id}"
        if key in st.session_state.chat_histories:
            st.session_state.chat_histories[key] = []
    
    def get_user_conversations(self, username: str) -> List[Dict[str, Any]]:
        """Get all conversations for a user (last 10)"""
        conversations = []
        
        for key, history in st.session_state.chat_histories.items():
            if key.startswith(f"{username}_") and history:
                conversation_id = key.replace(f"{username}_", "")
                last_message = history[-1] if history else None
                
                conversations.append({
                    'id': conversation_id,
                    'last_message': last_message,
                    'message_count': len(history),
                    'last_activity': last_message['timestamp'] if last_message else None
                })
        
        # Sort by last activity and return last 10
        conversations.sort(key=lambda x: x['last_activity'] or '', reverse=True)
        return conversations[:10]
    
    def delete_conversation(self, conversation_id: str, username: str):
        """Delete a conversation"""
        key = f"{username}_{conversation_id}"
        if key in st.session_state.chat_histories:
            del st.session_state.chat_histories[key]
    
    def get_recent_messages(self, username: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent messages across all conversations for a user"""
        all_messages = []
        
        for key, history in st.session_state.chat_histories.items():
            if key.startswith(f"{username}_"):
                conversation_id = key.replace(f"{username}_", "")
                for message in history:
                    message_copy = message.copy()
                    message_copy['conversation_id'] = conversation_id
                    all_messages.append(message_copy)
        
        # Sort by timestamp and return recent messages
        all_messages.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return all_messages[:limit]