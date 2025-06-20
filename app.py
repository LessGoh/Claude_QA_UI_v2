import streamlit as st
import os
from utils.auth import check_authentication, login_page, get_user_index_id
from utils.navigation import setup_navigation
from utils.config import get_api_keys
from utils.index_manager import IndexManager

# Page configuration
st.set_page_config(
    page_title="PDF QA Bot v2",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'current_index' not in st.session_state:
    st.session_state.current_index = None
if 'index_manager' not in st.session_state:
    st.session_state.index_manager = None
if 'document_metadata' not in st.session_state:
    st.session_state.document_metadata = []
if 'chat_histories' not in st.session_state:
    st.session_state.chat_histories = {}
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'

def main():
    # Check authentication
    if not st.session_state.authenticated:
        login_page()
        return
    
    # Get API keys
    openai_key, pinecone_key, using_secrets = get_api_keys()
    
    if not openai_key or not pinecone_key:
        st.error("🔐 API keys not configured. Please check your secrets configuration.")
        st.info("Required: OPENAI_API_KEY and PINECONE_API_KEY")
        return
    
    # Set environment variables
    os.environ["OPENAI_API_KEY"] = openai_key
    os.environ["PINECONE_API_KEY"] = pinecone_key
    
    # Initialize index manager
    if not st.session_state.index_manager:
        st.session_state.index_manager = IndexManager(pinecone_key)
    
    # Setup navigation
    setup_navigation()
    
    # Get current page
    from utils.navigation import get_current_page
    current_page = get_current_page()
    
    # Route to appropriate page
    if current_page == 'chat':
        from pages.chat import render_chat_page
        render_chat_page()
    elif current_page == 'upload':
        from pages.upload import render_upload_page
        render_upload_page()
    elif current_page == 'documents':
        from pages.documents import render_documents_page
        render_documents_page()
    elif current_page == 'index_management':
        render_index_management_page()
    else:
        # Home page
        render_home_page()

def render_home_page():
    """Render the home page"""
    st.markdown("## 📄 PDF Question Answering System v2")
    
    # Display current user and index info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"👤 **Пользователь:** {st.session_state.username}")
    with col2:
        user_id = get_user_index_id(st.session_state.username)
        current_index_type = "Личный" if st.session_state.current_index and st.session_state.current_index.endswith(f"-{user_id}") else "Общий"
        st.info(f"📊 **Индекс:** {current_index_type}")
    with col3:
        if st.button("🚪 Выйти"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.rerun()
    
    # Welcome message
    st.markdown("""
    ### Добро пожаловать в улучшенную систему работы с PDF документами!
    
    **Новые возможности:**
    - 🔐 Система аутентификации для команды
    - 📁 Личные и общие индексы документов
    - 💬 Улучшенный чат-интерфейс
    - 📎 Удобная загрузка документов
    - 📋 Управление документами
    
    Используйте навигацию слева для перехода между разделами.
    """)
    
    # Quick stats
    if st.session_state.index_manager:
        st.markdown("### 📊 Быстрая статистика")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Personal index stats
            user_id = get_user_index_id(st.session_state.username)
            personal_index = f"pdf-qa-personal-{user_id}"
            personal_stats = st.session_state.index_manager.get_index_stats(personal_index)
            if personal_stats["status"] == "success":
                st.metric("🔒 Личные документы", f"{personal_stats['total_vectors']:,}")
            else:
                st.metric("🔒 Личные документы", "0")
        
        with col2:
            # Shared index stats
            shared_stats = st.session_state.index_manager.get_index_stats("pdf-qa-shared")
            if shared_stats["status"] == "success":
                st.metric("👥 Общие документы", f"{shared_stats['total_vectors']:,}")
            else:
                st.metric("👥 Общие документы", "0")

def render_index_management_page():
    """Render index management page"""
    st.markdown("# 🔄 Управление индексами")
    
    if not st.session_state.index_manager:
        st.error("Менеджер индексов не инициализирован")
        return
    
    # Current index info
    st.markdown("## 📊 Текущий индекс")
    user_id = get_user_index_id(st.session_state.username)
    current_index = st.session_state.current_index or f"pdf-qa-personal-{user_id}"
    index_type = "Личный" if current_index.endswith(f"-{user_id}") else "Общий"
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Активный индекс:** {index_type}")
        st.info(f"**Имя индекса:** {current_index}")
    
    with col2:
        stats = st.session_state.index_manager.get_index_stats(current_index)
        if stats["status"] == "success":
            st.metric("Документов в индексе", f"{stats['total_vectors']:,}")
            st.metric("Заполненность", f"{stats['index_fullness']:.2%}")
        else:
            st.error("Не удалось получить статистику")
    
    # Index switching
    st.markdown("## 🔄 Переключение индексов")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔒 Переключиться на личный индекс", use_container_width=True):
            user_id = get_user_index_id(st.session_state.username)
            st.session_state.current_index = f"pdf-qa-personal-{user_id}"
            st.success("Переключились на личный индекс")
            st.rerun()
    
    with col2:
        if st.button("👥 Переключиться на общий индекс", use_container_width=True):
            st.session_state.current_index = "pdf-qa-shared"
            st.success("Переключились на общий индекс")
            st.rerun()
    
    # Advanced operations
    st.markdown("## ⚙️ Дополнительные операции")
    
    with st.expander("🗑️ Очистка индекса"):
        st.warning("⚠️ Это действие нельзя отменить!")
        
        if st.button("Очистить текущий индекс"):
            result = st.session_state.index_manager.clear_index(current_index)
            if result["status"] == "success":
                st.success(result["message"])
            else:
                st.error(result["message"])

if __name__ == "__main__":
    main()