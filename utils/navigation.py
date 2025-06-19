import streamlit as st

def setup_navigation():
    """Setup sidebar navigation"""
    with st.sidebar:
        st.markdown("## 🧭 Навигация")
        
        # Current user info
        st.markdown(f"👤 **Пользователь:** {st.session_state.username}")
        
        # Navigation links
        st.markdown("### 📋 Разделы:")
        
        # Home page
        if st.button("🏠 Главная", use_container_width=True):
            st.session_state.current_page = "home"
            st.rerun()
        
        # Chat page
        if st.button("💬 Чат с документами", use_container_width=True):
            st.session_state.current_page = "chat"
            st.rerun()
        
        # Upload page
        if st.button("📎 Загрузка документов", use_container_width=True):
            st.session_state.current_page = "upload"
            st.rerun()
        
        # Documents page  
        if st.button("📋 Управление документами", use_container_width=True):
            st.session_state.current_page = "documents"
            st.rerun()
        
        # Index management
        if st.button("🔄 Переключение индексов", use_container_width=True):
            st.session_state.current_page = "index_management"
            st.rerun()
        
        st.markdown("---")
        
        # Index switcher
        st.markdown("### 📊 Текущий индекс:")
        index_options = ["Личный", "Общий"]
        current_index_type = "Личный" if st.session_state.current_index and st.session_state.current_index.endswith(f"-{st.session_state.username}") else "Общий"
        
        new_index_type = st.selectbox(
            "Выберите индекс:",
            index_options,
            index=index_options.index(current_index_type) if current_index_type in index_options else 0,
            key="index_selector"
        )
        
        # Switch index if changed
        if new_index_type != current_index_type:
            if new_index_type == "Личный":
                st.session_state.current_index = f"pdf-qa-personal-{st.session_state.username}"
            else:
                st.session_state.current_index = "pdf-qa-shared"
            st.rerun()
        
        st.markdown("---")
        
        # Logout button
        if st.button("🚪 Выйти", use_container_width=True):
            from utils.auth import logout
            logout()

def get_current_page():  
    """Get current page from session state"""
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'home'
    return st.session_state.current_page