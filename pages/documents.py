import streamlit as st
from datetime import datetime
from utils.document_manager import DocumentManager

class DocumentsPage:
    def __init__(self):
        self.doc_manager = DocumentManager()
    
    def render(self):
        st.markdown("# 📋 Управление документами")
        
        # Statistics
        self._render_statistics()
        
        # Document lists
        st.markdown("## 📚 Списки документов")
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["🔒 Мои документы", "👥 Общие документы", "📊 Все документы"])
        
        with tab1:
            self._render_personal_documents()
        
        with tab2:
            self._render_shared_documents()
        
        with tab3:
            self._render_all_documents()
    
    def _render_statistics(self):
        """Render document statistics"""
        stats = self.doc_manager.get_statistics()
        
        st.markdown("### 📊 Статистика")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Всего документов", stats['total_documents'])
        
        with col2:
            st.metric("Общий размер", f"{stats['total_size_mb']:.1f} MB")
        
        with col3:
            st.metric("Фрагментов текста", stats['total_chunks'])
        
        with col4:
            st.metric("Общих документов", stats['shared_documents'])
    
    def _render_personal_documents(self):
        """Render personal documents list"""
        personal_docs = self.doc_manager.get_personal_documents(st.session_state.username)
        
        if not personal_docs:
            st.info("📭 У вас пока нет личных документов. Загрузите документы через раздел 'Загрузка документов'.")
            return
        
        st.markdown(f"### 🔒 Ваши личные документы ({len(personal_docs)})")
        
        for doc in sorted(personal_docs, key=lambda x: x.get('upload_date', ''), reverse=True):
            self._render_document_card(doc, show_owner=False)
    
    def _render_shared_documents(self):
        """Render shared documents list"""
        shared_docs = self.doc_manager.get_shared_documents()
        
        if not shared_docs:
            st.info("📭 В общем индексе пока нет документов.")
            return
        
        st.markdown(f"### 👥 Общие документы команды ({len(shared_docs)})")
        
        for doc in sorted(shared_docs, key=lambda x: x.get('upload_date', ''), reverse=True):
            self._render_document_card(doc, show_owner=True)
    
    def _render_all_documents(self):
        """Render all documents list"""
        all_docs = self.doc_manager.get_all_documents()
        
        if not all_docs:
            st.info("📭 Документы не найдены.")
            return
        
        st.markdown(f"### 📊 Все документы ({len(all_docs)})")
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            filter_user = st.selectbox(
                "Фильтр по пользователю:",
                ["Все"] + [doc['upload_user'] for doc in all_docs if 'upload_user' in doc],
                key="user_filter"
            )
        
        with col2:
            filter_index = st.selectbox(
                "Фильтр по индексу:",
                ["Все", "Личные", "Общие"],
                key="index_filter"
            )
        
        # Apply filters
        filtered_docs = all_docs
        
        if filter_user != "Все":
            filtered_docs = [doc for doc in filtered_docs if doc.get('upload_user') == filter_user]
        
        if filter_index == "Личные":
            filtered_docs = [doc for doc in filtered_docs if 'pdf-qa-personal-' in doc.get('index_name', '')]
        elif filter_index == "Общие":
            filtered_docs = [doc for doc in filtered_docs if doc.get('index_name') == 'pdf-qa-shared']
        
        # Display filtered documents
        for doc in sorted(filtered_docs, key=lambda x: x.get('upload_date', ''), reverse=True):
            self._render_document_card(doc, show_owner=True, show_index=True)
    
    def _render_document_card(self, doc, show_owner=True, show_index=False):
        """Render a single document card"""
        # Parse upload date
        try:
            upload_date = datetime.fromisoformat(doc.get('upload_date', ''))
            date_str = upload_date.strftime("%d.%m.%Y %H:%M")
        except:
            date_str = "Неизвестно"
        
        # File size
        file_size_mb = doc.get('file_size', 0) / (1024 * 1024)
        
        # Index type
        index_name = doc.get('index_name', '')
        if 'pdf-qa-personal-' in index_name:
            index_type = "🔒 Личный"
        else:
            index_type = "👥 Общий"
        
        # Create expandable card
        with st.expander(f"📄 {doc.get('filename', 'Неизвестный файл')}"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**📅 Дата загрузки:** {date_str}")
                if show_owner:
                    st.markdown(f"**👤 Загрузил:** {doc.get('upload_user', 'Неизвестно')}")
                st.markdown(f"**📏 Размер:** {file_size_mb:.1f} MB")
                st.markdown(f"**🧩 Фрагментов:** {doc.get('chunk_count', 0)}")
                if show_index:
                    st.markdown(f"**📊 Индекс:** {index_type}")
            
            with col2:
                # Action buttons
                if doc.get('upload_user') == st.session_state.username:
                    if st.button("🗑️ Удалить", key=f"delete_{doc.get('id')}", help="Удалить документ"):
                        self._delete_document(doc)
                else:
                    st.markdown("*Чужой документ*")
    
    def _delete_document(self, doc):
        """Delete a document"""
        if st.session_state.get(f"confirm_delete_{doc.get('id')}", False):
            # Actually delete
            self.doc_manager.delete_document(doc.get('id'))
            st.success(f"✅ Документ '{doc.get('filename')}' удален")
            st.rerun()
        else:
            # Show confirmation
            st.session_state[f"confirm_delete_{doc.get('id')}"] = True
            st.warning(f"⚠️ Подтвердите удаление документа '{doc.get('filename')}'")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Подтвердить", key=f"confirm_yes_{doc.get('id')}"):
                    self.doc_manager.delete_document(doc.get('id'))
                    st.success(f"✅ Документ '{doc.get('filename')}' удален")
                    del st.session_state[f"confirm_delete_{doc.get('id')}"]
                    st.rerun()
            
            with col2:
                if st.button("❌ Отмена", key=f"confirm_no_{doc.get('id')}"):
                    del st.session_state[f"confirm_delete_{doc.get('id')}"]
                    st.rerun()

def render_documents_page():
    """Render the documents page"""
    docs_page = DocumentsPage()
    docs_page.render()