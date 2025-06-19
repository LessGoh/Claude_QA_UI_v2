import streamlit as st

# Team members
TEAM_MEMBERS = ["Стас", "Макс", "Вова", "Семен"]

# Username to index-safe mapping
USERNAME_TO_INDEX = {
    "Стас": "stas",
    "Макс": "max", 
    "Вова": "vova",
    "Семен": "semen"
}

def check_authentication():
    """Check if user is authenticated"""
    return st.session_state.get('authenticated', False) and st.session_state.get('username') in TEAM_MEMBERS

def login_page():
    """Display login page"""
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <h1>🔐 Вход в систему</h1>
        <p>Выберите ваше имя для входа в систему</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### Выберите пользователя:")
        
        # User selection
        selected_user = st.selectbox(
            "Имя пользователя:",
            options=[""] + TEAM_MEMBERS,
            index=0,
            format_func=lambda x: "Выберите пользователя..." if x == "" else x
        )
        
        # Login button
        if st.button("🚀 Войти", disabled=not selected_user, use_container_width=True):
            if selected_user:
                st.session_state.authenticated = True
                st.session_state.username = selected_user
                st.success(f"Добро пожаловать, {selected_user}!")
                st.rerun()
        
        st.markdown("---")
        st.info("💡 Выберите ваше имя из списка для входа в систему")

def logout():
    """Logout current user"""
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.current_index = None
    if 'index_manager' in st.session_state:
        del st.session_state.index_manager
    st.rerun()

def get_current_user():
    """Get current authenticated user"""
    return st.session_state.get('username')

def get_user_index_id(username):
    """Get index-safe user ID"""
    return USERNAME_TO_INDEX.get(username, username.lower())