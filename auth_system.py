import os
import json
import hashlib
import secrets
import datetime
import streamlit as st
from pathlib import Path

# Constants
USER_DB_PATH = "users.json"
SESSION_EXPIRY_DAYS = 7

class AuthSystem:
    def __init__(self):
        self.users_db_path = USER_DB_PATH
        self._initialize_user_db()
        
    def _initialize_user_db(self):
        """Initialize the user database if it doesn't exist."""
        if not os.path.exists(self.users_db_path):
            with open(self.users_db_path, 'w') as f:
                json.dump({"users": []}, f)
    
    def _load_users(self):
        """Load users from the JSON file."""
        try:
            with open(self.users_db_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {"users": []}
    
    def _save_users(self, data):
        """Save users to the JSON file."""
        with open(self.users_db_path, 'w') as f:
            json.dump(data, f, indent=4)
    
    def _hash_password(self, password, salt=None):
        """Hash a password with a salt for secure storage."""
        if salt is None:
            salt = secrets.token_hex(16)
        
        # Create a hash with the password and salt
        hash_obj = hashlib.sha256((password + salt).encode())
        password_hash = hash_obj.hexdigest()
        
        return password_hash, salt
    
    def register_user(self, username, email, password):
        """Register a new user."""
        data = self._load_users()
        
        # Check if username or email already exists
        if any(user['username'] == username for user in data['users']):
            return False, "Username already exists."
        
        if any(user['email'] == email for user in data['users']):
            return False, "Email already exists."
        
        # Hash the password
        password_hash, salt = self._hash_password(password)
        
        # Create new user
        new_user = {
            "username": username,
            "email": email,
            "password_hash": password_hash,
            "salt": salt,
            "created_at": str(datetime.datetime.now()),
            "last_login": None
        }
        
        # Add user to database
        data['users'].append(new_user)
        self._save_users(data)
        
        return True, "Registration successful!"
    
    def authenticate_user(self, username, password):
        """Authenticate a user by username and password."""
        data = self._load_users()
        
        # Find user with matching username
        user = next((user for user in data['users'] if user['username'] == username), None)
        
        if not user:
            return False, "Invalid username or password."
        
        # Verify password
        password_hash, _ = self._hash_password(password, salt=user['salt'])
        
        if password_hash != user['password_hash']:
            return False, "Invalid username or password."
        
        # Update last login
        for i, u in enumerate(data['users']):
            if u['username'] == username:
                data['users'][i]['last_login'] = str(datetime.datetime.now())
        
        self._save_users(data)
        
        return True, "Login successful!"
    
    def create_session(self, username):
        """Create a session for a user using Streamlit's session state."""
        # Store authentication info in session state
        st.session_state['username'] = username
        st.session_state['authenticated'] = True
        st.session_state['expiry'] = (datetime.datetime.now() + 
                                    datetime.timedelta(days=SESSION_EXPIRY_DAYS)).timestamp()
    
    def verify_session(self):
        """Verify if a user has a valid session using Streamlit's session state."""
        # Check if authentication info exists in session state
        if not st.session_state.get('authenticated', False):
            return False, None
        
        # Check if session is expired
        expiry = st.session_state.get('expiry')
        if expiry and datetime.datetime.now().timestamp() > expiry:
            self.end_session()
            return False, None
        
        return True, st.session_state.get('username')
    
    def end_session(self):
        """End a user's session by clearing session state variables."""
        if 'username' in st.session_state:
            del st.session_state['username']
        if 'authenticated' in st.session_state:
            del st.session_state['authenticated']
        if 'expiry' in st.session_state:
            del st.session_state['expiry']
        
        st.session_state['auth_page'] = 'login'

def login_page():
    st.title("🔐 Login")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            if not username or not password:
                st.error("Please fill in all fields.")
                return
                
            auth_system = AuthSystem()
            success, message = auth_system.authenticate_user(username, password)
            
            if success:
                auth_system.create_session(username)
                st.success(message)
                st.rerun()
            else:
                st.error(message)
    
    st.markdown("---")
    st.write("Don't have an account?")
    if st.button("Register"):
        st.session_state['auth_page'] = 'register'
        st.rerun()

def register_page():
    st.title("📝 Register")
    
    with st.form("register_form"):
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submit = st.form_submit_button("Register")
        
        if submit:
            if not username or not email or not password or not confirm_password:
                st.error("Please fill in all fields.")
                return
                
            if password != confirm_password:
                st.error("Passwords do not match.")
                return
                
            if len(password) < 8:
                st.error("Password must be at least 8 characters long.")
                return
                
            auth_system = AuthSystem()
            success, message = auth_system.register_user(username, email, password)
            
            if success:
                st.success(message)
                st.session_state['auth_page'] = 'login'
                st.rerun()
            else:
                st.error(message)
    
    st.markdown("---")
    st.write("Already have an account?")
    if st.button("Login"):
        st.session_state['auth_page'] = 'login'
        st.rerun()

def logout():
    auth_system = AuthSystem()
    auth_system.end_session()
    st.success("You have been logged out.")
    st.rerun()

def auth_required(func):
    """Decorator to require authentication for a page."""
    def wrapper(*args, **kwargs):
        auth_system = AuthSystem()
        authenticated, username = auth_system.verify_session()
        
        if authenticated:
            return func(*args, **kwargs)
        else:
            # Redirect to login
            st.session_state['auth_page'] = 'login'
            st.rerun()
    
    return wrapper