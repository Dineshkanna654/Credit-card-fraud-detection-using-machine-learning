import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from auth_system import AuthSystem, login_page, register_page, logout, auth_required

# Set page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

    # Add CSS to improve the UI
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    .auth-container {
        max-width: 500px;
        margin: 0 auto;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        background-color: #f8f9fa;
    }
    .stForm {
        background-color: white;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_and_process_data():
    """Load and process the data with caching for better performance"""
    try:
        data = pd.read_csv('creditcard.csv')
        
        # Separate legitimate and fraudulent transactions
        legit = data[data.Class == 0]
        fraud = data[data.Class == 1]
        
        # Undersample legitimate transactions
        legit_sample = legit.sample(n=len(fraud), random_state=2)
        balanced_data = pd.concat([legit_sample, fraud], axis=0)
        
        return balanced_data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_resource
def train_model(X_train, y_train):
    """Train the model with caching"""
    try:
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None

@auth_required
def main_app():
    st.title("🔍 Credit Card Fraud Detection System")
    st.write(f"Welcome, {st.session_state['username']}! This system uses machine learning to detect potentially fraudulent credit card transactions.")
    
    # Add logout button in sidebar
    with st.sidebar:
        st.title(f"👤 {st.session_state['username']}")
        st.button("Logout", on_click=logout)
    
    # Load and process data
    data = load_and_process_data()
    if data is None:
        return
    
    # Data preparation
    X = data.drop(columns="Class", axis=1)
    y = data["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Train model
    model = train_model(X_train, y_train)
    if model is None:
        return
    
    # Display model performance metrics
    with st.expander("📊 Model Performance Metrics"):
        col1, col2 = st.columns(2)
        
        with col1:
            train_acc = accuracy_score(model.predict(X_train), y_train)
            test_acc = accuracy_score(model.predict(X_test), y_test)
            st.metric("Training Accuracy", f"{train_acc:.2%}")
            st.metric("Testing Accuracy", f"{test_acc:.2%}")
        
        with col2:
            conf_matrix = confusion_matrix(y_test, model.predict(X_test))
            st.write("Confusion Matrix:")
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
            st.pyplot(fig)
    
    # Transaction input section
    st.header("🔎 Transaction Analysis")
    st.write("Enter the transaction features to analyze:")
    
    # Create columns for input fields
    cols = st.columns(3)
    input_values = []
    
    for i in range(30):
        with cols[i % 3]:
            input_val = st.number_input(
                f'Feature V{i + 1}',
                value=0.0,
                step=0.1,
                format="%.6f",
                help=f"Enter value for feature V{i + 1}"
            )
            input_values.append(input_val)
    
    # Prediction section
    if st.button("Analyze Transaction", use_container_width=True):
        features = np.array(input_values).reshape(1, -1)
        prediction = model.predict(features)
        probability = model.predict_proba(features)
        
        # Display results with proper formatting
        st.markdown("---")
        if prediction[0] == 0:
            st.success("✅ Legitimate Transaction Detected")
            st.info(f"Confidence: {probability[0][0]:.2%}")
        else:
            st.error("⚠️ Fraudulent Transaction Detected")
            st.warning(f"Confidence: {probability[0][1]:.2%}")
        
        # Show feature importance
        with st.expander("📈 Feature Importance Analysis"):
            importance = pd.DataFrame({
                'Feature': [f'V{i + 1}' for i in range(30)],
                'Importance': abs(model.coef_[0])
            }).sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.barplot(data=importance.head(10), x='Feature', y='Importance')
            plt.xticks(rotation=45)
            st.pyplot(fig)

def main():
    # Initialize session state
    if 'auth_page' not in st.session_state:
        st.session_state['auth_page'] = 'login'
    
    # Check if user is already authenticated
    auth_system = AuthSystem()
    authenticated, username = auth_system.verify_session()
    
    if authenticated:
        main_app()
    else:
        # Render authentication pages
        with st.container():
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown('<div class="auth-container">', unsafe_allow_html=True)
                
                if st.session_state['auth_page'] == 'login':
                    login_page()
                elif st.session_state['auth_page'] == 'register':
                    register_page()
                
                st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()