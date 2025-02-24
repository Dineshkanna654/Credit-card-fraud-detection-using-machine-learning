# Credit Card Fraud Detection System with User Authentication

This application uses machine learning to detect fraudulent credit card transactions. The system includes user authentication with secure password storage and session management.

## Features

- **User Authentication**
  - User registration with email and username
  - Secure password storage using SHA-256 hashing with salt
  - Session management using Streamlit's session state
  - Authentication-protected routes

- **Fraud Detection**
  - Machine learning model to detect fraudulent transactions
  - Interactive user interface to input transaction features
  - Visualization of model performance and feature importance
  - Real-time transaction analysis

## Project Structure

```
creditcard-fraud-detection/
├── app.py              # Main application file
├── auth_system.py      # Authentication system module
├── users.json          # User database (created automatically)
├── creditcard.csv      # Dataset for fraud detection
└── README.md           # This file
```

## Setup and Installation

1. Make sure you have Python 3.7+ installed
2. Install dependencies:
   ```
   pip install streamlit pandas numpy scikit-learn matplotlib seaborn
   ```
3. Place all files in the same directory
4. Run the application:
   ```
   streamlit run app.py
   ```

## Usage

1. **Registration**
   - Click "Register" and fill in the registration form
   - Provide a username, email, and password (minimum 8 characters)
   - Submit the form to create your account

2. **Login**
   - Use your username and password to log in
   - Sessions last for 7 days by default (can be modified in `auth_system.py`)

3. **Fraud Detection**
   - Once logged in, you'll have access to the fraud detection system
   - Input transaction features in the provided fields
   - Click "Analyze Transaction" to get the prediction result
   - Explore model performance metrics and feature importance

4. **Logout**
   - Use the logout button in the sidebar to end your session

## Security Features

- Passwords are never stored in plain text
- Each password is hashed with a unique salt
- Session expiration after 7 days
- Protected routes using the `@auth_required` decorator

## Notes

- User data is stored in a JSON file (`users.json`)
- For production use, consider using a more robust database system
- The fraud detection model uses logistic regression and balanced sampling