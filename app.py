# app.py
import streamlit as st
import pandas as pd
import os
import hashlib

# Import dashboards
from pages import farmer, agronomist, researcher, technician
from services.utils import init_session_state

USER_FILE = "users.csv"

# ----------------------------
# Utils
# ----------------------------
def hash_password(password: str) -> str:
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    """Load users from CSV"""
    if os.path.exists(USER_FILE):
        return pd.read_csv(USER_FILE)
    return pd.DataFrame(columns=["username", "password", "role"])

def save_user(username, password, role):
    """Save new user into CSV (hashed password)"""
    df = load_users()
    if username in df["username"].values:
        return False  # already exists
    hashed = hash_password(password)
    new_user = pd.DataFrame([[username, hashed, role]], columns=df.columns)
    df = pd.concat([df, new_user], ignore_index=True)
    df.to_csv(USER_FILE, index=False)
    return True

def verify_user(username, password):
    """Check login credentials"""
    df = load_users()
    if df.empty:
        return None
    hashed = hash_password(password)
    user = df[(df["username"] == username) & (df["password"] == hashed)]
    if not user.empty:
        return user.iloc[0]["role"]
    return None

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="AgronIQ — AI Crop & Soil Monitoring",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🌱 AgronIQ – Turning Field Data into Smart Decisions")

# ----------------------------
# Init session
# ----------------------------
init_session_state(st)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "role" not in st.session_state:
    st.session_state.role = None
if "username" not in st.session_state:
    st.session_state.username = None

# ----------------------------
# Login / Register
# ----------------------------
if not st.session_state.logged_in:
    menu = ["🔑 Login", "📝 Register"]
    choice = st.sidebar.radio("Select Action", menu)

    if choice == "🔑 Login":
        st.subheader("🔑 Login to AgronIQ")

        username = st.text_input("👤 Username")
        password = st.text_input("🔑 Password", type="password")

        if st.button("Login"):
            role = verify_user(username, password)
            if role:
                st.session_state.logged_in = True
                st.session_state.role = role
                st.session_state.username = username
                st.success(f"✅ Logged in as {st.session_state.role}")
                st.rerun()
            else:
                st.error("❌ Invalid username or password")

    else:  # Register
        st.subheader("📝 Register New Account")

        new_user = st.text_input("Choose Username")
        new_pass = st.text_input("Choose Password", type="password")
        new_role = st.selectbox(
            "Select Role", ["Farmer", "Agronomist", "Researcher", "Field Technician"]
        )

        if st.button("Register"):
            if new_user and new_pass:
                if save_user(new_user, new_pass, new_role):
                    st.success("🎉 Registration successful! Please login.")
                else:
                    st.error("⚠ Username already exists")
            else:
                st.error("Please fill all fields")

# ----------------------------
# Role-based dashboards
# ----------------------------
else:
    st.sidebar.success(f"👤 Logged in as {st.session_state.username} ({st.session_state.role})")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.role = None
        st.session_state.username = None
        st.rerun()

    # Redirect to dashboards
    if st.session_state.role == "Farmer":
        farmer.app()
    elif st.session_state.role == "Agronomist":
        agronomist.app()
    elif st.session_state.role == "Researcher":
        researcher.app()
    elif st.session_state.role == "Field Technician":
        technician.app()
    else:
        st.error("Unknown role selected.")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("🌱 AgronIQ — AI + IoT Agriculture Platform | Demo Prototype for SIH 2025")
