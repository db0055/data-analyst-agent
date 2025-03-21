import streamlit as st
import os
import sys
from app import *

# This is the main entry point for Streamlit Cloud

if __name__ == "__main__":
    # Set page configuration and run the app directly
    st.set_page_config(
        page_title="Data Detective Agent",
        page_icon="📊",
        layout="wide"
    )
    
    # The rest of the app will be imported from app.py
    # This file only serves as the entry point for Streamlit Cloud
