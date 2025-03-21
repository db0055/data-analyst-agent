import os
import subprocess
import sys

# Ensure we're using the latest version of the groq package
try:
    import groq
    print(f"Using groq package version: {groq.__version__}")
except (ImportError, AttributeError):
    print("Groq package not found or version info unavailable.")
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

# Run the app
print("\nStarting Data Detective Agent...\n")
subprocess.call(["streamlit", "run", "app.py"])