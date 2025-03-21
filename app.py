import streamlit as st
import pandas as pd
import os
from file_processor import process_file
from agent import DataAnalystAgent

# Page configuration
st.set_page_config(
    page_title="Data Detective Agent",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding: 1rem;
    }
    .stButton button {
        width: 100%;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
    }
    .chat-message.user {
        background-color: #f0f2f6;
    }
    .chat-message.assistant {
        background-color: #e1f5fe;
    }
    .chat-message .avatar {
        width: 40px;
        margin-right: 1rem;
    }
    .chat-message .content {
        flex-grow: 1;
    }
    /* Improved visualization container */
    .visualization-container {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    /* Make sure charts are responsive */
    .plotly-chart {
        width: 100%;
        height: auto;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "file_data" not in st.session_state:
    st.session_state.file_data = None
if "file_type" not in st.session_state:
    st.session_state.file_type = None
if "agent" not in st.session_state:
    st.session_state.agent = None
if "error" not in st.session_state:
    st.session_state.error = None
if "reset_chat" not in st.session_state:
    st.session_state.reset_chat = False

# Header
st.title("📊 Data Detective Agent")
st.markdown("Upload your data files and ask questions to get insights and visualizations.")

# Create two columns for layout
col1, col2 = st.columns([1, 3])

# File upload and information panel
with col1:
    st.subheader("Upload Data")
    uploaded_file = st.file_uploader("Choose a file", 
                                     type=["csv", "xlsx", "pdf", "txt", "docx", "png", "jpg", "jpeg"],
                                     help="Upload CSV, Excel, PDF, Word, Text, or Image files")
    
    if uploaded_file:
        try:
            # Process the file
            data, file_type = process_file(uploaded_file)
            
            # If a new file is uploaded, reset the chat
            if st.session_state.file_data is not None and st.session_state.reset_chat:
                st.session_state.messages = []
                st.session_state.reset_chat = False
            
            # Store in session state
            st.session_state.file_data = data
            st.session_state.file_type = file_type
            
            # Initialize the agent
            st.session_state.agent = DataAnalystAgent(data, file_type)
            
            # Display file information
            st.success(f"File uploaded successfully: {uploaded_file.name}")
            st.text(f"File type detected: {file_type}")
            
            # Display preview based on file type
            if file_type == "tabular":
                st.subheader("Data Preview")
                st.dataframe(data.head())
                st.text(f"Shape: {data.shape[0]} rows, {data.shape[1]} columns")
                
                # Add data profiling information
                st.subheader("Data Profile")
                num_cols = data.select_dtypes(include=['int64', 'float64']).columns
                if not num_cols.empty:
                    st.text("Numerical Columns Summary:")
                    st.dataframe(data[num_cols].describe())
                
                # Missing values
                missing_values = data.isnull().sum()
                if missing_values.sum() > 0:
                    st.text("Missing Values:")
                    st.dataframe(missing_values[missing_values > 0])
                
            elif file_type in ["text", "pdf", "docx"]:
                st.subheader("Text Preview")
                st.text(data[:500] + "..." if len(data) > 500 else data)
                st.text(f"Length: {len(data)} characters")
            elif file_type == "image":
                st.subheader("Extracted Text")
                st.text(data[:500] + "..." if len(data) > 500 else data)
                
            # Clear any previous errors
            st.session_state.error = None
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.session_state.error = f"Error processing file: {str(e)}"
    
    # Add a reset button
    if st.session_state.file_data is not None:
        if st.button("Reset Chat"):
            st.session_state.messages = []
            st.session_state.reset_chat = True
            st.rerun()

# Main chat interface
with col2:
    st.subheader("Ask Questions About Your Data")
    
    # Display any errors
    if st.session_state.error:
        st.error(st.session_state.error)
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("type") == "chart":
                try:
                    # Use a container for better styling
                    with st.container():
                        st.plotly_chart(message["content"], use_container_width=True)
                        if message.get("explanation"):
                            st.markdown(message["explanation"])
                except Exception as e:
                    st.error(f"Error displaying chart: {str(e)}")
            else:
                st.markdown(message["content"])
    
    # Input for new questions
    if st.session_state.file_data is not None:
        query = st.chat_input("Ask your question here...")
        if query:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": query})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(query)
            
            # Get response from agent
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    if st.session_state.agent:
                        message_placeholder = st.empty()
                        
                        try:
                            # Process the query and get response
                            response = st.session_state.agent.process_query(query)
                            
                            # Handle different response types
                            if isinstance(response, tuple) and len(response) >= 2:
                                fig, explanation = response[:2]
                                
                                # Display the visualization
                                st.plotly_chart(fig, use_container_width=True)
                                st.markdown(explanation)
                                
                                # Add to chat history
                                st.session_state.messages.append({
                                    "role": "assistant", 
                                    "content": fig, 
                                    "type": "chart", 
                                    "explanation": explanation
                                })
                            else:
                                # Display text response
                                message_placeholder.markdown(response)
                                
                                # Add to chat history
                                st.session_state.messages.append({
                                    "role": "assistant", 
                                    "content": response
                                })
                        except Exception as e:
                            error_message = f"Error: {str(e)}"
                            message_placeholder.error(error_message)
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": error_message
                            })
    else:
        st.info("Please upload a file to start asking questions.")

# Add sidebar with information
with st.sidebar:
    st.title("About Data Detective")
    st.markdown("""
    This tool helps you analyze data files by:
    - Processing various file formats
    - Answering questions about your data
    - Creating visualizations
    - Extracting insights
    
    **Supported file types:**
    - CSV, Excel (tabular data)
    - PDF, Word, Text (text data)
    - Images (OCR for text extraction)
    
    **Example questions:**
    - "Summarize this data"
    - "Show me a bar chart of sales by region"
    - "What are the key insights from this text?"
    - "What's the correlation between price and rating?"
    - "Visualize the distribution of values"
    - "Create a dashboard with key metrics"
    """)
    
    # Add information about visualization capabilities
    st.subheader("Visualization Capabilities")
    st.markdown("""
    The Data Detective can create various types of visualizations:
    - Bar charts and histograms
    - Line charts and time series
    - Scatter plots and bubble charts
    - Pie charts and donut charts
    - Heatmaps and correlation matrices
    - Box plots and violin plots
    
    Just ask questions like:
    - "Show me a visualization of..."
    - "Create a chart showing..."
    - "Visualize the relationship between..."
    """)
