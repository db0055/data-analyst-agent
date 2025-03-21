# Data Detective Agent

An interactive AI-powered data analysis tool that processes various file formats and answers questions about the data.

## Features

- Process multiple file formats (CSV, Excel, PDF, Text, Word, Images)
- Chat-based interface for asking questions about your data
- Visualizations and insights powered by AI
- Responsive interface with data preview and profiling

## Deployment on Streamlit Cloud

### Prerequisites
1. A [GitHub](https://github.com) account
2. A [Streamlit Cloud](https://share.streamlit.io) account
3. A [Groq](https://console.groq.com) account for the API key

### How to Deploy

1. Fork or clone this repository to your GitHub account
2. Go to [Streamlit Cloud](https://share.streamlit.io) and sign in
3. Click "New app"
4. Select this repository
5. Set the main file path as: `streamlit_app.py`
6. Add your secrets in the Streamlit Cloud dashboard:
   - Go to "Advanced settings"
   - In the "Secrets" section, add:
   ```
   GROQ_API_KEY = "your-actual-groq-api-key"
   ```
7. Click "Deploy"

### Local Development

1. Clone the repository
2. Install requirements: `pip install -r requirements.txt`
3. Create a `.streamlit/secrets.toml` file with your GROQ_API_KEY
4. Run: `streamlit run streamlit_app.py`

## Supported File Types

- CSV, Excel (tabular data)
- PDF, Word, Text (text data)
- Images (OCR for text extraction)

## Example Questions

- "Summarize this data"
- "Show me a bar chart of sales by region"
- "What are the key insights from this text?"
- "What's the correlation between price and rating?"
