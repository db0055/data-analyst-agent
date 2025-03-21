import pandas as pd
import PyPDF2
import docx
import re
from PIL import Image
import pytesseract
import io
import os

def process_file(uploaded_file):
    """
    Process uploaded file and return the data and file type.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
    
    Returns:
        tuple: (data, file_type) where data is the processed content and
               file_type is one of: "tabular", "text", "pdf", "docx", "image"
    """
    # Get file extension
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    try:
        # Process CSV and Excel files (tabular data)
        if file_extension in ['csv', 'xlsx', 'xls']:
            if file_extension == 'csv':
                try:
                    # Try different encodings if utf-8 fails
                    try:
                        data = pd.read_csv(uploaded_file, encoding='utf-8')
                    except UnicodeDecodeError:
                        data = pd.read_csv(uploaded_file, encoding='latin1')
                except Exception as e:
                    # Try with additional parameters for common CSV issues
                    data = pd.read_csv(uploaded_file, sep=None, engine='python')
            else:
                data = pd.read_excel(uploaded_file)
            
            # Validate data
            if data.empty:
                raise ValueError("The file is empty. Please upload a file with data.")
                
            return data, "tabular"
        
        # Process PDF files
        elif file_extension == 'pdf':
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text()
            
            if not text.strip():
                raise ValueError("Could not extract text from PDF. The file might be scanned or protected.")
                
            return text, "pdf"
        
        # Process Word documents
        elif file_extension == 'docx':
            doc = docx.Document(uploaded_file)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            
            if not text.strip():
                raise ValueError("Could not extract text from Word document. The file might be corrupted.")
                
            return text, "docx"
        
        # Process text files
        elif file_extension == 'txt':
            try:
                text = uploaded_file.getvalue().decode('utf-8')
            except UnicodeDecodeError:
                text = uploaded_file.getvalue().decode('latin1')
                
            return text, "text"
        
        # Process image files with OCR
        elif file_extension in ['png', 'jpg', 'jpeg']:
            try:
                image = Image.open(uploaded_file)
                text = pytesseract.image_to_string(image)
                
                if not text.strip():
                    raise ValueError("Could not extract text from image. Try a clearer image or a different file format.")
                    
                return text, "image"
            except Exception as e:
                raise ValueError(f"Error processing image: {str(e)}. Make sure pytesseract is properly installed.")
        
        else:
            raise ValueError(f"Unsupported file type: {file_extension}. Supported formats include CSV, Excel, PDF, Word, Text, and image files.")
    
    except Exception as e:
        raise Exception(f"Error processing {file_extension} file: {str(e)}")

