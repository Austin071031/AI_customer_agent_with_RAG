"""
Streamlit Web Interface for AI Customer Agent.

This module provides a web-based user interface for the AI Customer Agent
using Streamlit. It includes chat interface, knowledge base management,
configuration panel, and GPU status monitoring.
"""

import streamlit as st
import requests
import json
import time
from typing import List, Dict, Optional
import os
import sys

# Add the project root to the Python path to allow imports from src
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Constants
API_BASE_URL = "http://localhost:8001"
CHAT_ENDPOINT = f"{API_BASE_URL}/api/chat"
KB_ENDPOINT = f"{API_BASE_URL}/api/knowledge-base"
EXCEL_FILES_ENDPOINT = f"{API_BASE_URL}/api/excel-files"
CONFIG_ENDPOINT = f"{API_BASE_URL}/api/config"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"
INFO_ENDPOINT = f"{API_BASE_URL}/info"


def check_api_health() -> bool:
    """
    Check if the FastAPI backend is running and healthy.
    
    Returns:
        bool: True if API is healthy, False otherwise
    """
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        return response.status_code == 200 and response.json().get("status") == "healthy"
    except requests.exceptions.RequestException:
        return False


def get_system_info() -> Optional[Dict]:
    """
    Get system information from the API.
    
    Returns:
        Optional[Dict]: System information dictionary or None if API is unavailable
    """
    try:
        response = requests.get(INFO_ENDPOINT, timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except requests.exceptions.RequestException:
        return None


def send_chat_message(message: str, use_kb: bool = True, stream: bool = False) -> Dict:
    """
    Send a chat message to the API.
    
    Args:
        message: The user message to send
        use_kb: Whether to use knowledge base context
        stream: Whether to stream the response
        
    Returns:
        Dict: Response from the API
        
    Raises:
        Exception: If the API request fails
    """
    url = f"{CHAT_ENDPOINT}/stream" if stream else f"{CHAT_ENDPOINT}/"
    
    payload = {
        "message": message,
        "use_knowledge_base": use_kb,
        "stream": stream
    }
    
    response = requests.post(url, json=payload, timeout=300)
    
    if response.status_code != 200:
        raise Exception(f"API Error: {response.status_code} - {response.text}")
    
    return response.json()


def stream_chat_message(message: str, use_kb: bool = True):
    """
    Stream a chat message and yield response chunks.
    
    Args:
        message: The user message to send
        use_kb: Whether to use knowledge base context
        
    Yields:
        str: Response chunks from the streaming API
    """
    url = f"{CHAT_ENDPOINT}/stream"
    
    payload = {
        "message": message,
        "use_knowledge_base": use_kb,
        "stream": True
    }
    
    try:
        response = requests.post(url, json=payload, stream=True, timeout=60)
        
        if response.status_code != 200:
            yield f"Error: {response.status_code} - {response.text}"
            return
        
        for line in response.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                chunk = line[6:]  # Remove "data: " prefix
                if chunk == "[DONE]":
                    break
                elif chunk.startswith("ERROR:"):
                    yield f"❌ {chunk}"
                    break
                else:
                    yield chunk
                    
    except requests.exceptions.RequestException as e:
        yield f"❌ Connection error: {str(e)}"


def get_conversation_history() -> List[Dict]:
    """
    Get the current conversation history from the API.
    
    Returns:
        List[Dict]: List of chat messages
        
    Raises:
        Exception: If the API request fails
    """
    response = requests.get(f"{CHAT_ENDPOINT}/history", timeout=5)
    
    if response.status_code != 200:
        raise Exception(f"API Error: {response.status_code} - {response.text}")
    
    data = response.json()
    return data.get("messages", [])


def clear_conversation_history() -> bool:
    """
    Clear the conversation history.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        response = requests.delete(f"{CHAT_ENDPOINT}/history", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def upload_to_knowledge_base(file) -> bool:
    """
    Upload a file to the knowledge base.
    
    Args:
        file: The file object to upload
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        files = {"files": (file.name, file.getvalue(), file.type)}
        response = requests.post(f"{KB_ENDPOINT}/documents", files=files, timeout=300)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def search_knowledge_base(query: str, k: int = 3) -> List[Dict]:
    """
    Search the knowledge base for similar documents.
    
    Args:
        query: The search query
        k: Number of results to return
        
    Returns:
        List[Dict]: List of similar documents
        
    Raises:
        Exception: If the API request fails
    """
    response = requests.get(f"{KB_ENDPOINT}/search", params={"query": query, "k": k}, timeout=10)
    
    if response.status_code != 200:
        raise Exception(f"API Error: {response.status_code} - {response.text}")
    
    return response.json()


def clear_knowledge_base() -> bool:
    """
    Clear all documents from the knowledge base.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        response = requests.delete(f"{KB_ENDPOINT}/", timeout=10)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def list_excel_files() -> List[Dict]:
    """
    List all Excel files from the API.
    
    Returns:
        List of Excel file dictionaries
    """
    try:
        response = requests.get(f"{EXCEL_FILES_ENDPOINT}/", timeout=10)
        if response.status_code == 200:
            return response.json()
        return []
    except requests.exceptions.RequestException:
        return []


def get_excel_file(file_id: str) -> Optional[Dict]:
    """
    Get specific Excel file details.
    
    Args:
        file_id: ID of the Excel file
        
    Returns:
        Excel file dictionary if found, None otherwise
    """
    try:
        response = requests.get(f"{EXCEL_FILES_ENDPOINT}/{file_id}", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except requests.exceptions.RequestException:
        return None


def delete_excel_file(file_id: str) -> bool:
    """
    Delete an Excel file.
    
    Args:
        file_id: ID of the Excel file to delete
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        response = requests.delete(f"{EXCEL_FILES_ENDPOINT}/{file_id}", timeout=10)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def get_excel_sheets(file_id: str, sheet_name: Optional[str] = None) -> List[Dict]:
    """
    Get sheet data for an Excel file.
    
    Args:
        file_id: ID of the Excel file
        sheet_name: Optional specific sheet name
        
    Returns:
        List of sheet data dictionaries
    """
    try:
        params = {}
        if sheet_name:
            params["sheet_name"] = sheet_name
            
        response = requests.get(f"{EXCEL_FILES_ENDPOINT}/{file_id}/sheets", 
                              params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        return []
    except requests.exceptions.RequestException:
        return []


def search_excel_data(query: str, file_id: Optional[str] = None, 
                     sheet_name: Optional[str] = None, 
                     column_name: Optional[str] = None,
                     max_results: int = 50) -> List[Dict]:
    """
    Search for data within Excel files.
    
    Args:
        query: Search query string
        file_id: Optional specific file ID to search within
        sheet_name: Optional specific sheet name
        column_name: Optional specific column name
        max_results: Maximum number of results to return
        
    Returns:
        List of search result dictionaries
    """
    try:
        if file_id:
            # Search within specific file
            params = {
                "query": query,
                "max_results": max_results
            }
            if sheet_name:
                params["sheet_name"] = sheet_name
            if column_name:
                params["column_name"] = column_name
                
            response = requests.get(f"{EXCEL_FILES_ENDPOINT}/{file_id}/search", 
                                  params=params, timeout=10)
        else:
            # Search across all files
            payload = {
                "query": query,
                "max_results": max_results
            }
            if sheet_name:
                payload["sheet_name"] = sheet_name
            if column_name:
                payload["column_name"] = column_name
                
            response = requests.post(f"{EXCEL_FILES_ENDPOINT}/search", 
                                   json=payload, timeout=10)
            
        if response.status_code == 200:
            data = response.json()
            return data.get("results", [])
        return []
    except requests.exceptions.RequestException:
        return []


def get_configuration() -> Optional[Dict]:
    """
    Get current configuration from the API.
    
    Returns:
        Optional[Dict]: Configuration dictionary or None if API is unavailable
    """
    try:
        response = requests.get(CONFIG_ENDPOINT, timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except requests.exceptions.RequestException:
        return None


def update_configuration(config: Dict) -> bool:
    """
    Update configuration settings.
    
    Args:
        config: Configuration dictionary to update
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        response = requests.put(CONFIG_ENDPOINT, json=config, timeout=10)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def check_gpu_status() -> Dict:
    """
    Check GPU status and utilization.
    
    Returns:
        Dict: GPU status information
    """
    try:
        import torch
        gpu_status = {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        
        if gpu_status["available"]:
            # Get GPU memory info
            gpu_status["memory_allocated"] = torch.cuda.memory_allocated(0) / 1024**3  # Convert to GB
            gpu_status["memory_reserved"] = torch.cuda.memory_reserved(0) / 1024**3  # Convert to GB
            
        return gpu_status
    except ImportError:
        return {"available": False, "error": "PyTorch not installed"}


def init_session_state():
    """
    Initialize Streamlit session state variables.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "api_healthy" not in st.session_state:
        st.session_state.api_healthy = False
    
    if "use_knowledge_base" not in st.session_state:
        st.session_state.use_knowledge_base = True
    
    if "stream_responses" not in st.session_state:
        st.session_state.stream_responses = True
    
    if "system_info" not in st.session_state:
        st.session_state.system_info = None
    
    if "gpu_status" not in st.session_state:
        st.session_state.gpu_status = {}


def display_chat_message(role: str, content: str):
    """
    Display a chat message in the Streamlit interface.
    
    Args:
        role: Message role ("user" or "assistant")
        content: Message content
    """
    with st.chat_message(role):
        st.markdown(content)


def main():
    """
    Main function for the Streamlit application.
    """
    # Page configuration
    st.set_page_config(
        page_title="AI Customer Agent",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    init_session_state()
    
    # Header
    st.title("🤖 AI Customer Agent")
    st.markdown("A local AI customer service agent with DeepSeek API integration and knowledge base support")
    
    # Check API health
    st.session_state.api_healthy = check_api_health()
    st.session_state.system_info = get_system_info()
    st.session_state.gpu_status = check_gpu_status()
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # API Status
        status_color = "🟢" if st.session_state.api_healthy else "🔴"
        st.subheader(f"{status_color} API Status")
        
        if st.session_state.api_healthy:
            st.success("FastAPI backend is running and healthy")
            if st.session_state.system_info:
                st.write(f"**Version:** {st.session_state.system_info.get('system', {}).get('version', 'Unknown')}")
                st.write(f"**Environment:** {st.session_state.system_info.get('system', {}).get('environment', 'Unknown')}")
        else:
            st.error("FastAPI backend is not available")
            st.info("Please ensure the FastAPI server is running on http://localhost:8001")
        
        # GPU Status
        st.subheader("🖥️ GPU Status")
        if st.session_state.gpu_status.get("available"):
            st.success("GPU acceleration available")
            st.write(f"**Device:** {st.session_state.gpu_status.get('device_name', 'Unknown')}")
            if "memory_allocated" in st.session_state.gpu_status:
                st.write(f"**Memory Used:** {st.session_state.gpu_status['memory_allocated']:.2f} GB")
        else:
            st.warning("GPU acceleration not available")
            if st.session_state.gpu_status.get("error"):
                st.write(f"**Note:** {st.session_state.gpu_status['error']}")
        
        # Chat Settings
        st.subheader("💬 Chat Settings")
        st.session_state.use_knowledge_base = st.checkbox(
            "Use Knowledge Base", 
            value=st.session_state.use_knowledge_base,
            help="Use local knowledge base for context in responses"
        )
        st.session_state.stream_responses = st.checkbox(
            "Stream Responses", 
            value=st.session_state.stream_responses,
            help="Stream responses in real-time for better user experience"
        )
        
        # Knowledge Base Management
        st.subheader("📚 Knowledge Base")
        
        # File upload - now includes Excel formats
        uploaded_files = st.file_uploader(
            "Add documents to knowledge base",
            type=["pdf", "txt", "docx", "doc", "md", "xlsx", "xls", "xlsm", "xlsb"],
            accept_multiple_files=True,
            help="Supported formats: PDF, TXT, DOCX, DOC, MD, XLSX, XLS, XLSM, XLSB"
        )
        
        if uploaded_files and st.button("Upload Documents"):
            with st.spinner("Uploading documents..."):
                success_count = 0
                for uploaded_file in uploaded_files:
                    if upload_to_knowledge_base(uploaded_file):
                        success_count += 1
                
                if success_count > 0:
                    st.success(f"Successfully uploaded {success_count} document(s)")
                else:
                    st.error("Failed to upload documents")
        
        # Knowledge base actions
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔍 Search KB", use_container_width=True):
                query = st.text_input("Enter search query:")
                if query:
                    try:
                        results = search_knowledge_base(query)
                        if results:
                            st.write("Search Results:")
                            for i, result in enumerate(results[:3]):
                                with st.expander(f"Result {i+1}"):
                                    st.write(result.get("content", "")[:200] + "...")
                        else:
                            st.info("No results found")
                    except Exception as e:
                        st.error(f"Search failed: {str(e)}")
        
        with col2:
            if st.button("🗑️ Clear KB", use_container_width=True):
                if clear_knowledge_base():
                    st.success("Knowledge base cleared")
                else:
                    st.error("Failed to clear knowledge base")
        
        # Excel File Management Panel
        st.subheader("📊 Excel File Management")
        
        # List Excel files
        if st.button("📋 List Excel Files", use_container_width=True):
            excel_files = list_excel_files()
            if excel_files:
                st.write(f"Found {len(excel_files)} Excel files:")
                for file in excel_files:
                    with st.expander(f"📄 {file.get('file_name', 'Unknown')}"):
                        st.write(f"**ID:** {file.get('id', 'N/A')}")
                        st.write(f"**Size:** {file.get('file_size', 0)} bytes")
                        st.write(f"**Sheets:** {', '.join(file.get('sheet_names', []))}")
                        st.write(f"**Uploaded:** {file.get('uploaded_at', 'Unknown')}")
                        
                        # File actions
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if st.button("View Details", key=f"view_{file.get('id')}"):
                                file_details = get_excel_file(file.get('id'))
                                if file_details:
                                    st.json(file_details)
                        with col2:
                            if st.button("View Sheets", key=f"sheets_{file.get('id')}"):
                                sheets = get_excel_sheets(file.get('id'))
                                if sheets:
                                    st.write(f"Found {len(sheets)} sheets:")
                                    for sheet in sheets:
                                        with st.expander(f"Sheet: {sheet.get('sheet_name', 'Unknown')}"):
                                            st.write(f"**Rows:** {sheet.get('row_count', 0)}")
                                            st.write(f"**Columns:** {sheet.get('column_count', 0)}")
                                            st.write(f"**Sample Data:**")
                                            if sheet.get('sample_data'):
                                                st.dataframe(sheet['sample_data'])
                        with col3:
                            if st.button("Delete", key=f"delete_{file.get('id')}"):
                                if delete_excel_file(file.get('id')):
                                    st.success("File deleted successfully")
                                    st.rerun()
                                else:
                                    st.error("Failed to delete file")
            else:
                st.info("No Excel files found in the database")
        
        # Excel Search
        if st.button("🔍 Search Excel Data", use_container_width=True):
            search_query = st.text_input("Search query:", key="excel_search")
            if search_query:
                with st.spinner("Searching Excel data..."):
                    results = search_excel_data(search_query, max_results=20)
                    if results:
                        st.write(f"Found {len(results)} matches:")
                        for result in results:
                            with st.expander(f"Match in {result.get('file_name', 'Unknown')}"):
                                st.write(f"**File:** {result.get('file_name')}")
                                st.write(f"**Sheet:** {result.get('sheet_name')}")
                                st.write(f"**Location:** Row {result.get('row_number')}, Column {result.get('column_name')}")
                                st.write(f"**Value:** {result.get('cell_value')}")
                    else:
                        st.info("No matches found")
        
        # Configuration
        st.subheader("⚙️ Configuration")
        if st.button("View Current Config"):
            config = get_configuration()
            if config:
                st.json(config)
            else:
                st.error("Failed to retrieve configuration")
    
    # Main content area - Chat Interface
    st.header("💬 Chat with AI Agent")
    
    # Display conversation history
    for message in st.session_state.messages:
        display_chat_message(message["role"], message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_chat_message("user", prompt)
        
        # Generate AI response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                if st.session_state.stream_responses and st.session_state.api_healthy:
                    # Stream the response
                    for chunk in stream_chat_message(prompt, st.session_state.use_knowledge_base):
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                    
                    message_placeholder.markdown(full_response)
                else:
                    # Non-streaming response
                    with st.spinner("Thinking..."):
                        response = send_chat_message(
                            prompt, 
                            st.session_state.use_knowledge_base, 
                            stream=False
                        )
                        full_response = response.get("response", "No response received")
                        message_placeholder.markdown(full_response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                error_message = f"❌ Error: {str(e)}"
                message_placeholder.markdown(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
    
    # Conversation management
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔄 Clear Chat"):
            if clear_conversation_history():
                st.session_state.messages = []
                st.rerun()
            else:
                st.error("Failed to clear conversation history")
    
    with col2:
        if st.session_state.messages:
            st.info(f"Conversation has {len(st.session_state.messages)} messages")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**AI Customer Agent** • Built with Streamlit, FastAPI, and DeepSeek API • "
        "[Report Issues](https://github.com/your-repo/ai-customer-agent/issues)"
    )


if __name__ == "__main__":
    main()
