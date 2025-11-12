"""
Basic Usage Examples for AI Customer Agent

This module provides examples of how to use the AI Customer Agent API
for basic chat functionality and knowledge base integration.
"""

import requests
import json
import os
from typing import Dict, List, Optional


class AICustomerAgentClient:
    """Client for interacting with the AI Customer Agent API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def send_chat_message(self, message: str, use_knowledge_base: bool = True) -> Dict:
        """
        Send a chat message to the AI agent
        
        Args:
            message: The user's message
            use_knowledge_base: Whether to use knowledge base context
            
        Returns:
            Dict containing the AI response
        """
        response = self.session.post(
            f"{self.base_url}/api/chat",
            json={
                "message": message,
                "use_kb": use_knowledge_base
            }
        )
        response.raise_for_status()
        return response.json()
    
    def get_conversation_history(self) -> List[Dict]:
        """Get the current conversation history"""
        response = self.session.get(f"{self.base_url}/api/chat/history")
        response.raise_for_status()
        return response.json()
    
    def clear_conversation_history(self) -> Dict:
        """Clear the conversation history"""
        response = self.session.delete(f"{self.base_url}/api/chat/history")
        response.raise_for_status()
        return response.json()
    
    def search_knowledge_base(self, query: str, k: int = 3) -> List[Dict]:
        """
        Search the knowledge base for relevant documents
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant documents
        """
        response = self.session.get(
            f"{self.base_url}/api/knowledge-base/search",
            params={"query": query, "k": k}
        )
        response.raise_for_status()
        return response.json()
    
    def get_configuration(self) -> Dict:
        """Get current application configuration"""
        response = self.session.get(f"{self.base_url}/api/config")
        response.raise_for_status()
        return response.json()


def example_basic_chat():
    """Example: Basic chat interaction"""
    print("=== Basic Chat Example ===")
    
    client = AICustomerAgentClient()
    
    # Send a message
    response = client.send_chat_message(
        "Hello! I need help with customer service inquiries."
    )
    print(f"AI Response: {response.get('response', 'No response')}")
    print(f"Response Time: {response.get('response_time', 'N/A')}s")
    print()


def example_conversation_flow():
    """Example: Multi-turn conversation"""
    print("=== Conversation Flow Example ===")
    
    client = AICustomerAgentClient()
    
    messages = [
        "Hi, I'm having trouble with my account login.",
        "What are the steps to reset my password?",
        "Thank you! Can you also tell me about your refund policy?"
    ]
    
    for message in messages:
        print(f"You: {message}")
        response = client.send_chat_message(message)
        print(f"AI: {response.get('response', 'No response')}")
        print()
    
    # Show conversation history
    history = client.get_conversation_history()
    print(f"Conversation has {len(history)} messages")
    print()


def example_knowledge_base_search():
    """Example: Searching the knowledge base"""
    print("=== Knowledge Base Search Example ===")
    
    client = AICustomerAgentClient()
    
    # Search for relevant documents
    results = client.search_knowledge_base("customer service policy", k=2)
    
    print(f"Found {len(results)} relevant documents:")
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.get('content', 'No content')[:100]}...")
        print(f"   Source: {doc.get('metadata', {}).get('file_path', 'Unknown')}")
    print()


def example_without_knowledge_base():
    """Example: Chat without knowledge base context"""
    print("=== Chat Without Knowledge Base Example ===")
    
    client = AICustomerAgentClient()
    
    # Chat without using knowledge base
    response = client.send_chat_message(
        "What is your company's mission?",
        use_knowledge_base=False
    )
    
    print(f"Response (no KB): {response.get('response', 'No response')}")
    print()


def example_configuration_check():
    """Example: Checking application configuration"""
    print("=== Configuration Check Example ===")
    
    client = AICustomerAgentClient()
    
    config = client.get_configuration()
    print("Current Configuration:")
    print(f"API Model: {config.get('api', {}).get('model', 'N/A')}")
    print(f"Knowledge Base Status: {config.get('knowledge_base', {}).get('status', 'N/A')}")
    print(f"GPU Available: {config.get('system', {}).get('gpu_available', 'N/A')}")
    print()


if __name__ == "__main__":
    print("AI Customer Agent - Basic Usage Examples")
    print("=" * 50)
    print()
    
    try:
        # Run all examples
        example_basic_chat()
        example_conversation_flow()
        example_knowledge_base_search()
        example_without_knowledge_base()
        example_configuration_check()
        
        print("All examples completed successfully!")
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the AI Customer Agent API.")
        print("Make sure the API server is running on http://localhost:8000")
        print("Start the server with: python run_api.py")
        
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        print("Check if the API endpoints are available and configured correctly.")
        
    except Exception as e:
        print(f"Unexpected error: {e}")
