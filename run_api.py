#!/usr/bin/env python3
"""
API Server Entry Point for AI Customer Agent.

This script provides a dedicated entry point for running only the FastAPI backend
of the AI Customer Agent. It focuses on API-specific configuration and management.

Usage:
    python run_api.py [--port PORT] [--host HOST] [--reload] [--workers WORKERS]

Examples:
    python run_api.py                    # Run API on default port 8001
    python run_api.py --port 8080        # Run API on port 8080
    python run_api.py --host 0.0.0.0     # Run API on all interfaces
    python run_api.py --no-reload        # Disable auto-reload for production
"""

import argparse
import logging
import signal
import sys
import time
import uvicorn
from contextlib import asynccontextmanager
import os

# Add the project root to Python path to ensure imports work
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/api_server.log')
    ]
)
logger = logging.getLogger(__name__)


class APIServer:
    """Manages the FastAPI server lifecycle and configuration."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8001, reload: bool = True, workers: int = 1):
        self.host = host
        self.port = port
        self.reload = reload
        self.workers = workers
        self.server = None
        self.shutdown_event = False
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_event = True
        
        # If we have a running server, trigger shutdown
        if self.server:
            self.server.should_exit = True
    
    def print_startup_info(self):
        """Print API server startup information."""
        print("\n" + "="*50)
        print("🚀 AI Customer Agent - API Server")
        print("="*50)
        print(f"📡 Server Configuration:")
        print(f"   • Host: {self.host}")
        print(f"   • Port: {self.port}")
        print(f"   • Auto-reload: {'Enabled' if self.reload else 'Disabled'}")
        print(f"   • Workers: {self.workers}")
        print(f"\n🔗 API Endpoints:")
        print(f"   • API Base URL: http://localhost:{self.port}")
        print(f"   • Documentation: http://localhost:{self.port}/docs")
        print(f"   • Health Check: http://localhost:{self.port}/health")
        print(f"   • System Info: http://localhost:{self.port}/info")
        print(f"\n📊 Logs:")
        print(f"   • Console: Real-time logging")
        print(f"   • File: ./logs/api_server.log")
        print(f"\n⏳ Starting API server... (Press Ctrl+C to stop)")
        print("="*50 + "\n")
    
    def validate_configuration(self) -> bool:
        """Validate server configuration before starting."""
        try:
            # Check if port is available
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex((self.host, self.port))
                if result == 0:
                    logger.error(f"Port {self.port} is already in use")
                    return False
            
            # Check if required modules are available
            try:
                import src.api.main
                logger.debug("FastAPI application module loaded successfully")
            except ImportError as e:
                logger.error(f"Failed to import FastAPI application: {str(e)}")
                return False
            
            # Check if logs directory exists
            os.makedirs("logs", exist_ok=True)
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {str(e)}")
            return False
    
    def run_server(self):
        """Run the FastAPI server with uvicorn."""
        try:
            logger.info(f"Starting FastAPI server on {self.host}:{self.port}")
            
            # Configure uvicorn
            config = uvicorn.Config(
                app="src.api.main:app",
                host=self.host,
                port=self.port,
                reload=self.reload,
                workers=self.workers,
                log_level="info",
                access_log=True,
                timeout_keep_alive=5,
                timeout_graceful_shutdown=10
            )
            
            self.server = uvicorn.Server(config)
            
            # Start the server
            self.server.run()
            
        except Exception as e:
            logger.error(f"Failed to start API server: {str(e)}")
            raise
    
    def health_check(self) -> bool:
        """Perform a health check on the running API server."""
        try:
            import requests
            # Bypass proxy settings for localhost health check
            session = requests.Session()
            session.trust_env = False  # Don't use proxy settings
            response = session.get(f"http://127.0.0.1:{self.port}/health", timeout=10)
            if response.status_code == 200:
                logger.debug(f"Health check successful: {response.status_code}")
                return True
            else:
                logger.error(f"Health check returned non-200 status: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Health check failed: {str(e)}")
            return False
    
    def wait_for_ready(self, timeout: int = 60) -> bool:
        """Wait for the API server to be ready."""
        logger.info("Waiting for API server to be ready...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.health_check():
                logger.info("API server is ready and responding to requests")
                return True
            time.sleep(1)
        
        logger.error(f"API server failed to become ready within {timeout} seconds")
        return False


def parse_arguments():
    """Parse command line arguments for the API server."""
    parser = argparse.ArgumentParser(
        description="AI Customer Agent - API Server Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_api.py                    # Run API on default port 8001
  python run_api.py --port 8080        # Run API on port 8080  
  python run_api.py --host 0.0.0.0     # Run API on all interfaces
  python run_api.py --no-reload        # Disable auto-reload for production
  python run_api.py --workers 4        # Run with 4 worker processes
        """
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port to run the server on (default: 8001)"
    )
    
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="Disable auto-reload (useful for production)"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Logging level (default: info)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the API server."""
    args = parse_arguments()
    
    # Set log level based on arguments
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    # Create API server instance
    api_server = APIServer(
        host=args.host,
        port=args.port,
        reload=not args.no_reload,
        workers=args.workers
    )
    
    # Print startup information
    api_server.print_startup_info()
    
    # Validate configuration
    if not api_server.validate_configuration():
        logger.error("Configuration validation failed. Exiting.")
        sys.exit(1)
    
    try:
        # Start the server in a separate thread to allow for graceful shutdown
        import threading
        
        def run_server():
            try:
                api_server.run_server()
            except Exception as e:
                logger.error(f"Server thread error: {str(e)}")
                sys.exit(1)
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # Wait for server to be ready
        if not api_server.wait_for_ready():
            logger.error("Server failed to start properly")
            sys.exit(1)
        
        # Keep main thread alive until shutdown signal
        logger.info("API server is running and ready to accept requests")
        while not api_server.shutdown_event:
            time.sleep(1)
            
        logger.info("Shutdown signal received, stopping server...")
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)
    finally:
        logger.info("API server shutdown complete")


if __name__ == "__main__":
    main()
