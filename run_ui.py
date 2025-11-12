#!/usr/bin/env python3
"""
UI Server Entry Point for AI Customer Agent.

This script provides a dedicated entry point for running only the Streamlit UI
of the AI Customer Agent. It focuses on UI-specific configuration and management.

Usage:
    python run_ui.py [--port PORT] [--host HOST] [--api-url API_URL]

Examples:
    python run_ui.py                    # Run UI on default port 8501
    python run_ui.py --port 8502        # Run UI on port 8502
    python run_ui.py --host 0.0.0.0     # Run UI on all interfaces
    python run_ui.py --api-url http://localhost:8080  # Connect to custom API URL
"""

import argparse
import logging
import signal
import sys
import time
import subprocess
import os
import requests
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/ui_server.log')
    ]
)
logger = logging.getLogger(__name__)


class UIServer:
    """Manages the Streamlit UI server lifecycle and configuration."""
    
    def __init__(self, host: str = "localhost", port: int = 8501, api_url: str = "http://localhost:8001"):
        self.host = host
        self.port = port
        self.api_url = api_url
        self.ui_process: Optional[subprocess.Popen] = None
        self.shutdown_event = False
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_event = True
        self.stop_server()
    
    def print_startup_info(self):
        """Print UI server startup information."""
        print("\n" + "="*50)
        print("🎨 AI Customer Agent - UI Server")
        print("="*50)
        print(f"📡 Server Configuration:")
        print(f"   • Host: {self.host}")
        print(f"   • Port: {self.port}")
        print(f"   • API URL: {self.api_url}")
        print(f"\n🔗 UI Endpoints:")
        print(f"   • Streamlit UI: http://{self.host}:{self.port}")
        print(f"   • API Health: {self.api_url}/health")
        print(f"\n📊 Logs:")
        print(f"   • Console: Real-time logging")
        print(f"   • File: ./logs/ui_server.log")
        print(f"\n⏳ Starting UI server... (Press Ctrl+C to stop)")
        print("="*50 + "\n")
    
    def check_api_availability(self) -> bool:
        """Check if the API server is available and healthy."""
        try:
            health_url = f"{self.api_url}/health"
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                status = health_data.get("status", "unknown")
                if status in ["healthy", "degraded"]:
                    logger.info(f"API server is available and {status}")
                    return True
                else:
                    logger.warning(f"API server is available but status is {status}")
                    return True
            else:
                logger.warning(f"API server health check returned status {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.warning(f"API server is not available: {str(e)}")
            return False
    
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
            
            # Check if Streamlit is available
            try:
                import streamlit
                logger.debug("Streamlit module is available")
            except ImportError as e:
                logger.error(f"Streamlit is not installed: {str(e)}")
                return False
            
            # Check if UI module exists
            ui_module_path = os.path.join("src", "ui", "streamlit_app.py")
            if not os.path.exists(ui_module_path):
                logger.error(f"UI module not found at {ui_module_path}")
                return False
            
            # Check if logs directory exists
            os.makedirs("logs", exist_ok=True)
            
            # Warn if API is not available (but don't fail - UI can still start)
            if not self.check_api_availability():
                logger.warning("API server is not available. UI will start but may not function properly.")
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {str(e)}")
            return False
    
    def start_server(self) -> bool:
        """Start the Streamlit UI server."""
        try:
            logger.info(f"Starting Streamlit UI on {self.host}:{self.port}...")
            
            # Set environment variables for Streamlit configuration
            env = os.environ.copy()
            env["STREAMLIT_SERVER_PORT"] = str(self.port)
            env["STREAMLIT_SERVER_ADDRESS"] = self.host
            env["STREAMLIT_SERVER_HEADLESS"] = "true"
            env["STREAMLIT_BROWSER_SERVER_ADDRESS"] = self.host
            
            # Build the command for Streamlit
            cmd = [
                sys.executable, "-m", "streamlit",
                "run", "src/ui/streamlit_app.py",
                "--server.port", str(self.port),
                "--server.address", self.host,
                "--server.headless", "true",
                "--browser.serverAddress", self.host,
                "--logger.level", "info"
            ]
            
            # Start the Streamlit process
            self.ui_process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for UI to be ready
            if self._wait_for_ui_ready():
                logger.info(f"Streamlit UI started successfully on http://{self.host}:{self.port}")
                return True
            else:
                logger.error("Failed to start Streamlit UI")
                return False
                
        except Exception as e:
            logger.error(f"Error starting UI server: {str(e)}")
            return False
    
    def _wait_for_ui_ready(self, timeout: int = 30) -> bool:
        """Wait for UI server to be ready by checking if port is listening."""
        import socket
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(1)
                    result = sock.connect_ex((self.host, self.port))
                    if result == 0:
                        return True
            except Exception:
                pass
            
            # Check if process is still running
            if self.ui_process and self.ui_process.poll() is not None:
                logger.error("UI process terminated unexpectedly")
                return False
            
            time.sleep(1)
        
        logger.error(f"UI readiness check timeout after {timeout} seconds")
        return False
    
    def stop_server(self):
        """Stop the UI server gracefully."""
        if self.ui_process and self.ui_process.poll() is None:
            logger.info("Stopping Streamlit UI...")
            self.ui_process.terminate()
            try:
                self.ui_process.wait(timeout=10)
                logger.info("Streamlit UI stopped")
            except subprocess.TimeoutExpired:
                logger.warning("Streamlit UI did not stop gracefully, forcing termination...")
                self.ui_process.kill()
    
    def monitor_server(self):
        """Monitor the running UI server and restart if needed."""
        while not self.shutdown_event:
            # Check UI process
            if self.ui_process and self.ui_process.poll() is not None:
                logger.warning("UI process terminated, attempting restart...")
                if not self.start_server():
                    logger.error("Failed to restart UI server")
                    break
            
            # Periodically check API availability
            if time.time() % 30 < 1:  # Check every ~30 seconds
                self.check_api_availability()
            
            time.sleep(5)
    
    def get_server_status(self) -> dict:
        """Get current status of the UI server."""
        status = {
            "running": self.ui_process and self.ui_process.poll() is None,
            "host": self.host,
            "port": self.port,
            "api_url": self.api_url,
            "api_available": self.check_api_availability()
        }
        
        # Check UI health (simplified - just check if port is open)
        if status["running"]:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex((self.host, self.port))
                status["health"] = "healthy" if result == 0 else "unreachable"
        else:
            status["health"] = "stopped"
        
        return status


def parse_arguments():
    """Parse command line arguments for the UI server."""
    parser = argparse.ArgumentParser(
        description="AI Customer Agent - UI Server Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_ui.py                    # Run UI on default port 8501
  python run_ui.py --port 8502        # Run UI on port 8502
  python run_ui.py --host 0.0.0.0     # Run UI on all interfaces
  python run_ui.py --api-url http://localhost:8080  # Connect to custom API URL
        """
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to bind the server to (default: localhost)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port to run the server on (default: 8501)"
    )
    
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8001",
        help="URL of the API server (default: http://localhost:8001)"
    )
    
    parser.add_argument(
        "--no-monitor",
        action="store_true",
        help="Disable server monitoring and auto-restart"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Logging level (default: info)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the UI server."""
    args = parse_arguments()
    
    # Set log level based on arguments
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    # Create UI server instance
    ui_server = UIServer(
        host=args.host,
        port=args.port,
        api_url=args.api_url
    )
    
    # Print startup information
    ui_server.print_startup_info()
    
    # Validate configuration
    if not ui_server.validate_configuration():
        logger.error("Configuration validation failed. Exiting.")
        sys.exit(1)
    
    try:
        # Start the server
        if not ui_server.start_server():
            logger.error("Failed to start UI server")
            sys.exit(1)
        
        logger.info("UI server is running and ready to accept requests")
        
        # Monitor server unless disabled
        if not args.no_monitor:
            logger.info("Server monitoring started (auto-restart enabled)")
            ui_server.monitor_server()
        else:
            logger.info("Server monitoring disabled")
            # Keep main thread alive until shutdown signal
            while not ui_server.shutdown_event:
                time.sleep(1)
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
    finally:
        # Always stop server on exit
        ui_server.stop_server()
        logger.info("UI server shutdown complete")


if __name__ == "__main__":
    main()
