#!/usr/bin/env python3
"""
Main Application Entry Point for AI Customer Agent.

This script provides the main entry point for running the AI Customer Agent
with both FastAPI backend and Streamlit UI. It manages service startup,
health monitoring, and graceful shutdown.

Usage:
    python main.py [--api-only] [--ui-only] [--port API_PORT] [--ui-port UI_PORT]

Examples:
    python main.py                    # Run both API and UI
    python main.py --api-only         # Run only the API server
    python main.py --ui-only          # Run only the Streamlit UI
    python main.py --port 8080 --ui-port 8502  # Run on custom ports
"""

import argparse
import logging
import signal
import sys
import time
import threading
import subprocess
import os
from typing import Optional, Dict, Any
import requests
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/app.log')
    ]
)
logger = logging.getLogger(__name__)


class ServiceManager:
    """Manages the lifecycle of API and UI services."""
    
    def __init__(self, api_port: int = 8001, ui_port: int = 8501):
        self.api_port = api_port
        self.ui_port = ui_port
        self.api_process: Optional[subprocess.Popen] = None
        self.ui_process: Optional[subprocess.Popen] = None
        self.shutdown_event = threading.Event()
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_event.set()
        self.stop_services()
    
    def start_api_service(self) -> bool:
        """Start the FastAPI backend service."""
        try:
            logger.info(f"Starting FastAPI backend on port {self.api_port}...")
            
            # Use uvicorn to run the FastAPI app
            self.api_process = subprocess.Popen([
                sys.executable, "-m", "uvicorn",
                "src.api.main:app",
                "--host", "0.0.0.0",
                "--port", str(self.api_port),
                "--reload",  # Enable auto-reload for development
                "--log-level", "info"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait for API to be ready
            if self._wait_for_api_ready():
                logger.info(f"FastAPI backend started successfully on http://localhost:{self.api_port}")
                return True
            else:
                logger.error("Failed to start FastAPI backend")
                return False
                
        except Exception as e:
            logger.error(f"Error starting API service: {str(e)}")
            return False
    
    def start_ui_service(self) -> bool:
        """Start the Streamlit UI service."""
        try:
            logger.info(f"Starting Streamlit UI on port {self.ui_port}...")
            
            # Set environment variable for Streamlit port
            env = os.environ.copy()
            env["STREAMLIT_SERVER_PORT"] = str(self.ui_port)
            
            # Use streamlit to run the UI app
            self.ui_process = subprocess.Popen([
                sys.executable, "-m", "streamlit",
                "run", "src/ui/streamlit_app.py",
                "--server.port", str(self.ui_port),
                "--server.headless", "true",
                "--browser.serverAddress", "localhost",
                "--logger.level", "info"
            ], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait for UI to be ready
            if self._wait_for_ui_ready():
                logger.info(f"Streamlit UI started successfully on http://localhost:{self.ui_port}")
                return True
            else:
                logger.error("Failed to start Streamlit UI")
                return False
                
        except Exception as e:
            logger.error(f"Error starting UI service: {str(e)}")
            return False
    
    def _wait_for_api_ready(self, timeout: int = 30) -> bool:
        """Wait for API service to be ready by checking health endpoint."""
        health_url = f"http://localhost:{self.api_port}/health"
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(health_url, timeout=5)
                if response.status_code == 200:
                    health_data = response.json()
                    if health_data.get("status") in ["healthy", "degraded"]:
                        return True
            except requests.exceptions.RequestException:
                pass
            
            # Check if process is still running
            if self.api_process and self.api_process.poll() is not None:
                logger.error("API process terminated unexpectedly")
                return False
            
            time.sleep(1)
        
        logger.error(f"API health check timeout after {timeout} seconds")
        return False
    
    def _wait_for_ui_ready(self, timeout: int = 30) -> bool:
        """Wait for UI service to be ready by checking if port is listening."""
        import socket
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(1)
                    result = sock.connect_ex(('localhost', self.ui_port))
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
    
    def stop_services(self):
        """Stop all running services gracefully."""
        logger.info("Stopping services...")
        
        # Stop UI service
        if self.ui_process and self.ui_process.poll() is None:
            logger.info("Stopping Streamlit UI...")
            self.ui_process.terminate()
            try:
                self.ui_process.wait(timeout=10)
                logger.info("Streamlit UI stopped")
            except subprocess.TimeoutExpired:
                logger.warning("Streamlit UI did not stop gracefully, forcing termination...")
                self.ui_process.kill()
        
        # Stop API service
        if self.api_process and self.api_process.poll() is None:
            logger.info("Stopping FastAPI backend...")
            self.api_process.terminate()
            try:
                self.api_process.wait(timeout=10)
                logger.info("FastAPI backend stopped")
            except subprocess.TimeoutExpired:
                logger.warning("FastAPI backend did not stop gracefully, forcing termination...")
                self.api_process.kill()
    
    def monitor_services(self):
        """Monitor running services and restart if needed."""
        while not self.shutdown_event.is_set():
            # Check API process
            if self.api_process and self.api_process.poll() is not None:
                logger.warning("API process terminated, attempting restart...")
                if not self.start_api_service():
                    logger.error("Failed to restart API service")
                    break
            
            # Check UI process
            if self.ui_process and self.ui_process.poll() is not None:
                logger.warning("UI process terminated, attempting restart...")
                if not self.start_ui_service():
                    logger.error("Failed to restart UI service")
                    break
            
            time.sleep(5)
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get current status of all services."""
        status = {
            "api": {
                "running": self.api_process and self.api_process.poll() is None,
                "port": self.api_port,
                "health": "unknown"
            },
            "ui": {
                "running": self.ui_process and self.ui_process.poll() is None,
                "port": self.ui_port,
                "health": "unknown"
            }
        }
        
        # Check API health
        if status["api"]["running"]:
            try:
                response = requests.get(f"http://localhost:{self.api_port}/health", timeout=5)
                if response.status_code == 200:
                    status["api"]["health"] = response.json().get("status", "unknown")
            except requests.exceptions.RequestException:
                status["api"]["health"] = "unreachable"
        
        # Check UI health (simplified - just check if port is open)
        if status["ui"]["running"]:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', self.ui_port))
                status["ui"]["health"] = "healthy" if result == 0 else "unreachable"
        
        return status


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AI Customer Agent - Main Application Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run both API and UI
  python main.py --api-only         # Run only the API server
  python main.py --ui-only          # Run only the Streamlit UI
  python main.py --port 8080 --ui-port 8502  # Run on custom ports
        """
    )
    
    parser.add_argument(
        "--api-only",
        action="store_true",
        help="Run only the FastAPI backend service"
    )
    
    parser.add_argument(
        "--ui-only", 
        action="store_true",
        help="Run only the Streamlit UI service"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port for the FastAPI backend (default: 8001)"
    )
    
    parser.add_argument(
        "--ui-port",
        type=int,
        default=8501,
        help="Port for the Streamlit UI (default: 8501)"
    )
    
    parser.add_argument(
        "--no-monitor",
        action="store_true",
        help="Disable service monitoring and auto-restart"
    )
    
    return parser.parse_args()


def print_startup_info(service_manager: ServiceManager, args):
    """Print startup information and URLs."""
    print("\n" + "="*60)
    print("AI Customer Agent - Starting Services")
    print("="*60)
    
    if not args.api_only and not args.ui_only:
        print("Starting both API and UI services...")
    elif args.api_only:
        print("Starting API service only...")
    elif args.ui_only:
        print("Starting UI service only...")
    
    print(f"\nService Configuration:")
    if not args.ui_only:
        print(f"   - FastAPI Backend: http://localhost:{service_manager.api_port}")
        print(f"   - API Documentation: http://localhost:{service_manager.api_port}/docs")
        print(f"   - Health Check: http://localhost:{service_manager.api_port}/health")
    
    if not args.api_only:
        print(f"   - Streamlit UI: http://localhost:{service_manager.ui_port}")
    
    print(f"\nAdditional Options:")
    print(f"   - Service Monitoring: {'Disabled' if args.no_monitor else 'Enabled'}")
    print(f"   - Logs Directory: ./logs/")
    
    print("\nStarting services... (Press Ctrl+C to stop)")
    print("="*60 + "\n")


def main():
    """Main application entry point."""
    args = parse_arguments()
    
    # Validate arguments
    if args.api_only and args.ui_only:
        logger.error("Cannot specify both --api-only and --ui-only")
        sys.exit(1)
    
    # Create service manager
    service_manager = ServiceManager(
        api_port=args.port,
        ui_port=args.ui_port
    )
    
    # Print startup information
    print_startup_info(service_manager, args)
    
    # Start services based on arguments
    services_started = []
    
    try:
        # Start API service if needed
        if not args.ui_only:
            if service_manager.start_api_service():
                services_started.append("api")
            else:
                logger.error("Failed to start API service")
                if not args.api_only:
                    sys.exit(1)
        
        # Start UI service if needed
        if not args.api_only:
            if service_manager.start_ui_service():
                services_started.append("ui")
            else:
                logger.error("Failed to start UI service")
                if not args.ui_only:
                    sys.exit(1)
        
        # Check if any services started successfully
        if not services_started:
            logger.error("No services started successfully")
            sys.exit(1)
        
        logger.info(f"Successfully started services: {', '.join(services_started)}")
        
        # Monitor services unless disabled
        if not args.no_monitor:
            logger.info("Service monitoring started (auto-restart enabled)")
            service_manager.monitor_services()
        else:
            logger.info("Service monitoring disabled")
            # Keep main thread alive until shutdown signal
            while not service_manager.shutdown_event.is_set():
                time.sleep(1)
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
    finally:
        # Always stop services on exit
        service_manager.stop_services()
        logger.info("Application shutdown complete")


if __name__ == "__main__":
    main()
