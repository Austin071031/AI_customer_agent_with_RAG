#!/usr/bin/env python3
"""
Unit Tests for AI Customer Agent Entry Points.

This module contains unit tests for the main application entry points:
- main.py (combined application)
- run_api.py (API server only)  
- run_ui.py (UI server only)

The tests focus on configuration validation, argument parsing, and service
management without actually starting the servers.
"""

import unittest
import sys
import os
import tempfile
import argparse
from unittest.mock import patch, MagicMock, Mock
import logging

# Add the project root to Python path to allow imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import the entry point modules
try:
    import main
    import run_api
    import run_ui
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running tests from the project root directory")


class TestMainEntryPoint(unittest.TestCase):
    """Test cases for main.py entry point."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.original_argv = sys.argv
        sys.argv = ['main.py']  # Default arguments
        
    def tearDown(self):
        """Clean up test fixtures."""
        sys.argv = self.original_argv
    
    def test_parse_arguments_default(self):
        """Test argument parsing with default values."""
        with patch('sys.argv', ['main.py']):
            args = main.parse_arguments()
            
            self.assertFalse(args.api_only)
            self.assertFalse(args.ui_only)
            self.assertEqual(args.port, 8000)
            self.assertEqual(args.ui_port, 8501)
            self.assertFalse(args.no_monitor)
    
    def test_parse_arguments_api_only(self):
        """Test argument parsing with --api-only flag."""
        with patch('sys.argv', ['main.py', '--api-only']):
            args = main.parse_arguments()
            
            self.assertTrue(args.api_only)
            self.assertFalse(args.ui_only)
    
    def test_parse_arguments_ui_only(self):
        """Test argument parsing with --ui-only flag."""
        with patch('sys.argv', ['main.py', '--ui-only']):
            args = main.parse_arguments()
            
            self.assertFalse(args.api_only)
            self.assertTrue(args.ui_only)
    
    def test_parse_arguments_custom_ports(self):
        """Test argument parsing with custom ports."""
        with patch('sys.argv', ['main.py', '--port', '8080', '--ui-port', '8502']):
            args = main.parse_arguments()
            
            self.assertEqual(args.port, 8080)
            self.assertEqual(args.ui_port, 8502)
    
    def test_parse_arguments_no_monitor(self):
        """Test argument parsing with --no-monitor flag."""
        with patch('sys.argv', ['main.py', '--no-monitor']):
            args = main.parse_arguments()
            
            self.assertTrue(args.no_monitor)
    
    def test_service_manager_initialization(self):
        """Test ServiceManager initialization with default values."""
        service_manager = main.ServiceManager()
        
        self.assertEqual(service_manager.api_port, 8000)
        self.assertEqual(service_manager.ui_port, 8501)
        self.assertIsNone(service_manager.api_process)
        self.assertIsNone(service_manager.ui_process)
        self.assertFalse(service_manager.shutdown_event.is_set())
    
    def test_service_manager_custom_ports(self):
        """Test ServiceManager initialization with custom ports."""
        service_manager = main.ServiceManager(api_port=8080, ui_port=8502)
        
        self.assertEqual(service_manager.api_port, 8080)
        self.assertEqual(service_manager.ui_port, 8502)
    
    @patch('main.subprocess.Popen')
    @patch('main.requests.get')
    def test_start_api_service_success(self, mock_requests_get, mock_popen):
        """Test successful API service startup."""
        # Mock the subprocess
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process
        
        # Mock the health check response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy"}
        mock_requests_get.return_value = mock_response
        
        service_manager = main.ServiceManager()
        result = service_manager.start_api_service()
        
        self.assertTrue(result)
        mock_popen.assert_called_once()
        mock_requests_get.assert_called()
    
    @patch('main.subprocess.Popen')
    def test_start_api_service_failure(self, mock_popen):
        """Test API service startup failure."""
        # Mock subprocess to raise an exception
        mock_popen.side_effect = Exception("Failed to start")
        
        service_manager = main.ServiceManager()
        result = service_manager.start_api_service()
        
        self.assertFalse(result)
    
    @patch('main.subprocess.Popen')
    def test_start_ui_service_success(self, mock_popen):
        """Test successful UI service startup."""
        # Mock the subprocess
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process
        
        # Mock socket module at system level
        with patch('socket.socket') as mock_socket:
            # Mock socket connection success
            mock_sock_instance = MagicMock()
            mock_sock_instance.connect_ex.return_value = 0
            mock_socket.return_value.__enter__.return_value = mock_sock_instance
            
            service_manager = main.ServiceManager()
            result = service_manager.start_ui_service()
            
            self.assertTrue(result)
            mock_popen.assert_called_once()
    
    def test_stop_services_no_processes(self):
        """Test stopping services when no processes are running."""
        service_manager = main.ServiceManager()
        
        # Should not raise any exceptions
        service_manager.stop_services()
    
    @patch('main.requests.get')
    def test_get_service_status(self, mock_requests_get):
        """Test getting service status."""
        # Mock API health check response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy"}
        mock_requests_get.return_value = mock_response
        
        service_manager = main.ServiceManager()
        status = service_manager.get_service_status()
        
        self.assertIn('api', status)
        self.assertIn('ui', status)
        self.assertEqual(status['api']['port'], 8000)
        self.assertEqual(status['ui']['port'], 8501)


class TestAPIEntryPoint(unittest.TestCase):
    """Test cases for run_api.py entry point."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.original_argv = sys.argv
        sys.argv = ['run_api.py']  # Default arguments
    
    def tearDown(self):
        """Clean up test fixtures."""
        sys.argv = self.original_argv
    
    def test_parse_arguments_default(self):
        """Test argument parsing with default values."""
        with patch('sys.argv', ['run_api.py']):
            args = run_api.parse_arguments()
            
            self.assertEqual(args.host, "0.0.0.0")
            self.assertEqual(args.port, 8000)
            self.assertFalse(args.no_reload)
            self.assertEqual(args.workers, 1)
            self.assertEqual(args.log_level, "info")
    
    def test_parse_arguments_custom_values(self):
        """Test argument parsing with custom values."""
        with patch('sys.argv', [
            'run_api.py', 
            '--host', '127.0.0.1',
            '--port', '8080',
            '--no-reload',
            '--workers', '4',
            '--log-level', 'debug'
        ]):
            args = run_api.parse_arguments()
            
            self.assertEqual(args.host, "127.0.0.1")
            self.assertEqual(args.port, 8080)
            self.assertTrue(args.no_reload)
            self.assertEqual(args.workers, 4)
            self.assertEqual(args.log_level, "debug")
    
    def test_api_server_initialization(self):
        """Test APIServer initialization with default values."""
        api_server = run_api.APIServer()
        
        self.assertEqual(api_server.host, "0.0.0.0")
        self.assertEqual(api_server.port, 8000)
        self.assertTrue(api_server.reload)
        self.assertEqual(api_server.workers, 1)
        self.assertIsNone(api_server.server)
        self.assertFalse(api_server.shutdown_event)
    
    def test_api_server_custom_values(self):
        """Test APIServer initialization with custom values."""
        api_server = run_api.APIServer(
            host="127.0.0.1",
            port=8080,
            reload=False,
            workers=4
        )
        
        self.assertEqual(api_server.host, "127.0.0.1")
        self.assertEqual(api_server.port, 8080)
        self.assertFalse(api_server.reload)
        self.assertEqual(api_server.workers, 4)
    
    def test_validate_configuration_success(self):
        """Test successful configuration validation."""
        api_server = run_api.APIServer()
        
        # Mock socket module
        with patch('run_api.socket') as mock_socket:
            # Mock socket to show port is available
            mock_sock_instance = MagicMock()
            mock_sock_instance.connect_ex.return_value = 1  # Port not in use
            mock_socket.socket.return_value.__enter__.return_value = mock_sock_instance
            
            # Mock importlib
            with patch('run_api.importlib') as mock_importlib:
                mock_importlib.import_module.return_value = True
                
                result = api_server.validate_configuration()
                self.assertTrue(result)
    
    def test_validate_configuration_port_in_use(self):
        """Test configuration validation when port is in use."""
        api_server = run_api.APIServer()
        
        # Mock socket module
        with patch('run_api.socket') as mock_socket:
            # Mock socket to show port is in use
            mock_sock_instance = MagicMock()
            mock_sock_instance.connect_ex.return_value = 0  # Port in use
            mock_socket.socket.return_value.__enter__.return_value = mock_sock_instance
            
            result = api_server.validate_configuration()
            self.assertFalse(result)
    
    @patch('run_api.requests.get')
    def test_health_check_success(self, mock_requests_get):
        """Test successful health check."""
        # Mock successful health check response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_requests_get.return_value = mock_response
        
        api_server = run_api.APIServer()
        result = api_server.health_check()
        
        self.assertTrue(result)
        mock_requests_get.assert_called_with("http://localhost:8000/health", timeout=5)
    
    @patch('run_api.requests.get')
    def test_health_check_failure(self, mock_requests_get):
        """Test health check failure."""
        # Mock failed health check (connection error)
        mock_requests_get.side_effect = Exception("Connection failed")
        
        api_server = run_api.APIServer()
        result = api_server.health_check()
        
        self.assertFalse(result)


class TestUIEntryPoint(unittest.TestCase):
    """Test cases for run_ui.py entry point."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.original_argv = sys.argv
        sys.argv = ['run_ui.py']  # Default arguments
    
    def tearDown(self):
        """Clean up test fixtures."""
        sys.argv = self.original_argv
    
    def test_parse_arguments_default(self):
        """Test argument parsing with default values."""
        with patch('sys.argv', ['run_ui.py']):
            args = run_ui.parse_arguments()
            
            self.assertEqual(args.host, "localhost")
            self.assertEqual(args.port, 8501)
            self.assertEqual(args.api_url, "http://localhost:8000")
            self.assertFalse(args.no_monitor)
            self.assertEqual(args.log_level, "info")
    
    def test_parse_arguments_custom_values(self):
        """Test argument parsing with custom values."""
        with patch('sys.argv', [
            'run_ui.py',
            '--host', '0.0.0.0',
            '--port', '8502',
            '--api-url', 'http://localhost:8080',
            '--no-monitor',
            '--log-level', 'warning'
        ]):
            args = run_ui.parse_arguments()
            
            self.assertEqual(args.host, "0.0.0.0")
            self.assertEqual(args.port, 8502)
            self.assertEqual(args.api_url, "http://localhost:8080")
            self.assertTrue(args.no_monitor)
            self.assertEqual(args.log_level, "warning")
    
    def test_ui_server_initialization(self):
        """Test UIServer initialization with default values."""
        ui_server = run_ui.UIServer()
        
        self.assertEqual(ui_server.host, "localhost")
        self.assertEqual(ui_server.port, 8501)
        self.assertEqual(ui_server.api_url, "http://localhost:8000")
        self.assertIsNone(ui_server.ui_process)
        self.assertFalse(ui_server.shutdown_event)
    
    def test_ui_server_custom_values(self):
        """Test UIServer initialization with custom values."""
        ui_server = run_ui.UIServer(
            host="0.0.0.0",
            port=8502,
            api_url="http://localhost:8080"
        )
        
        self.assertEqual(ui_server.host, "0.0.0.0")
        self.assertEqual(ui_server.port, 8502)
        self.assertEqual(ui_server.api_url, "http://localhost:8080")
    
    @patch('run_ui.requests.get')
    def test_check_api_availability_success(self, mock_requests_get):
        """Test successful API availability check."""
        # Mock successful health check response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy"}
        mock_requests_get.return_value = mock_response
        
        ui_server = run_ui.UIServer()
        result = ui_server.check_api_availability()
        
        self.assertTrue(result)
        mock_requests_get.assert_called_with("http://localhost:8000/health", timeout=5)
    
    @patch('run_ui.requests.get')
    def test_check_api_availability_degraded(self, mock_requests_get):
        """Test API availability check with degraded status."""
        # Mock degraded health check response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "degraded"}
        mock_requests_get.return_value = mock_response
        
        ui_server = run_ui.UIServer()
        result = ui_server.check_api_availability()
        
        self.assertTrue(result)  # Should still return True for degraded
    
    @patch('run_ui.requests.get')
    def test_check_api_availability_failure(self, mock_requests_get):
        """Test API availability check failure."""
        # Mock connection error
        mock_requests_get.side_effect = Exception("Connection failed")
        
        ui_server = run_ui.UIServer()
        result = ui_server.check_api_availability()
        
        self.assertFalse(result)
    
    def test_validate_configuration_success(self):
        """Test successful configuration validation."""
        ui_server = run_ui.UIServer()
        
        # Mock socket module
        with patch('run_ui.socket') as mock_socket:
            # Mock socket to show port is available
            mock_sock_instance = MagicMock()
            mock_sock_instance.connect_ex.return_value = 1  # Port not in use
            mock_socket.socket.return_value.__enter__.return_value = mock_sock_instance
            
            # Mock importlib
            with patch('run_ui.importlib') as mock_importlib:
                mock_importlib.import_module.return_value = True
                
                # Mock os.path.exists
                with patch('run_ui.os.path.exists') as mock_exists:
                    mock_exists.return_value = True
                    
                    result = ui_server.validate_configuration()
                    self.assertTrue(result)
    
    def test_validate_configuration_port_in_use(self):
        """Test configuration validation when port is in use."""
        ui_server = run_ui.UIServer()
        
        # Mock socket module
        with patch('run_ui.socket') as mock_socket:
            # Mock socket to show port is in use
            mock_sock_instance = MagicMock()
            mock_sock_instance.connect_ex.return_value = 0  # Port in use
            mock_socket.socket.return_value.__enter__.return_value = mock_sock_instance
            
            result = ui_server.validate_configuration()
            self.assertFalse(result)
    
    def test_get_server_status(self):
        """Test getting server status."""
        ui_server = run_ui.UIServer()
        status = ui_server.get_server_status()
        
        self.assertIn('running', status)
        self.assertIn('host', status)
        self.assertIn('port', status)
        self.assertIn('api_url', status)
        self.assertIn('api_available', status)
        self.assertIn('health', status)


class TestIntegration(unittest.TestCase):
    """Integration tests for entry point interactions."""
    
    def test_main_argument_conflict(self):
        """Test that --api-only and --ui-only cannot be used together."""
        with patch('sys.argv', ['main.py', '--api-only', '--ui-only']):
            with self.assertRaises(SystemExit):
                main.main()
    
    @patch('main.ServiceManager.start_api_service')
    @patch('main.ServiceManager.start_ui_service')
    def test_main_api_only_mode(self, mock_start_ui, mock_start_api):
        """Test main.py in API-only mode."""
        mock_start_api.return_value = True
        
        with patch('sys.argv', ['main.py', '--api-only']):
            with patch('main.ServiceManager.monitor_services'):
                with patch('main.ServiceManager.stop_services'):
                    with self.assertRaises(SystemExit):  # Will exit due to mock
                        main.main()
        
        mock_start_api.assert_called_once()
        mock_start_ui.assert_not_called()
    
    @patch('main.ServiceManager.start_api_service')
    @patch('main.ServiceManager.start_ui_service')
    def test_main_ui_only_mode(self, mock_start_ui, mock_start_api):
        """Test main.py in UI-only mode."""
        mock_start_ui.return_value = True
        
        with patch('sys.argv', ['main.py', '--ui-only']):
            with patch('main.ServiceManager.monitor_services'):
                with patch('main.ServiceManager.stop_services'):
                    with self.assertRaises(SystemExit):  # Will exit due to mock
                        main.main()
        
        mock_start_api.assert_not_called()
        mock_start_ui.assert_called_once()


if __name__ == '__main__':
    # Configure logging for tests
    logging.basicConfig(level=logging.CRITICAL)  # Suppress log output during tests
    
    # Run the tests
    unittest.main(verbosity=2)
