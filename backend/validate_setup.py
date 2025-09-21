#!/usr/bin/env python3
"""
Setup Validation Script for Chatbot API
Validates hardware optimization and GPU acceleration
"""

import subprocess
import sys
import time
import json
import requests
from pathlib import Path


class SetupValidator:
    """Validates the complete chatbot API setup"""
    
    def __init__(self):
        self.results = {}
        self.errors = []
    
    def check_system_requirements(self):
        """Check system requirements"""
        print("🔍 Checking system requirements...")
        
        # Check Docker
        try:
            result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  ✅ Docker: {result.stdout.strip()}")
                self.results['docker'] = True
            else:
                print("  ❌ Docker not found")
                self.results['docker'] = False
                self.errors.append("Docker is not installed")
        except FileNotFoundError:
            print("  ❌ Docker not found")
            self.results['docker'] = False
            self.errors.append("Docker is not installed")
        
        # Check Docker Compose
        try:
            result = subprocess.run(['docker-compose', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  ✅ Docker Compose: {result.stdout.strip()}")
                self.results['docker_compose'] = True
            else:
                print("  ❌ Docker Compose not found")
                self.results['docker_compose'] = False
                self.errors.append("Docker Compose is not installed")
        except FileNotFoundError:
            print("  ❌ Docker Compose not found")
            self.results['docker_compose'] = False
            self.errors.append("Docker Compose is not installed")
        
        # Check NVIDIA Docker support
        try:
            result = subprocess.run(['docker', 'run', '--rm', '--gpus', 'all', 
                                   'nvidia/cuda:12.1-base-ubuntu22.04', 'nvidia-smi'], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print("  ✅ NVIDIA Docker support available")
                self.results['nvidia_docker'] = True
            else:
                print("  ⚠️  NVIDIA Docker support not available")
                self.results['nvidia_docker'] = False
                self.errors.append("NVIDIA Docker support not available")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print("  ⚠️  Could not test NVIDIA Docker support")
            self.results['nvidia_docker'] = False
    
    def check_hardware_optimization(self):
        """Check hardware optimization settings"""
        print("\n⚡ Checking hardware optimization...")
        
        # Check CPU governor
        try:
            with open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor', 'r') as f:
                governor = f.read().strip()
                if governor == 'performance':
                    print(f"  ✅ CPU Governor: {governor}")
                    self.results['cpu_governor'] = True
                else:
                    print(f"  ⚠️  CPU Governor: {governor} (should be 'performance')")
                    self.results['cpu_governor'] = False
                    self.errors.append(f"CPU governor is '{governor}', should be 'performance'")
        except FileNotFoundError:
            print("  ❌ Could not check CPU governor")
            self.results['cpu_governor'] = False
        
        # Check swappiness
        try:
            with open('/proc/sys/vm/swappiness', 'r') as f:
                swappiness = int(f.read().strip())
                if swappiness <= 10:
                    print(f"  ✅ Swappiness: {swappiness}")
                    self.results['swappiness'] = True
                else:
                    print(f"  ⚠️  Swappiness: {swappiness} (should be ≤10)")
                    self.results['swappiness'] = False
                    self.errors.append(f"Swappiness is {swappiness}, should be ≤10")
        except FileNotFoundError:
            print("  ❌ Could not check swappiness")
            self.results['swappiness'] = False
        
        # Check GPU availability
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', 
                                   '--format=csv,noheader'], capture_output=True, text=True)
            if result.returncode == 0:
                gpu_info = result.stdout.strip()
                print(f"  ✅ GPU: {gpu_info}")
                self.results['gpu_available'] = True
            else:
                print("  ❌ GPU not detected")
                self.results['gpu_available'] = False
                self.errors.append("GPU not detected")
        except FileNotFoundError:
            print("  ❌ nvidia-smi not found")
            self.results['gpu_available'] = False
            self.errors.append("nvidia-smi not found")
    
    def check_project_structure(self):
        """Check project structure and files"""
        print("\n📁 Checking project structure...")
        
        required_files = [
            'Dockerfile',
            'docker-compose.yml',
            'requirements.txt',
            'app/main.py',
            'app/config.py',
            'app/models.py',
            'app/chatbot_service.py',
            '.env.example'
        ]
        
        for file_path in required_files:
            if Path(file_path).exists():
                print(f"  ✅ {file_path}")
            else:
                print(f"  ❌ {file_path} missing")
                self.errors.append(f"Required file {file_path} is missing")
        
        self.results['project_structure'] = len(self.errors) == 0
    
    def build_and_test_container(self):
        """Build and test the Docker container"""
        print("\n🐳 Building Docker container...")
        
        try:
            # Build the container
            result = subprocess.run(['docker-compose', 'build'], 
                                  capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print("  ✅ Container built successfully")
                self.results['container_build'] = True
            else:
                print("  ❌ Container build failed")
                print(f"  Error: {result.stderr}")
                self.results['container_build'] = False
                self.errors.append("Container build failed")
                return False
        except subprocess.TimeoutExpired:
            print("  ❌ Container build timed out")
            self.results['container_build'] = False
            self.errors.append("Container build timed out")
            return False
        
        return True
    
    def test_api_functionality(self):
        """Test API functionality"""
        print("\n🚀 Testing API functionality...")
        
        # Start the container
        print("  Starting container...")
        try:
            subprocess.run(['docker-compose', 'up', '-d'], 
                          capture_output=True, text=True, timeout=60)
            
            # Wait for startup
            print("  Waiting for API to start...")
            time.sleep(30)
            
            # Test health endpoint
            try:
                response = requests.get('http://localhost:8000/health', timeout=10)
                if response.status_code == 200:
                    print("  ✅ Health endpoint working")
                    health_data = response.json()
                    print(f"    Model loaded: {health_data.get('model_loaded', False)}")
                    print(f"    GPU available: {health_data.get('gpu_available', False)}")
                    self.results['health_endpoint'] = True
                else:
                    print(f"  ❌ Health endpoint failed: {response.status_code}")
                    self.results['health_endpoint'] = False
                    self.errors.append(f"Health endpoint returned {response.status_code}")
            except requests.RequestException as e:
                print(f"  ❌ Health endpoint error: {e}")
                self.results['health_endpoint'] = False
                self.errors.append(f"Health endpoint error: {e}")
            
            # Test chat endpoint
            try:
                chat_payload = {"message": "Hello, this is a test"}
                response = requests.post('http://localhost:8000/chat', 
                                       json=chat_payload, timeout=30)
                if response.status_code == 200:
                    print("  ✅ Chat endpoint working")
                    chat_data = response.json()
                    print(f"    Response time: {chat_data.get('processing_time', 0):.2f}s")
                    print(f"    GPU used: {chat_data.get('gpu_used', False)}")
                    self.results['chat_endpoint'] = True
                    
                    # Check response time
                    if chat_data.get('processing_time', 999) < 5.0:
                        print("  ✅ Response time acceptable")
                        self.results['response_time'] = True
                    else:
                        print("  ⚠️  Response time slow")
                        self.results['response_time'] = False
                        self.errors.append("Response time > 5 seconds")
                else:
                    print(f"  ❌ Chat endpoint failed: {response.status_code}")
                    self.results['chat_endpoint'] = False
                    self.errors.append(f"Chat endpoint returned {response.status_code}")
            except requests.RequestException as e:
                print(f"  ❌ Chat endpoint error: {e}")
                self.results['chat_endpoint'] = False
                self.errors.append(f"Chat endpoint error: {e}")
            
        except subprocess.TimeoutExpired:
            print("  ❌ Container startup timed out")
            self.results['api_test'] = False
            self.errors.append("Container startup timed out")
        
        finally:
            # Stop the container
            print("  Stopping container...")
            subprocess.run(['docker-compose', 'down'], 
                          capture_output=True, text=True, timeout=30)
    
    def run_validation(self):
        """Run complete validation"""
        print("🔧 Chatbot API Setup Validation")
        print("=" * 50)
        
        self.check_system_requirements()
        self.check_hardware_optimization()
        self.check_project_structure()
        
        if self.results.get('docker') and self.results.get('docker_compose'):
            if self.build_and_test_container():
                self.test_api_functionality()
        else:
            print("\n⚠️  Skipping container tests due to missing Docker")
        
        self.print_summary()
    
    def print_summary(self):
        """Print validation summary"""
        print("\n" + "=" * 50)
        print("VALIDATION SUMMARY")
        print("=" * 50)
        
        total_checks = len(self.results)
        passed_checks = sum(1 for v in self.results.values() if v)
        
        print(f"\nChecks passed: {passed_checks}/{total_checks}")
        
        if self.errors:
            print(f"\n❌ Issues found ({len(self.errors)}):")
            for error in self.errors:
                print(f"  • {error}")
        else:
            print("\n✅ All checks passed!")
        
        # Recommendations
        print("\n💡 Recommendations:")
        if not self.results.get('cpu_governor'):
            print("  • Set CPU governor to performance: sudo cpupower frequency-set -g performance")
        if not self.results.get('swappiness'):
            print("  • Reduce swappiness: echo 10 | sudo tee /proc/sys/vm/swappiness")
        if not self.results.get('nvidia_docker'):
            print("  • Install NVIDIA Container Toolkit for GPU acceleration")
        
        print(f"\n📊 Overall Status: {'✅ READY' if len(self.errors) == 0 else '⚠️  NEEDS ATTENTION'}")


if __name__ == "__main__":
    validator = SetupValidator()
    validator.run_validation()
