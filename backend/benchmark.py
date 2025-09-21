#!/usr/bin/env python3
"""
Performance Benchmark Script for Chatbot API
Optimized for AMD Ryzen 9 8945HX + NVIDIA RTX 5070
"""

import asyncio
import time
import statistics
import json
import argparse
from typing import List, Dict, Any
import httpx
import psutil
import GPUtil


class ChatbotBenchmark:
    """Comprehensive benchmark suite for the chatbot API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = {}
    
    async def test_single_request(self, message: str) -> Dict[str, Any]:
        """Test a single request and measure performance"""
        async with httpx.AsyncClient() as client:
            start_time = time.time()
            
            try:
                response = await client.post(
                    f"{self.base_url}/chat",
                    json={"message": message},
                    timeout=30.0
                )
                
                end_time = time.time()
                total_time = end_time - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "success": True,
                        "total_time": total_time,
                        "processing_time": data.get("processing_time", 0),
                        "gpu_used": data.get("gpu_used", False),
                        "response_length": len(data.get("response", "")),
                        "status_code": response.status_code
                    }
                else:
                    return {
                        "success": False,
                        "total_time": total_time,
                        "status_code": response.status_code,
                        "error": response.text
                    }
                    
            except Exception as e:
                return {
                    "success": False,
                    "total_time": time.time() - start_time,
                    "error": str(e)
                }
    
    async def test_concurrent_requests(self, messages: List[str], concurrency: int = 5) -> List[Dict[str, Any]]:
        """Test concurrent requests"""
        print(f"Testing {len(messages)} requests with concurrency {concurrency}...")
        
        semaphore = asyncio.Semaphore(concurrency)
        
        async def limited_request(message):
            async with semaphore:
                return await self.test_single_request(message)
        
        tasks = [limited_request(msg) for msg in messages]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = [r for r in results if isinstance(r, dict)]
        return valid_results
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        metrics = {
            "cpu_usage": psutil.cpu_percent(interval=1),
            "memory_usage": psutil.virtual_memory().percent,
            "cpu_freq": psutil.cpu_freq().current if psutil.cpu_freq() else 0,
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total / (1024**3),  # GB
        }
        
        # GPU metrics
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                metrics.update({
                    "gpu_available": True,
                    "gpu_memory_used": gpu.memoryUsed,
                    "gpu_memory_total": gpu.memoryTotal,
                    "gpu_utilization": gpu.load * 100,
                    "gpu_temperature": gpu.temperature
                })
            else:
                metrics["gpu_available"] = False
        except Exception:
            metrics["gpu_available"] = False
        
        return metrics
    
    async def run_latency_test(self) -> Dict[str, Any]:
        """Test response latency with various message types"""
        print("Running latency tests...")
        
        test_messages = [
            "Hello",
            "What is artificial intelligence?",
            "Explain machine learning in detail",
            "Tell me a story about a robot",
            "What are the benefits and drawbacks of neural networks in modern AI applications?"
        ]
        
        results = []
        for i, message in enumerate(test_messages):
            print(f"  Test {i+1}/{len(test_messages)}: {message[:50]}...")
            result = await self.test_single_request(message)
            results.append(result)
            await asyncio.sleep(1)  # Brief pause between tests
        
        successful_results = [r for r in results if r.get("success")]
        
        if successful_results:
            processing_times = [r["processing_time"] for r in successful_results]
            total_times = [r["total_time"] for r in successful_results]
            
            return {
                "test_count": len(test_messages),
                "success_count": len(successful_results),
                "success_rate": len(successful_results) / len(test_messages),
                "processing_time": {
                    "min": min(processing_times),
                    "max": max(processing_times),
                    "mean": statistics.mean(processing_times),
                    "median": statistics.median(processing_times)
                },
                "total_time": {
                    "min": min(total_times),
                    "max": max(total_times),
                    "mean": statistics.mean(total_times),
                    "median": statistics.median(total_times)
                },
                "gpu_usage": sum(1 for r in successful_results if r.get("gpu_used")) / len(successful_results)
            }
        else:
            return {"error": "No successful requests"}
    
    async def run_throughput_test(self, duration: int = 60) -> Dict[str, Any]:
        """Test throughput over a specified duration"""
        print(f"Running throughput test for {duration} seconds...")
        
        message = "What is machine learning?"
        start_time = time.time()
        end_time = start_time + duration
        
        request_count = 0
        successful_requests = 0
        total_processing_time = 0
        
        while time.time() < end_time:
            result = await self.test_single_request(message)
            request_count += 1
            
            if result.get("success"):
                successful_requests += 1
                total_processing_time += result.get("processing_time", 0)
            
            # Brief pause to prevent overwhelming
            await asyncio.sleep(0.1)
        
        actual_duration = time.time() - start_time
        
        return {
            "duration": actual_duration,
            "total_requests": request_count,
            "successful_requests": successful_requests,
            "requests_per_second": request_count / actual_duration,
            "success_rate": successful_requests / request_count if request_count > 0 else 0,
            "avg_processing_time": total_processing_time / successful_requests if successful_requests > 0 else 0
        }
    
    async def run_stress_test(self, concurrency: int = 10, requests_per_worker: int = 10) -> Dict[str, Any]:
        """Run stress test with high concurrency"""
        print(f"Running stress test: {concurrency} concurrent workers, {requests_per_worker} requests each...")
        
        messages = ["Stress test message"] * (concurrency * requests_per_worker)
        
        start_time = time.time()
        results = await self.test_concurrent_requests(messages, concurrency)
        end_time = time.time()
        
        successful_results = [r for r in results if r.get("success")]
        
        if successful_results:
            processing_times = [r["processing_time"] for r in successful_results]
            
            return {
                "total_requests": len(messages),
                "successful_requests": len(successful_results),
                "success_rate": len(successful_results) / len(messages),
                "total_duration": end_time - start_time,
                "requests_per_second": len(messages) / (end_time - start_time),
                "avg_processing_time": statistics.mean(processing_times),
                "max_processing_time": max(processing_times),
                "min_processing_time": min(processing_times)
            }
        else:
            return {"error": "No successful requests in stress test"}
    
    async def run_full_benchmark(self) -> Dict[str, Any]:
        """Run complete benchmark suite"""
        print("Starting comprehensive benchmark...")
        print("=" * 60)
        
        # Get initial system metrics
        initial_metrics = self.get_system_metrics()
        
        # Run tests
        latency_results = await self.run_latency_test()
        print()
        
        throughput_results = await self.run_throughput_test(30)  # 30 second test
        print()
        
        stress_results = await self.run_stress_test(5, 5)  # 5 workers, 5 requests each
        print()
        
        # Get final system metrics
        final_metrics = self.get_system_metrics()
        
        return {
            "timestamp": time.time(),
            "system_info": {
                "initial": initial_metrics,
                "final": final_metrics
            },
            "latency_test": latency_results,
            "throughput_test": throughput_results,
            "stress_test": stress_results
        }
    
    def print_results(self, results: Dict[str, Any]):
        """Print formatted benchmark results"""
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        
        # System info
        if "system_info" in results:
            sys_info = results["system_info"]["initial"]
            print(f"\nSystem Information:")
            print(f"  CPU Cores: {sys_info['cpu_count']}")
            print(f"  CPU Frequency: {sys_info['cpu_freq']:.0f} MHz")
            print(f"  Memory Total: {sys_info['memory_total']:.1f} GB")
            print(f"  GPU Available: {sys_info.get('gpu_available', False)}")
            if sys_info.get('gpu_available'):
                print(f"  GPU Memory: {sys_info['gpu_memory_used']}/{sys_info['gpu_memory_total']} MB")
        
        # Latency results
        if "latency_test" in results and "processing_time" in results["latency_test"]:
            lat = results["latency_test"]
            print(f"\nLatency Test Results:")
            print(f"  Success Rate: {lat['success_rate']:.1%}")
            print(f"  Processing Time (avg): {lat['processing_time']['mean']:.3f}s")
            print(f"  Processing Time (min/max): {lat['processing_time']['min']:.3f}s / {lat['processing_time']['max']:.3f}s")
            print(f"  GPU Usage Rate: {lat.get('gpu_usage', 0):.1%}")
        
        # Throughput results
        if "throughput_test" in results:
            thr = results["throughput_test"]
            print(f"\nThroughput Test Results:")
            print(f"  Requests per Second: {thr['requests_per_second']:.2f}")
            print(f"  Success Rate: {thr['success_rate']:.1%}")
            print(f"  Average Processing Time: {thr['avg_processing_time']:.3f}s")
        
        # Stress test results
        if "stress_test" in results:
            stress = results["stress_test"]
            print(f"\nStress Test Results:")
            print(f"  Total Requests: {stress['total_requests']}")
            print(f"  Success Rate: {stress['success_rate']:.1%}")
            print(f"  Requests per Second: {stress['requests_per_second']:.2f}")
            print(f"  Processing Time (avg/max): {stress['avg_processing_time']:.3f}s / {stress['max_processing_time']:.3f}s")


async def main():
    parser = argparse.ArgumentParser(description="Chatbot API Benchmark")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--output", help="Output file for results (JSON)")
    parser.add_argument("--test", choices=["latency", "throughput", "stress", "all"], 
                       default="all", help="Test type to run")
    
    args = parser.parse_args()
    
    benchmark = ChatbotBenchmark(args.url)
    
    # Check if API is available
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{args.url}/health", timeout=10)
            if response.status_code != 200:
                print(f"API health check failed: {response.status_code}")
                return
    except Exception as e:
        print(f"Cannot connect to API at {args.url}: {e}")
        return
    
    # Run selected tests
    if args.test == "all":
        results = await benchmark.run_full_benchmark()
    elif args.test == "latency":
        results = {"latency_test": await benchmark.run_latency_test()}
    elif args.test == "throughput":
        results = {"throughput_test": await benchmark.run_throughput_test()}
    elif args.test == "stress":
        results = {"stress_test": await benchmark.run_stress_test()}
    
    # Print results
    benchmark.print_results(results)
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
