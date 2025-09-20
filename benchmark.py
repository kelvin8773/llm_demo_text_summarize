#!/usr/bin/env python3
"""
Performance benchmark script for LLM Text Summarization Tool

This script runs comprehensive performance benchmarks to measure
the efficiency and speed of different components.
"""

import time
import psutil
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any
import statistics
import json

# Import project modules
from utils.fast_summarize import fast_summarize_text
from utils.enhance_summarize import enhance_summarize_text
from utils.chinese_summarize import chinese_summarize_text
from utils.insights import extract_keywords, extract_keywords_phrases
from utils.chinese_insights import extract_chinese_keywords
from utils.performance import (
    model_cache, 
    memory_manager, 
    performance_monitor,
    cleanup_resources
)
from tests.fixtures.sample_texts import (
    ENGLISH_SHORT_TEXT, ENGLISH_MEDIUM_TEXT, ENGLISH_LONG_TEXT,
    CHINESE_SHORT_TEXT, CHINESE_MEDIUM_TEXT, CHINESE_LONG_TEXT
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Performance benchmark runner."""
    
    def __init__(self, iterations: int = 3, warmup_iterations: int = 1):
        self.iterations = iterations
        self.warmup_iterations = warmup_iterations
        self.results = {}
    
    def measure_function(self, func, *args, **kwargs) -> Dict[str, float]:
        """Measure function execution time and memory usage."""
        times = []
        memory_before = memory_manager.get_memory_usage()
        
        # Warmup iterations
        for _ in range(self.warmup_iterations):
            try:
                func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Warmup failed: {e}")
        
        # Actual measurements
        for _ in range(self.iterations):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                times.append(end_time - start_time)
            except Exception as e:
                logger.error(f"Benchmark failed: {e}")
                times.append(float('inf'))
        
        memory_after = memory_manager.get_memory_usage()
        
        return {
            'times': times,
            'avg_time': statistics.mean(times) if times else 0,
            'min_time': min(times) if times else 0,
            'max_time': max(times) if times else 0,
            'std_time': statistics.stdev(times) if len(times) > 1 else 0,
            'memory_before_mb': memory_before['process_memory_mb'],
            'memory_after_mb': memory_after['process_memory_mb'],
            'memory_delta_mb': memory_after['process_memory_mb'] - memory_before['process_memory_mb']
        }
    
    def benchmark_summarization(self) -> Dict[str, Any]:
        """Benchmark summarization functions."""
        logger.info("Running summarization benchmarks...")
        
        results = {}
        
        # English fast summarization
        logger.info("Benchmarking English fast summarization...")
        results['fast_summarize'] = {
            'short': self.measure_function(fast_summarize_text, ENGLISH_SHORT_TEXT, max_sentences=3),
            'medium': self.measure_function(fast_summarize_text, ENGLISH_MEDIUM_TEXT, max_sentences=5),
            'long': self.measure_function(fast_summarize_text, ENGLISH_LONG_TEXT, max_sentences=10)
        }
        
        # English enhanced summarization
        logger.info("Benchmarking English enhanced summarization...")
        results['enhance_summarize'] = {
            'short': self.measure_function(enhance_summarize_text, ENGLISH_SHORT_TEXT, max_sentences=3),
            'medium': self.measure_function(enhance_summarize_text, ENGLISH_MEDIUM_TEXT, max_sentences=5),
            'long': self.measure_function(enhance_summarize_text, ENGLISH_LONG_TEXT, max_sentences=10)
        }
        
        # Chinese summarization
        logger.info("Benchmarking Chinese summarization...")
        results['chinese_summarize'] = {
            'short': self.measure_function(chinese_summarize_text, CHINESE_SHORT_TEXT, max_sentences=3),
            'medium': self.measure_function(chinese_summarize_text, CHINESE_MEDIUM_TEXT, max_sentences=5),
            'long': self.measure_function(chinese_summarize_text, CHINESE_LONG_TEXT, max_sentences=10)
        }
        
        return results
    
    def benchmark_keyword_extraction(self) -> Dict[str, Any]:
        """Benchmark keyword extraction functions."""
        logger.info("Running keyword extraction benchmarks...")
        
        results = {}
        
        # English keyword extraction
        logger.info("Benchmarking English keyword extraction...")
        results['english_keywords'] = {
            'short': self.measure_function(extract_keywords, ENGLISH_SHORT_TEXT, top_n=10),
            'medium': self.measure_function(extract_keywords, ENGLISH_MEDIUM_TEXT, top_n=15),
            'long': self.measure_function(extract_keywords, ENGLISH_LONG_TEXT, top_n=20)
        }
        
        # English phrase extraction
        logger.info("Benchmarking English phrase extraction...")
        results['english_phrases'] = {
            'short': self.measure_function(extract_keywords_phrases, ENGLISH_SHORT_TEXT, top_n=10),
            'medium': self.measure_function(extract_keywords_phrases, ENGLISH_MEDIUM_TEXT, top_n=15),
            'long': self.measure_function(extract_keywords_phrases, ENGLISH_LONG_TEXT, top_n=20)
        }
        
        # Chinese keyword extraction
        logger.info("Benchmarking Chinese keyword extraction...")
        results['chinese_keywords'] = {
            'short': self.measure_function(extract_chinese_keywords, CHINESE_SHORT_TEXT, top_n=10),
            'medium': self.measure_function(extract_chinese_keywords, CHINESE_MEDIUM_TEXT, top_n=15),
            'long': self.measure_function(extract_chinese_keywords, CHINESE_LONG_TEXT, top_n=20)
        }
        
        return results
    
    def benchmark_cache_performance(self) -> Dict[str, Any]:
        """Benchmark cache performance."""
        logger.info("Running cache performance benchmarks...")
        
        # Clear cache first
        model_cache.clear()
        
        results = {}
        
        # Test cache miss (first load)
        logger.info("Testing cache miss performance...")
        cache_miss_times = []
        for _ in range(self.iterations):
            model_cache.clear()  # Clear cache before each test
            start_time = time.time()
            fast_summarize_text(ENGLISH_MEDIUM_TEXT, max_sentences=5)
            end_time = time.time()
            cache_miss_times.append(end_time - start_time)
        
        # Test cache hit (subsequent loads)
        logger.info("Testing cache hit performance...")
        cache_hit_times = []
        for _ in range(self.iterations):
            start_time = time.time()
            fast_summarize_text(ENGLISH_MEDIUM_TEXT, max_sentences=5)
            end_time = time.time()
            cache_hit_times.append(end_time - start_time)
        
        results['cache_miss'] = {
            'avg_time': statistics.mean(cache_miss_times),
            'min_time': min(cache_miss_times),
            'max_time': max(cache_miss_times),
            'std_time': statistics.stdev(cache_miss_times) if len(cache_miss_times) > 1 else 0
        }
        
        results['cache_hit'] = {
            'avg_time': statistics.mean(cache_hit_times),
            'min_time': min(cache_hit_times),
            'max_time': max(cache_hit_times),
            'std_time': statistics.stdev(cache_hit_times) if len(cache_hit_times) > 1 else 0
        }
        
        results['cache_speedup'] = results['cache_miss']['avg_time'] / results['cache_hit']['avg_time']
        
        return results
    
    def benchmark_memory_usage(self) -> Dict[str, Any]:
        """Benchmark memory usage patterns."""
        logger.info("Running memory usage benchmarks...")
        
        results = {}
        
        # Measure memory usage for different text sizes
        text_sizes = [len(ENGLISH_SHORT_TEXT), len(ENGLISH_MEDIUM_TEXT), len(ENGLISH_LONG_TEXT)]
        texts = [ENGLISH_SHORT_TEXT, ENGLISH_MEDIUM_TEXT, ENGLISH_LONG_TEXT]
        
        memory_usage = []
        for text in texts:
            memory_before = memory_manager.get_memory_usage()
            fast_summarize_text(text, max_sentences=5)
            memory_after = memory_manager.get_memory_usage()
            
            memory_usage.append({
                'text_length': len(text),
                'memory_before_mb': memory_before['process_memory_mb'],
                'memory_after_mb': memory_after['process_memory_mb'],
                'memory_delta_mb': memory_after['process_memory_mb'] - memory_before['process_memory_mb']
            })
        
        results['memory_by_text_size'] = memory_usage
        
        # Measure memory cleanup effectiveness
        logger.info("Testing memory cleanup effectiveness...")
        memory_before = memory_manager.get_memory_usage()
        
        # Load some models to increase memory usage
        fast_summarize_text(ENGLISH_LONG_TEXT, max_sentences=10)
        enhance_summarize_text(ENGLISH_LONG_TEXT, max_sentences=10)
        chinese_summarize_text(CHINESE_LONG_TEXT, max_sentences=10)
        
        memory_after_loading = memory_manager.get_memory_usage()
        
        # Perform cleanup
        cleanup_stats = cleanup_resources()
        
        memory_after_cleanup = memory_manager.get_memory_usage()
        
        results['memory_cleanup'] = {
            'memory_before_mb': memory_before['process_memory_mb'],
            'memory_after_loading_mb': memory_after_loading['process_memory_mb'],
            'memory_after_cleanup_mb': memory_after_cleanup['process_memory_mb'],
            'cleanup_stats': cleanup_stats
        }
        
        return results
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all benchmarks."""
        logger.info("Starting comprehensive performance benchmarks...")
        
        all_results = {
            'timestamp': time.time(),
            'system_info': self._get_system_info(),
            'benchmark_config': {
                'iterations': self.iterations,
                'warmup_iterations': self.warmup_iterations
            }
        }
        
        try:
            all_results['summarization'] = self.benchmark_summarization()
        except Exception as e:
            logger.error(f"Summarization benchmark failed: {e}")
            all_results['summarization'] = {'error': str(e)}
        
        try:
            all_results['keyword_extraction'] = self.benchmark_keyword_extraction()
        except Exception as e:
            logger.error(f"Keyword extraction benchmark failed: {e}")
            all_results['keyword_extraction'] = {'error': str(e)}
        
        try:
            all_results['cache_performance'] = self.benchmark_cache_performance()
        except Exception as e:
            logger.error(f"Cache performance benchmark failed: {e}")
            all_results['cache_performance'] = {'error': str(e)}
        
        try:
            all_results['memory_usage'] = self.benchmark_memory_usage()
        except Exception as e:
            logger.error(f"Memory usage benchmark failed: {e}")
            all_results['memory_usage'] = {'error': str(e)}
        
        return all_results
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_frequency': psutil.cpu_freq().current if psutil.cpu_freq() else None,
            'total_memory_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
            'available_memory_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024,
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage_percent': psutil.disk_usage('/').percent
        }
    
    def save_results(self, results: Dict[str, Any], output_file: Path) -> None:
        """Save benchmark results to file."""
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Benchmark results saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def print_summary(self, results: Dict[str, Any]) -> None:
        """Print benchmark summary."""
        print("\n" + "="*60)
        print("PERFORMANCE BENCHMARK SUMMARY")
        print("="*60)
        
        # System info
        system_info = results.get('system_info', {})
        print(f"CPU Cores: {system_info.get('cpu_count', 'N/A')}")
        print(f"Total Memory: {system_info.get('total_memory_gb', 0):.1f} GB")
        print(f"Available Memory: {system_info.get('available_memory_gb', 0):.1f} GB")
        
        # Summarization performance
        if 'summarization' in results and 'error' not in results['summarization']:
            print("\nSUMMARIZATION PERFORMANCE:")
            for func_name, sizes in results['summarization'].items():
                print(f"  {func_name}:")
                for size, metrics in sizes.items():
                    print(f"    {size}: {metrics['avg_time']:.3f}s avg ({metrics['min_time']:.3f}s - {metrics['max_time']:.3f}s)")
        
        # Cache performance
        if 'cache_performance' in results and 'error' not in results['cache_performance']:
            cache_perf = results['cache_performance']
            print(f"\nCACHE PERFORMANCE:")
            print(f"  Cache Miss: {cache_perf['cache_miss']['avg_time']:.3f}s")
            print(f"  Cache Hit: {cache_perf['cache_hit']['avg_time']:.3f}s")
            print(f"  Speedup: {cache_perf['cache_speedup']:.2f}x")
        
        print("="*60)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Performance benchmark for LLM Text Summarization Tool")
    parser.add_argument("--iterations", type=int, default=3, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup iterations")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output file for results")
    parser.add_argument("--summary-only", action="store_true", help="Only print summary, don't save detailed results")
    
    args = parser.parse_args()
    
    # Create benchmark runner
    runner = BenchmarkRunner(iterations=args.iterations, warmup_iterations=args.warmup)
    
    # Run benchmarks
    results = runner.run_all_benchmarks()
    
    # Print summary
    runner.print_summary(results)
    
    # Save results if requested
    if not args.summary_only:
        output_file = Path(args.output)
        runner.save_results(results, output_file)


if __name__ == "__main__":
    main()