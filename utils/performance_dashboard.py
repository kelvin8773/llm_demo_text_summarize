# utils/performance_dashboard.py - Performance monitoring dashboard for Streamlit

"""
Performance monitoring dashboard for the Streamlit application.

This module provides real-time performance metrics, memory usage,
and system statistics for monitoring the application's health.
"""

import streamlit as st
import time
import psutil
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import Dict, List, Any
import threading
import queue
import logging
import functools
from .performance import (
    model_cache, 
    memory_manager, 
    performance_monitor,
    get_cache_stats,
    cleanup_resources
)

logger = logging.getLogger(__name__)

# Performance data storage
performance_data = {
    'timestamps': [],
    'memory_usage': [],
    'cpu_usage': [],
    'cache_size': [],
    'response_times': []
}

# Data collection settings
DATA_RETENTION_HOURS = 24
MAX_DATA_POINTS = 1000
COLLECTION_INTERVAL = 5  # seconds


class PerformanceCollector:
    """Background performance data collector."""
    
    def __init__(self):
        self.running = False
        self.thread = None
        self.data_queue = queue.Queue()
    
    def start(self):
        """Start collecting performance data."""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._collect_data, daemon=True)
            self.thread.start()
            logger.info("Performance collector started")
    
    def stop(self):
        """Stop collecting performance data."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        logger.info("Performance collector stopped")
    
    def _collect_data(self):
        """Collect performance data in background."""
        while self.running:
            try:
                # Get current metrics
                memory_stats = memory_manager.get_memory_usage()
                cpu_percent = psutil.cpu_percent()
                cache_stats = get_cache_stats()
                
                # Create data point
                data_point = {
                    'timestamp': datetime.now(),
                    'memory_mb': memory_stats['process_memory_mb'],
                    'memory_percent': memory_stats['process_memory_percent'],
                    'cpu_percent': cpu_percent,
                    'cache_size': cache_stats['cache_size']
                }
                
                # Add to queue
                self.data_queue.put(data_point)
                
                # Clean old data
                self._cleanup_old_data()
                
            except Exception as e:
                logger.error(f"Error collecting performance data: {e}")
            
            time.sleep(COLLECTION_INTERVAL)
    
    def _cleanup_old_data(self):
        """Remove old data points."""
        cutoff_time = datetime.now() - timedelta(hours=DATA_RETENTION_HOURS)
        
        # Clean up global performance data
        while performance_data['timestamps'] and performance_data['timestamps'][0] < cutoff_time:
            performance_data['timestamps'].pop(0)
            performance_data['memory_usage'].pop(0)
            performance_data['cpu_usage'].pop(0)
            performance_data['cache_size'].pop(0)
            if performance_data['response_times']:
                performance_data['response_times'].pop(0)
    
    def get_latest_data(self) -> Dict[str, Any]:
        """Get latest performance data."""
        latest_data = {}
        
        # Process queued data
        while not self.data_queue.empty():
            try:
                data_point = self.data_queue.get_nowait()
                
                # Add to global data
                performance_data['timestamps'].append(data_point['timestamp'])
                performance_data['memory_usage'].append(data_point['memory_mb'])
                performance_data['cpu_usage'].append(data_point['cpu_percent'])
                performance_data['cache_size'].append(data_point['cache_size'])
                
                # Keep only recent data
                if len(performance_data['timestamps']) > MAX_DATA_POINTS:
                    performance_data['timestamps'].pop(0)
                    performance_data['memory_usage'].pop(0)
                    performance_data['cpu_usage'].pop(0)
                    performance_data['cache_size'].pop(0)
                
                latest_data = data_point
                
            except queue.Empty:
                break
        
        return latest_data


# Global collector instance
collector = PerformanceCollector()


def start_performance_monitoring():
    """Start performance monitoring."""
    if 'performance_monitoring_started' not in st.session_state:
        collector.start()
        st.session_state.performance_monitoring_started = True


def stop_performance_monitoring():
    """Stop performance monitoring."""
    collector.stop()
    if 'performance_monitoring_started' in st.session_state:
        del st.session_state.performance_monitoring_started


def render_performance_dashboard():
    """Render the performance monitoring dashboard."""
    st.subheader("📊 Performance Dashboard")
    
    # Start monitoring if not already started
    start_performance_monitoring()
    
    # Get latest data
    latest_data = collector.get_latest_data()
    
    # Current metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        memory_stats = memory_manager.get_memory_usage()
        st.metric(
            "Memory Usage", 
            f"{memory_stats['process_memory_mb']:.1f} MB",
            delta=f"{memory_stats['process_memory_percent']:.1f}%"
        )
    
    with col2:
        cpu_percent = psutil.cpu_percent()
        st.metric("CPU Usage", f"{cpu_percent:.1f}%")
    
    with col3:
        cache_stats = get_cache_stats()
        st.metric("Cache Size", f"{cache_stats['cache_size']} items")
    
    with col4:
        system_memory = psutil.virtual_memory()
        st.metric(
            "System Memory", 
            f"{system_memory.percent:.1f}%",
            delta=f"{system_memory.available / 1024 / 1024 / 1024:.1f} GB available"
        )
    
    # Performance charts
    if performance_data['timestamps']:
        st.subheader("📈 Performance Trends")
        
        # Create charts
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Performance Metrics Over Time', fontsize=16)
        
        # Memory usage chart
        axes[0, 0].plot(performance_data['timestamps'], performance_data['memory_usage'], 'b-', linewidth=2)
        axes[0, 0].set_title('Memory Usage (MB)')
        axes[0, 0].set_ylabel('Memory (MB)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # CPU usage chart
        axes[0, 1].plot(performance_data['timestamps'], performance_data['cpu_usage'], 'r-', linewidth=2)
        axes[0, 1].set_title('CPU Usage (%)')
        axes[0, 1].set_ylabel('CPU (%)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Cache size chart
        axes[1, 0].plot(performance_data['timestamps'], performance_data['cache_size'], 'g-', linewidth=2)
        axes[1, 0].set_title('Cache Size')
        axes[1, 0].set_ylabel('Items')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Response times chart (if available)
        if performance_data['response_times']:
            axes[1, 1].plot(performance_data['timestamps'][:len(performance_data['response_times'])], 
                           performance_data['response_times'], 'm-', linewidth=2)
            axes[1, 1].set_title('Response Times (s)')
            axes[1, 1].set_ylabel('Time (s)')
        else:
            axes[1, 1].text(0.5, 0.5, 'No response time data', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Response Times')
        
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Format x-axis for all charts
        for ax in axes.flat:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Performance summary
    st.subheader("📋 Performance Summary")
    
    perf_summary = performance_monitor.get_performance_summary()
    if perf_summary:
        for operation, stats in perf_summary.items():
            with st.expander(f"Operation: {operation}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Count", stats['count'])
                with col2:
                    st.metric("Average Time", f"{stats['average']:.3f}s")
                with col3:
                    st.metric("Total Time", f"{stats['total']:.3f}s")
                
                st.metric("Min Time", f"{stats['min']:.3f}s")
                st.metric("Max Time", f"{stats['max']:.3f}s")
    
    # Cache statistics
    st.subheader("🗄️ Cache Statistics")
    cache_stats = get_cache_stats()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Cache Size", f"{cache_stats['cache_size']}/{cache_stats['max_cache_size']}")
        st.metric("Cache TTL", f"{cache_stats['cache_ttl']}s")
    
    with col2:
        memory_usage = cache_stats['memory_usage']
        st.metric("Process Memory", f"{memory_usage['process_memory_mb']:.1f} MB")
        st.metric("System Memory", f"{memory_usage['system_memory_percent']:.1f}%")
    
    # Control buttons
    st.subheader("🔧 Performance Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🧹 Clear Cache"):
            model_cache.clear()
            st.success("Cache cleared!")
            st.rerun()
    
    with col2:
        if st.button("💾 Cleanup Memory"):
            cleanup_stats = memory_manager.cleanup_memory()
            st.success(f"Memory cleanup completed! Freed {cleanup_stats['memory_freed_mb']:.1f} MB")
            st.rerun()
    
    with col3:
        if st.button("🔄 Reset Metrics"):
            performance_monitor.metrics.clear()
            performance_monitor.start_times.clear()
            st.success("Performance metrics reset!")
            st.rerun()
    
    # System information
    st.subheader("💻 System Information")
    
    system_info = {
        "CPU Count": psutil.cpu_count(),
        "CPU Frequency": f"{psutil.cpu_freq().current:.0f} MHz" if psutil.cpu_freq() else "N/A",
        "Total Memory": f"{psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f} GB",
        "Available Memory": f"{psutil.virtual_memory().available / 1024 / 1024 / 1024:.1f} GB",
        "Disk Usage": f"{psutil.disk_usage('/').percent:.1f}%",
        "Python Version": f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}.{psutil.sys.version_info.micro}"
    }
    
    for key, value in system_info.items():
        st.text(f"{key}: {value}")


def render_performance_widget():
    """Render a compact performance widget for the sidebar."""
    if st.sidebar.button("📊 Performance Dashboard"):
        st.session_state.show_performance_dashboard = True
    
    if st.session_state.get('show_performance_dashboard', False):
        with st.sidebar:
            st.subheader("📊 Quick Stats")
            
            # Get current metrics
            memory_stats = memory_manager.get_memory_usage()
            cpu_percent = psutil.cpu_percent()
            cache_stats = get_cache_stats()
            
            st.metric("Memory", f"{memory_stats['process_memory_mb']:.1f} MB")
            st.metric("CPU", f"{cpu_percent:.1f}%")
            st.metric("Cache", f"{cache_stats['cache_size']} items")
            
            if st.button("🧹 Clear Cache", key="sidebar_clear_cache"):
                model_cache.clear()
                st.success("Cache cleared!")
            
            if st.button("💾 Cleanup Memory", key="sidebar_cleanup"):
                cleanup_stats = memory_manager.cleanup_memory()
                st.success(f"Freed {cleanup_stats['memory_freed_mb']:.1f} MB")


def track_operation_time(operation_name: str):
    """Decorator to track operation time in Streamlit."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                
                # Add to response times data
                performance_data['response_times'].append(duration)
                
                # Keep only recent data
                if len(performance_data['response_times']) > MAX_DATA_POINTS:
                    performance_data['response_times'].pop(0)
                
                # Log performance
                logger.debug(f"Operation '{operation_name}' took {duration:.3f} seconds")
        
        return wrapper
    return decorator


def get_performance_alerts() -> List[str]:
    """Get performance alerts based on current metrics."""
    alerts = []
    
    # Memory alerts
    memory_stats = memory_manager.get_memory_usage()
    if memory_stats['process_memory_percent'] > 80:
        alerts.append(f"⚠️ High memory usage: {memory_stats['process_memory_percent']:.1f}%")
    
    if memory_stats['system_memory_percent'] > 90:
        alerts.append(f"🚨 Critical system memory: {memory_stats['system_memory_percent']:.1f}%")
    
    # CPU alerts
    cpu_percent = psutil.cpu_percent()
    if cpu_percent > 80:
        alerts.append(f"⚠️ High CPU usage: {cpu_percent:.1f}%")
    
    # Cache alerts
    cache_stats = get_cache_stats()
    if cache_stats['cache_size'] > cache_stats['max_cache_size'] * 0.9:
        alerts.append(f"⚠️ Cache nearly full: {cache_stats['cache_size']}/{cache_stats['max_cache_size']}")
    
    return alerts


def render_performance_alerts():
    """Render performance alerts in the UI."""
    alerts = get_performance_alerts()
    
    if alerts:
        st.subheader("🚨 Performance Alerts")
        for alert in alerts:
            st.warning(alert)
        
        # Auto-cleanup suggestion
        if any("memory" in alert.lower() for alert in alerts):
            if st.button("🧹 Auto Cleanup"):
                cleanup_stats = memory_manager.cleanup_memory()
                st.success(f"Auto cleanup completed! Freed {cleanup_stats['memory_freed_mb']:.1f} MB")
                st.rerun()