import cProfile
import pstats
import io
import line_profiler
import tracemalloc
import time
import logging
import os
import psutil
import torch
from functools import wraps
from memory_profiler import profile



class Profiler:
    def __init__(self, logger=None):
        self.logger = logger if logger else logging.getLogger(__name__)
        self.profiler = cProfile.Profile()
        self.line_profiler = line_profiler.LineProfiler()
        self.tracemalloc_enabled = False
        self.system_info = {}
        
    def start_profiling(self):
        """Start the cProfile profiler."""
        self.profiler.enable()
        self.logger.info("Profiling started...")

    def stop_profiling(self):
        """Stop the cProfile profiler and display results."""
        self.profiler.disable()
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s).sort_stats('cumulative')
        ps.print_stats()
        self.logger.info("Profiling results:\n" + s.getvalue())
    
    def profile_function(self, func):
        """Decorator for function-level profiling using cProfile."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.profiler.enable()
            result = func(*args, **kwargs)
            self.profiler.disable()
            return result
        return wrapper
    
    def profile_lines(self, func):
        """Decorator for line-by-line profiling."""
        self.line_profiler.add_function(func)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.line_profiler(func)(*args, **kwargs)
        
        return wrapper
    
    def print_line_profile(self):
        """Display results of line-by-line profiling."""
        s = io.StringIO()
        self.line_profiler.print_stats(stream=s)
        self.logger.info("Line-by-line profiling results:\n" + s.getvalue())
    
    def start_memory_profiling(self):
        """Start tracking memory usage for CPU and GPU."""
        tracemalloc.start()
        self.tracemalloc_enabled = True
        self.logger.info("Memory profiling started...")

    def stop_memory_profiling(self):
        """Stop memory tracking and log results."""
        if self.tracemalloc_enabled:
            snapshot = tracemalloc.take_snapshot()
            stats = snapshot.statistics('lineno')
            self.logger.info("Top 10 memory allocations:")
            for stat in stats[:10]:
                self.logger.info(stat)
            tracemalloc.stop()
            self.tracemalloc_enabled = False

    def get_system_info(self):
        """Collect system resource information."""
        self.system_info['CPU Usage (%)'] = psutil.cpu_percent(interval=1)
        self.system_info['RAM Usage (MB)'] = psutil.virtual_memory().used / (1024 * 1024)
        if torch.cuda.is_available():
            self.system_info['GPU Usage (%)'] = torch.cuda.utilization(0)
            self.system_info['GPU Memory (MB)'] = torch.cuda.memory_allocated(0) / (1024 * 1024)
        else:
            self.system_info['GPU Usage (%)'] = 'N/A'
            self.system_info['GPU Memory (MB)'] = 'N/A'
        return self.system_info

    def time_function(self, func):
        """Decorator for measuring execution time."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            self.logger.info(f"Execution time for {func.__name__}: {end - start:.6f} seconds")
            return result
        return wrapper
    
    def profile_memory(self, func):
        """Decorator for measuring memory usage."""
        return profile(func)
    
    def profile_all(self, func):
        """Decorator that applies multiple profiling techniques."""
        func = self.profile_function(func)
        func = self.profile_lines(func)
        func = self.time_function(func)
        func = self.profile_memory(func)
        return func

    def log_system_info(self):
        """Log system resource usage."""
        sys_info = self.get_system_info()
        self.logger.info(f"System Info: {sys_info}")

    def profile_script(self, script_path):
        """Run a script with profiling enabled."""
        self.logger.info(f"Profiling script: {script_path}")
        self.start_profiling()
        with open(script_path, "rb") as script_file:
            exec(script_file.read(), {})
        self.stop_profiling()
        self.log_system_info()
