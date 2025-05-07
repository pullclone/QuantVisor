# QuantVisor
have a look at real world numbers for employing different quantization technologies on your hardware

How psutil Monitoring Works in this Script:

    The run_benchmark function now uses subprocess.Popen to start llama.cpp. This is non-blocking.

    A psutil.Process object is created using the PID of the llama.cpp subprocess.

    In a while process.poll() is None: loop (meaning the process is still running):

        p_psutil.memory_info().rss gets the current Resident Set Size (physical memory). The peak value is tracked.

        p_psutil.cpu_percent(interval=PSUTIL_MONITOR_INTERVAL) gets the CPU percentage used by the process since the last call or over the specified interval. This is collected in a list.

    After the llama.cpp process finishes, the average CPU utilization and peak memory are calculated and added to the results.

This refined script should give you much more detailed performance data, including process-specific CPU and memory usage, for your chosen models and quantizations. Remember that the accuracy of cpu_percent over very short durations might vary, but for tasks lasting several seconds, it should provide a good indication.