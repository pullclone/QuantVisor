#!/usr/bin/env python3
# QuantVisor: LLM Performance Benchmarking Tool
# Combines setup, llama.cpp GGUF benchmarking, and Ollama benchmarking.

import subprocess
import re
import csv
import os
import sys
import time
import datetime
import json
from pathlib import Path

# --- Dependency Management ---
# Try to import, and list missing ones for the user.
MISSING_PACKAGES = []
INSTALL_INSTRUCTIONS = []

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    MISSING_PACKAGES.append("psutil")
    INSTALL_INSTRUCTIONS.append("pip install psutil")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    MISSING_PACKAGES.append("requests")
    INSTALL_INSTRUCTIONS.append("pip install requests")

try:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError, GatedRepoError, LocalEntryNotFoundError
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    MISSING_PACKAGES.append("huggingface-hub")
    INSTALL_INSTRUCTIONS.append("pip install huggingface-hub")

# --- User Configuration ---

# GENERAL SETTINGS
BENCHMARK_ENGINE = "ollama"  # Choose "llama.cpp" or "ollama"
# BENCHMARK_ENGINE = "llama.cpp" # Uncomment to use llama.cpp

# Common benchmark parameters
BENCHMARK_PROMPT = "Describe the process of photosynthesis in simple terms for a fifth grader, include key vocabulary."
TOKENS_TO_GENERATE = 128
REPETITIONS_PER_TEST = 2 # Number of times to run each specific test for averaging
OUTPUT_CSV_BASENAME = "quantvisor_benchmark_results" # Timestamp will be appended

# --- LLAMA.CPP SPECIFIC CONFIGURATION ---
# (Only used if BENCHMARK_ENGINE is "llama.cpp")

# Path to the llama.cpp executable (e.g., ./main, ./llama-cli, or full path)
# The script will prompt you if this is not found or not executable.
LLAMA_CPP_EXECUTABLE = "./main" # Example: "./main" or "C:\\Users\\YourUser\\llama.cpp\\main.exe"

# Base directory where GGUF models for llama.cpp are stored or will be downloaded.
LLAMA_CPP_MODELS_BASE_DIR = "./models_gguf"

# Define GGUF models and files for llama.cpp.
# Structure: { "repo_id_for_download": { "target_folder_name": ["file1.gguf", "file2.gguf"], ...}}
# target_folder_name is the directory created within LLAMA_CPP_MODELS_BASE_DIR
LLAMA_CPP_MODELS_TO_DOWNLOAD_AND_TEST = {
    "mradermacher/Llama-3.2-3B-Instruct-uncensored-GGUF": {
        "Llama-3.2-3B-Instruct-uncensored-GGUF": [ # This becomes a subfolder
            "Llama-3.2-3B-Instruct-uncensored.Q4_K_M.gguf",
            "Llama-3.2-3B-Instruct-uncensored.Q5_K_M.gguf",
            # "Llama-3.2-3B-Instruct-uncensored.Q6_K.gguf",
            # "Llama-3.2-3B-Instruct-uncensored.Q8_0.gguf",
        ]
    },
    "mradermacher/Llama-3.2-3B-Instruct-uncensored-i1-GGUF": {
        "Llama-3.2-3B-Instruct-uncensored-i1-GGUF": [
            "Llama-3.2-3B-Instruct-uncensored.IQ2_XS.gguf",
            "Llama-3.2-3B-Instruct-uncensored.IQ4_XS.gguf",
            # "Llama-3.2-3B-Instruct-uncensored.IQ2_S.gguf",
        ]
    },
}
# CPU threads to test with llama.cpp (e.g., for Intel i7-8550U with 4 cores / 8 threads, 4 is a good start)
LLAMA_CPP_THREADS_TO_TEST = [4] # Example: [4, 8]
# Context sizes to test with llama.cpp
LLAMA_CPP_CONTEXT_SIZES_TO_TEST = [512] # Example: [512, 1024, 2048]


# --- OLLAMA SPECIFIC CONFIGURATION ---
# (Only used if BENCHMARK_ENGINE is "ollama")

OLLAMA_API_BASE_URL = "http://localhost:11434" # Default Ollama API URL
# Define Ollama models to test. These are the model tags Ollama uses (e.g., "llama3:latest").
# The script will attempt to pull these if they are not available locally via 'ollama list'.
# You can find model tags on Ollama Hub: https://ollama.com/library
OLLAMA_MODELS_TO_TEST = [
    # General purpose models
    "llama3:8b-instruct-fp16",
    "llama3:8b-instruct-q5_K_M",
    "llama3:8b-instruct-q4_K_M",
    "mistral:7b-instruct-v0.2-q5_K_M",
    "mistral:7b-instruct-v0.2-q4_K_M",
    # If you have created custom Ollama models from the mradermacher GGUFs, add their names here:
    # "my-custom-llama-3.2-3b-q4km:latest",
    # "my-custom-llama-3.2-3b-iq2xs:latest",
]

# --- psutil Monitoring (Applies to llama.cpp engine if PSUTIL_AVAILABLE) ---
PSUTIL_MONITOR_INTERVAL = 0.25 # Seconds. Lower for more granularity, higher for less overhead.

# --- Internal Script Globals ---
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_OUTPUT_FILE = f"{OUTPUT_CSV_BASENAME}_{TIMESTAMP}.csv"
PYTHON_CMD = sys.executable # Path to current Python interpreter

# --- Helper Functions ---
def print_error(message): print(f"ERROR: {message}", file=sys.stderr)
def print_warning(message): print(f"WARNING: {message}", file=sys.stdout)
def print_info(message): print(f"INFO: {message}", file=sys.stdout)

def run_subprocess_command(command_list, capture_output=True, text=True, check_on_error=False, env=None):
    print_info(f"Running command: {' '.join(command_list)}")
    try:
        process = subprocess.run(command_list, capture_output=capture_output, text=text, check=check_on_error, encoding='utf-8', env=env)
        return process
    except FileNotFoundError:
        print_error(f"Command not found: {command_list[0]}. Ensure it's installed and in your PATH.")
        return None
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed: {' '.join(e.cmd)} (Return code: {e.returncode})")
        if e.stdout: print_error(f"Stdout: {e.stdout.strip()}")
        if e.stderr: print_error(f"Stderr: {e.stderr.strip()}")
        return None
    except Exception as e:
        print_error(f"An unexpected error occurred running command {' '.join(command_list)}: {e}")
        return None

def check_and_guide_dependencies():
    if not MISSING_PACKAGES:
        print_info("All required Python packages are installed.")
        return True

    print_warning(f"The following Python packages are missing or could not be imported: {', '.join(MISSING_PACKAGES)}")
    print_info("Please install them manually using pip. For example:")
    for instruction in INSTALL_INSTRUCTIONS:
        print_info(f"  {PYTHON_CMD} -m {instruction}")
    print_error("Dependencies missing. Please install them and re-run QuantVisor.")
    return False

# --- LLAMA.CPP Engine Specific Functions ---
def download_gguf_model(repo_id, filename, target_dir):
    if not HF_HUB_AVAILABLE:
        print_error("huggingface-hub library is not available for GGUF downloads.")
        return False
    
    target_file_path = Path(target_dir) / filename
    if target_file_path.exists():
        print_info(f"GGUF Model file '{filename}' already exists in '{target_dir}'. Skipping download.")
        return True

    print_info(f"Downloading '{filename}' from '{repo_id}' to '{target_dir}'...")
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    try:
        hf_hub_download(
            repo_id=repo_id, filename=filename, local_dir=target_dir,
            local_dir_use_symlinks=False, resume_download=True, token=True # token=True uses stored token
        )
        print_info(f"Successfully downloaded '{filename}'.")
        return True
    except LocalEntryNotFoundError: # File might not exist on the repo
        print_error(f"File '{filename}' not found in repo '{repo_id}'. Please check filename and repo.")
        return False
    except GatedRepoError:
        print_error(f"Repo '{repo_id}' is gated. You need to accept terms on Hugging Face and be logged in (`huggingface-cli login`).")
        return False
    except HfHubHTTPError as e:
        print_error(f"HTTP error downloading {filename} from {repo_id}: {e}")
        if "401" in str(e) or "403" in str(e):
            print_error("Ensure you are logged in (`huggingface-cli login`) and have access to the model.")
        return False
    except Exception as e:
        print_error(f"Failed to download '{filename}': {type(e).__name__}: {e}")
        return False

def setup_llama_cpp_environment():
    print_info("\n--- Setting up for llama.cpp ---")
    global LLAMA_CPP_EXECUTABLE
    llama_cpp_path = Path(LLAMA_CPP_EXECUTABLE).resolve() # Get absolute path early

    # Check for executable in current dir, then PATH if not absolute
    if not llama_cpp_path.is_absolute():
        # Check current directory
        if not (llama_cpp_path.exists() and os.access(llama_cpp_path, os.X_OK)):
            # Check PATH
            found_in_path = None
            for path_dir in os.environ.get("PATH", "").split(os.pathsep):
                potential_path = Path(path_dir) / LLAMA_CPP_EXECUTABLE
                if potential_path.exists() and os.access(potential_path, os.X_OK):
                    found_in_path = potential_path.resolve()
                    break
            if found_in_path:
                LLAMA_CPP_EXECUTABLE = str(found_in_path)
                llama_cpp_path = found_in_path
            else: # Prompt user if not found
                 while not (llama_cpp_path.exists() and os.access(llama_cpp_path, os.X_OK)):
                    print_warning(f"llama.cpp executable not found or not executable at '{LLAMA_CPP_EXECUTABLE}'.")
                    print_info("Please ensure llama.cpp is compiled (e.g., run 'make' or CMake build).")
                    user_path = input(f"Enter the correct path to your llama.cpp executable (or 'skip' to skip llama.cpp): ").strip()
                    if user_path.lower() == 'skip':
                        print_warning("Skipping llama.cpp benchmarks as executable is not configured.")
                        return False
                    LLAMA_CPP_EXECUTABLE = user_path
                    llama_cpp_path = Path(LLAMA_CPP_EXECUTABLE).resolve() # Re-resolve
    elif not (llama_cpp_path.exists() and os.access(llama_cpp_path, os.X_OK)): # Absolute path given but not valid
        print_error(f"llama.cpp executable at absolute path '{llama_cpp_path}' not found or not executable.")
        return False

    print_info(f"Using llama.cpp executable: {llama_cpp_path}")

    if not LLAMA_CPP_MODELS_TO_DOWNLOAD_AND_TEST:
        print_info("No GGUF models configured in LLAMA_CPP_MODELS_TO_DOWNLOAD_AND_TEST.")
        return True
    Path(LLAMA_CPP_MODELS_BASE_DIR).mkdir(parents=True, exist_ok=True)
    all_downloads_successful = True
    for repo_id, targets in LLAMA_CPP_MODELS_TO_DOWNLOAD_AND_TEST.items():
        for target_folder_name, gguf_files in targets.items():
            current_target_dir = Path(LLAMA_CPP_MODELS_BASE_DIR) / target_folder_name
            for gguf_file in gguf_files:
                if not download_gguf_model(repo_id, gguf_file, current_target_dir):
                    all_downloads_successful = False
    return all_downloads_successful

def parse_llama_cpp_output(output_text):
    # Same parsing logic as before
    metrics = {"load_time_ms": None, "sample_time_ms": None, "sample_tokens": None, "sample_ms_per_token": None, "sample_tokens_per_sec": None, "prompt_eval_time_ms": None, "prompt_eval_tokens": None, "prompt_eval_ms_per_token": None, "prompt_eval_tokens_per_sec": None, "eval_time_ms": None, "eval_tokens": None, "eval_ms_per_token": None, "eval_tokens_per_sec": None, "total_time_ms": None, "ggml_used_mem_mb": None}
    patterns = {"load_time_ms": r"load time\s*=\s*([\d\.]+)\s*ms", "sample_params": r"sample time\s*=\s*([\d\.]+)\s*ms\s*/\s*(\d+)\s*tokens\s*\(\s*([\d\.]+)\s*ms per token,\s*([\d\.]+)\s*tokens per second\)", "prompt_eval_params": r"prompt eval time\s*=\s*([\d\.]+)\s*ms\s*/\s*(\d+)\s*tokens\s*\(\s*([\d\.]+)\s*ms per token,\s*([\d\.]+)\s*tokens per second\)", "eval_params": r"eval time\s*=\s*([\d\.]+)\s*ms\s*/\s*(\d+)\s*tokens\s*\(\s*([\d\.]+)\s*ms per token,\s*([\d\.]+)\s*tokens per second\)", "total_time_ms": r"total time\s*=\s*([\d\.]+)\s*ms", "ggml_used_mem_mb": r"ggml_used_mem\s*=\s*([\d\.]+)\s*MB"}
    for line in output_text.splitlines():
        if "llama_print_timings:" in line or "ggml_used_mem" in line :
            for key, pattern in patterns.items():
                match = re.search(pattern, line)
                if match:
                    if key in ["load_time_ms", "total_time_ms", "ggml_used_mem_mb"]: metrics[key] = float(match.group(1))
                    elif key == "sample_params": metrics.update({"sample_time_ms": float(match.group(1)), "sample_tokens": int(match.group(2)), "sample_ms_per_token": float(match.group(3)), "sample_tokens_per_sec": float(match.group(4))})
                    elif key == "prompt_eval_params": metrics.update({"prompt_eval_time_ms": float(match.group(1)), "prompt_eval_tokens": int(match.group(2)), "prompt_eval_ms_per_token": float(match.group(3)), "prompt_eval_tokens_per_sec": float(match.group(4))})
                    elif key == "eval_params": metrics.update({"eval_time_ms": float(match.group(1)), "eval_tokens": int(match.group(2)), "eval_ms_per_token": float(match.group(3)), "eval_tokens_per_sec": float(match.group(4))})
                    break
    return metrics

def run_llama_cpp_benchmark(model_path, threads, context_size, prompt, n_predict):
    # Same execution logic as before, using the globally confirmed LLAMA_CPP_EXECUTABLE
    command = [ str(Path(LLAMA_CPP_EXECUTABLE).resolve()), "-m", str(model_path), "-t", str(threads), "-c", str(context_size), "-p", prompt, "-n", str(n_predict), "--no-display-prompt", "-ngl", "0" ] # -ngl 0 to ensure CPU for Intel Iris
    print_info(f"Running llama.cpp: {' '.join(command)}")
    llama_metrics, psutil_metrics = {}, {"peak_mem_rss_mb": None, "avg_cpu_percent": None}
    process_handle = None
    try:
        start_time = time.time()
        process_handle = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
        peak_mem_rss, cpu_percents = 0, []
        if PSUTIL_AVAILABLE and process_handle.pid:
            try:
                p_psutil = psutil.Process(process_handle.pid)
                while process_handle.poll() is None:
                    try:
                        mem_info = p_psutil.memory_info(); peak_mem_rss = max(peak_mem_rss, mem_info.rss)
                        cpu_percents.append(p_psutil.cpu_percent(interval=PSUTIL_MONITOR_INTERVAL))
                    except (psutil.NoSuchProcess, psutil.AccessDenied, Exception): break
            except (psutil.NoSuchProcess, psutil.AccessDenied): print_warning(f"psutil could not attach to llama.cpp PID {process_handle.pid}.")
        stdout, stderr = process_handle.communicate(timeout=600) # 10 min timeout per run
        end_time = time.time()
        if process_handle.returncode != 0:
            print_error(f"llama.cpp failed (code {process_handle.returncode}) for {model_path}:\nStdout: {stdout}\nStderr: {stderr}")
            return None
        full_output = stdout + "\n" + stderr
        llama_metrics = parse_llama_cpp_output(full_output)
        llama_metrics["wall_time_s"] = round(end_time - start_time, 3)
        if PSUTIL_AVAILABLE:
            if peak_mem_rss > 0: psutil_metrics["peak_mem_rss_mb"] = round(peak_mem_rss / (1024 * 1024), 2)
            valid_cpu_p = [p for p in cpu_percents if p > 0.0] # cpu_percent() can return 0.0 on first few calls
            if valid_cpu_p: psutil_metrics["avg_cpu_percent"] = round(sum(valid_cpu_p) / len(valid_cpu_p), 2)
            elif cpu_percents: psutil_metrics["avg_cpu_percent"] = 0.0 # If all were 0
        return {**llama_metrics, **psutil_metrics}
    except subprocess.TimeoutExpired:
        print_error(f"llama.cpp run timed out for {model_path}.")
        if process_handle: process_handle.kill(); process_handle.communicate()
        return None
    except Exception as e:
        if process_handle: process_handle.kill(); process_handle.communicate()
        print_error(f"Exception in run_llama_cpp_benchmark for {model_path}: {type(e).__name__}: {e}")
        return None

# --- OLLAMA Engine Specific Functions ---
def check_ollama_cli():
    process = run_subprocess_command(["ollama", "--version"], capture_output=True)
    if process and process.returncode == 0:
        print_info(f"Ollama CLI found: {process.stdout.strip()}")
        return True
    print_error("Ollama CLI ('ollama') not found in PATH. Please install Ollama and ensure CLI is accessible.")
    return False

def check_ollama_server():
    if not REQUESTS_AVAILABLE:
        print_error("Ollama engine requires 'requests' package. Please install it (see top of script).")
        return False
    try:
        response = requests.get(OLLAMA_API_BASE_URL, timeout=3)
        response.raise_for_status()
        print_info(f"Ollama server found and responding at {OLLAMA_API_BASE_URL}")
        return True
    except requests.exceptions.RequestException:
        print_error(f"Ollama server not found or not responding at {OLLAMA_API_BASE_URL}. Is Ollama running? (e.g., 'ollama serve')")
        return False

def get_local_ollama_models():
    try:
        response = requests.get(f"{OLLAMA_API_BASE_URL}/api/tags", timeout=5)
        response.raise_for_status(); models_data = response.json()
        return {model['name'] for model in models_data.get('models', [])}
    except requests.exceptions.RequestException as e:
        print_warning(f"Could not retrieve local Ollama models: {e}")
        return set()

def pull_ollama_model(model_name):
    print_info(f"Attempting to pull Ollama model: {model_name} (this may take a while)...")
    # Using subprocess for 'ollama pull' to see its progress output directly.
    try:
        process = subprocess.Popen(["ollama", "pull", model_name], stdout=sys.stdout, stderr=sys.stderr)
        process.communicate() # Wait for pull to complete
        if process.returncode == 0:
            print_info(f"Successfully pulled Ollama model: {model_name}")
            return True
        else:
            print_error(f"Failed to pull Ollama model: {model_name}. Exit code: {process.returncode}")
            return False
    except FileNotFoundError:
        print_error("'ollama' command not found. Ensure Ollama CLI is installed and in your PATH.")
        return False
    except Exception as e:
        print_error(f"Error during 'ollama pull {model_name}': {e}")
        return False


def setup_ollama_environment():
    print_info("\n--- Setting up for Ollama ---")
    if not check_ollama_cli(): return False
    if not check_ollama_server(): return False
    if not OLLAMA_MODELS_TO_TEST:
        print_info("No models configured in OLLAMA_MODELS_TO_TEST.")
        return True

    local_models = get_local_ollama_models()
    all_models_available_for_test = True
    for model_name in OLLAMA_MODELS_TO_TEST:
        if model_name not in local_models:
            print_warning(f"Ollama model '{model_name}' not found locally.")
            user_input = input(f"Attempt to pull '{model_name}' with 'ollama pull'? (y/N): ").strip().lower()
            if user_input == 'y':
                if not pull_ollama_model(model_name):
                    all_models_available_for_test = False # Mark that at least one model failed to pull
                    print_warning(f"Will skip benchmarking for '{model_name}' due to pull failure.")
            else:
                all_models_available_for_test = False # User chose not to pull
                print_warning(f"Skipping benchmark for '{model_name}' as it's not available and not pulled.")
    
    if not all_models_available_for_test and any(m in local_models for m in OLLAMA_MODELS_TO_TEST):
        print_warning("Some configured Ollama models are not available. Benchmarking will proceed with available ones.")
    elif not any(m in local_models for m in OLLAMA_MODELS_TO_TEST): # No models to test at all
        print_error("No configured Ollama models are available for testing after setup attempts.")
        return False # Critical failure if no models end up being testable
    return True


def run_ollama_benchmark(model_name, prompt, n_predict):
    if not REQUESTS_AVAILABLE: return None
    api_url, payload = f"{OLLAMA_API_BASE_URL}/api/generate", {"model": model_name, "prompt": prompt, "stream": False, "options": {"num_predict": n_predict, "seed": 42}} # Added seed for consistency
    print_info(f"Running Ollama: model={model_name}, n_predict={n_predict}")
    try:
        start_time = time.time()
        response = requests.post(api_url, json=payload, timeout=600) # 10 min timeout
        response.raise_for_status(); data = response.json(); end_time = time.time()
        metrics = {
            "ollama_model_name_reported": data.get("model"), "wall_time_s": round(end_time - start_time, 3),
            "total_duration_ms": round(data.get("total_duration", 0) / 1_000_000, 3),
            "load_duration_ms": round(data.get("load_duration", 0) / 1_000_000, 3),
            "prompt_eval_count": data.get("prompt_eval_count"),
            "prompt_eval_duration_ms": round(data.get("prompt_eval_duration", 0) / 1_000_000, 3),
            "eval_count": data.get("eval_count"), "eval_duration_ms": round(data.get("eval_duration", 0) / 1_000_000, 3),
            "eval_tokens_per_sec": round((data.get("eval_count",0) / (data.get("eval_duration",1)/1_000_000_000)), 2) if data.get("eval_duration",0) > 0 else 0,
            "prompt_eval_tokens_per_sec": round((data.get("prompt_eval_count",0) / (data.get("prompt_eval_duration",1)/1_000_000_000)), 2) if data.get("prompt_eval_duration",0) > 0 else 0,
            "peak_mem_rss_mb": "N/A (Ollama)", "avg_cpu_percent": "N/A (Ollama)" # These are not directly measurable for Ollama server process from here
        }
        return metrics
    except requests.exceptions.Timeout: print_error(f"Timeout for Ollama model {model_name}.")
    except requests.exceptions.RequestException as e: print_error(f"Ollama API request failed for {model_name}: {e}")
    except Exception as e: print_error(f"Unexpected error for Ollama model {model_name}: {type(e).__name__}: {e}")
    return None

# --- Main Orchestration ---
def main():
    print_info("=== QuantVisor: LLM Performance Benchmarking Tool ===")
    if not check_and_guide_dependencies(): sys.exit(1)

    results = []
    csv_headers = [] # Will be determined by the first successful run of the chosen engine

    if BENCHMARK_ENGINE == "llama.cpp":
        print_info("\n--- Starting llama.cpp Benchmarks ---")
        if not setup_llama_cpp_environment():
            print_warning("llama.cpp environment setup incomplete. No llama.cpp benchmarks will run.")
        else:
            # Define expected headers for llama.cpp results
            csv_headers = [ "Engine", "Model Repo", "Quant File", "Threads", "Context Size", "Repetition", "Load Time (ms)", "Sample Time (ms)", "Sample Tokens", "Sample ms/token", "Sample Tokens/sec", "Prompt Eval Time (ms)", "Prompt Eval Tokens", "Prompt Eval ms/token", "Prompt Eval Tokens/sec", "Eval Time (ms)", "Eval Tokens", "Eval ms/token", "Eval Tokens/sec", "Total Time (ms)", "GGML Used Mem (MB)", "Wall Time (s)", "Peak Process MEM (RSS MB)", "Avg Process CPU %" ]
            models_ready_to_test_count = 0
            for repo_id, targets in LLAMA_CPP_MODELS_TO_DOWNLOAD_AND_TEST.items():
                for target_folder_name, gguf_files in targets.items():
                    for gguf_file in gguf_files:
                        model_full_path = Path(LLAMA_CPP_MODELS_BASE_DIR) / target_folder_name / gguf_file
                        if not model_full_path.exists():
                            print_warning(f"GGUF Model file {model_full_path} not found post-setup. Skipping.")
                            continue
                        models_ready_to_test_count +=1
                        for threads in LLAMA_CPP_THREADS_TO_TEST:
                            for ctx_size in LLAMA_CPP_CONTEXT_SIZES_TO_TEST:
                                print_info(f"\nTesting llama.cpp: {target_folder_name}/{gguf_file} | Thr: {threads} | Ctx: {ctx_size}")
                                for rep in range(1, REPETITIONS_PER_TEST + 1):
                                    print_info(f"Repetition {rep}/{REPETITIONS_PER_TEST}...")
                                    metrics = run_llama_cpp_benchmark(model_full_path, threads, ctx_size, BENCHMARK_PROMPT, TOKENS_TO_GENERATE)
                                    time.sleep(0.5)
                                    row = {"Engine": "llama.cpp", "Model Repo": target_folder_name, "Quant File": gguf_file, "Threads": threads, "Context Size": ctx_size, "Repetition": rep}
                                    if metrics: row.update(metrics)
                                    else: row["Eval Tokens/sec"] = "FAILED"
                                    results.append(row)
            if models_ready_to_test_count == 0: print_warning("No llama.cpp models were available to test after setup.")


    elif BENCHMARK_ENGINE == "ollama":
        print_info("\n--- Starting Ollama Benchmarks ---")
        if not setup_ollama_environment():
            print_warning("Ollama environment setup incomplete. No Ollama benchmarks will run.")
        else:
            csv_headers = ["Engine", "Ollama Model Name", "Repetition", "Generated Tokens (Eval Count)", "Load Duration (ms)", "Prompt Eval Count", "Prompt Eval Duration (ms)", "Prompt Eval Tokens/sec", "Eval Duration (ms)", "Eval Tokens/sec", "Total API Duration (ms)", "Wall Time (s)", "Ollama Model Name Reported", "Peak Process MEM (RSS MB)", "Avg Process CPU %" ]
            
            available_ollama_models = get_local_ollama_models() # Re-check available models
            models_to_actually_test = [m for m in OLLAMA_MODELS_TO_TEST if m in available_ollama_models]

            if not models_to_actually_test:
                 print_warning("No Ollama models specified in OLLAMA_MODELS_TO_TEST are available locally after setup.")
            else:
                print_info(f"Will benchmark the following Ollama models: {', '.join(models_to_actually_test)}")

            for model_name in models_to_actually_test:
                print_info(f"\nTesting Ollama: {model_name}")
                for rep in range(1, REPETITIONS_PER_TEST + 1):
                    print_info(f"Repetition {rep}/{REPETITIONS_PER_TEST}...")
                    metrics = run_ollama_benchmark(model_name, BENCHMARK_PROMPT, TOKENS_TO_GENERATE)
                    time.sleep(0.5)
                    row = {"Engine": "ollama", "Ollama Model Name": model_name, "Repetition": rep}
                    if metrics:
                        # Mapping Ollama specific metric names to CSV headers if they differ slightly or for clarity
                        row.update({
                            "Generated Tokens (Eval Count)": metrics.get("eval_count"),
                            "Load Duration (ms)": metrics.get("load_duration_ms"),
                            "Prompt Eval Count": metrics.get("prompt_eval_count"),
                            "Prompt Eval Duration (ms)": metrics.get("prompt_eval_duration_ms"),
                            "Prompt Eval Tokens/sec": metrics.get("prompt_eval_tokens_per_sec"),
                            "Eval Duration (ms)": metrics.get("eval_duration_ms"),
                            "Eval Tokens/sec": metrics.get("eval_tokens_per_sec"),
                            "Total API Duration (ms)": metrics.get("total_duration_ms"),
                            "Wall Time (s)": metrics.get("wall_time_s"),
                            "Ollama Model Name Reported": metrics.get("ollama_model_name_reported"),
                            "Peak Process MEM (RSS MB)": metrics.get("peak_mem_rss_mb"), # N/A
                            "Avg Process CPU %": metrics.get("avg_cpu_percent") # N/A
                        })
                    else: row["Eval Tokens/sec"] = "FAILED"
                    results.append(row)
    else:
        print_error(f"Invalid BENCHMARK_ENGINE: '{BENCHMARK_ENGINE}'. Choose 'llama.cpp' or 'ollama'.")
        sys.exit(1)

    if results:
        print_info(f"\nWriting results to {CSV_OUTPUT_FILE}...")
        # Use the pre-defined headers for the engine, or if empty, dynamically get from first result.
        final_headers = csv_headers if csv_headers else list(results[0].keys())
        try:
            with open(CSV_OUTPUT_FILE, "w", newline="", encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=final_headers, extrasaction='ignore')
                writer.writeheader()
                for row_data in results: writer.writerow(row_data)
            print_info(f"Benchmarking complete. Results saved to {CSV_OUTPUT_FILE}")
        except IOError as e:
            print_error(f"Error writing CSV file {CSV_OUTPUT_FILE}: {e}.")
            print_info("Dumping results to console as fallback:")
            if final_headers: print(",".join(final_headers))
            for row_data in results: print(",".join(str(row_data.get(header, "")) for header in final_headers))
    else:
        print_warning("No benchmark results were generated.")
    print_info("\nQuantVisor finished.")

if __name__ == "__main__":
    main()