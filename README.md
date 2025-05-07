Howdy! README for QuantVisor
============================

So, you got QuantVisor, huh? Good on ya! This here script is built to help you figure out how fast them fancy AI language models run on your own rig, whether you're using llama.cpp or Ollama. It's pretty straightforward, kinda like findin' your way to the coast once you hit 99W.

Gettin' Set Up:
----------------

Download or clone this repository, or just download the script itself, quantvisor.py

1.  **Open a Terminal in the Location of Your Script:**
    Use your context menu or navigate via commanad line.

2.  **Run It!**
    Once you're in the `QuantVisor` folder in your terminal, type the following and hit Enter:
    `python3 quantvisor.py`

    If `python3` gives you grief, try just `python quantvisor.py` or vice versa. Just make sure you got Python version 3.6 or newer installed on your machine.

First Time Runnin' It - Important Stuff:
----------------------------------------

When you run it the first time, it might need a few things:

*   **Python Bits (Dependencies):**
    The script uses a few extra Python tools like `psutil`, `requests`, and `huggingface-hub`. If you don't have 'em, the script will tell ya. It'll even show you the commands to type into your terminal to install 'em, somethin' like `python3 -m pip install psutil`. Just run those install commands, then try runnin' `quantvisor.py` again.

*   **Pick Your Poison (Benchmark Engine):**
    You gotta tell QuantVisor if you wanna test `llama.cpp` models or `Ollama` models.
    *   Open up that `quantvisor.py` file again with your text editor.
    *   Near the top, you'll see a line like `BENCHMARK_ENGINE = "ollama"` (or it might say `"llama.cpp"`).
    *   Change the part in the quotes to either `"llama.cpp"` or `"ollama"`, dependin' on what you're testin'.
    *   Save the file!

*   **If You Picked `llama.cpp`:**
    *   **`LLAMA_CPP_EXECUTABLE` (The Program Itself):** You need a compiled `llama.cpp` program (usually a file called `main` or `main.exe`). The script will try to find it as `./main`. If it can't, or the one it finds won't run, it'll ask you to type in the full path to where that program is. (Like, `C:\MyStuff\llama.cpp\main.exe` or `/home/yourname/llama.cpp/main`).
    *   **Model Downloads:** The script will try to download the AI models (them GGUF files) it needs from the internet (Hugging Face). This can take a bit, especially if your internet's like tryin' to get through Salem at 5 PM. If a model needs a login, you might need to have used `huggingface-cli login` in your terminal beforehand. These models go into that `models_gguf` folder.

*   **If You Picked `Ollama`:**
    *   **Ollama Installed?:** Make sure you've actually installed Ollama on your computer and that the `ollama` command works in your terminal.
    *   **Ollama Runnin'?:** The Ollama app or server needs to be up and runnin' in the background. Usually, you just start the Ollama desktop app, or type `ollama serve` in a separate terminal. The script checks if it can talk to it at `http://localhost:11434`.
    *   **Gettin' Ollama Models:** For each model you told it to test (in the `OLLAMA_MODELS_TO_TEST` list in the script), if it ain't already on your machine, QuantVisor will ask if you want it to try and `ollama pull` it for ya. This also needs internet and can take a spell.

What You Get (The Output):
--------------------------
As it's runnin', you'll see stuff print out in your terminal. When it's all done, it'll save all the results in a CSV file (that's a spreadsheet file) right there in your `QuantVisor` folder. It'll be named somethin' like `quantvisor_benchmark_results_YYYYMMDD_HHMMSS.csv`. You can open that up with Excel, Google Sheets, LibreOffice Calc, or whatever spreadsheet tool you got.

That's about the long and short of it. Hope it helps you wrangle them AI models! If somethin's not makin' sense, just take it slow and read what the script tells ya in the terminal.

Cheers!
