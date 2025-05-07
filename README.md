Howdy! README for QuantVisor
============================

So, you got QuantVisor, huh? Good on ya! This here script is built to help you figure out how fast them fancy AI language models run on your own rig, whether you're using llama.cpp or Ollama. It's pretty straightforward, kinda like findin' your way to the coast once you hit 99W.

Gettin' Set Up:
----------------

1.  **Make a Spot for It (Your QuantVisor Folder):**
    First thing, you gotta make a folder for QuantVisor to live in. Just create a new folder anywhere you like – maybe on your Desktop or in your Documents. Call it `QuantVisor`. Easy peasy.

2.  **Put the Script in There:**
    *   Take all that Python code for `quantvisor.py` I gave ya.
    *   Open up a plain ol' text editor. If you're on Windows, Notepad'll do. Mac folks, TextEdit is fine, just make sure it's in "Plain Text" mode (Format > Make Plain Text). If you're fancy, VS Code or somethin' similar works too.
    *   Paste all that code right into the blank file.
    *   Now, save it. Name it exactly `quantvisor.py` (all lowercase is good) and stick it right inside that `QuantVisor` folder you just made.

    When you're done, your `QuantVisor` folder should look like this:
    QuantVisor/
        └── quantvisor.py

    If you end up usin' the `llama.cpp` part, the script will make another folder called `models_gguf` inside `QuantVisor` for all the model files. Don't you worry 'bout that part yet.

How to Kick It Off (Runnin' the Script):
----------------------------------------

1.  **Open Your Terminal (Command Line Thingy):**
    *   **Windows:** Hit the Start button and type `PowerShell`, then open it. `Command Prompt` works too.
    *   **Mac or Linux:** Look for an app called `Terminal` and open that up.

2.  **Head to Your QuantVisor Folder:**
    You gotta tell the terminal where you put that `QuantVisor` folder. Use the `cd` command (that means "change directory").
    *   If it's on your Desktop:
        *   Mac/Linux: `cd ~/Desktop/QuantVisor`
        *   Windows (PowerShell): `cd $HOME\Desktop\QuantVisor`
        *   Windows (Command Prompt): `cd %USERPROFILE%\Desktop\QuantVisor`
    *   If it's somewhere else, just change the path to wherever you stuck it.

3.  **Run the Dang Thing!**
    Once you're "in" the `QuantVisor` folder in your terminal, type this and hit Enter:
    `python3 quantvisor.py`

    If `python3` gives you grief, try just `python quantvisor.py`. Just make sure you got Python version 3.6 or newer installed on your machine.

First Time Runnin' It - Important Stuff:
----------------------------------------

When you run it the first time, it might need a few things:

*   **Python Helper Bits (Dependencies):**
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