import os
import subprocess
import sys

def setup_environment():
    try:
        print("Installing spaCy English model...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        
        print("Creating necessary directories...")
        os.makedirs("data/pdfs", exist_ok=True)
        os.makedirs("data/outputs", exist_ok=True)
        os.makedirs("model", exist_ok=True)
        os.makedirs("utils", exist_ok=True)
        os.makedirs("pipeline", exist_ok=True)
        
        print("Setup complete!")
        
    except Exception as e:
        print(f"Setup error: {e}")

if __name__ == "__main__":
    setup_environment()