import os
import pathlib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

list_of_files = list(set([
    "src/__init__.py",
    "src/helper.py",
    "src/prompts.py",
    ".env",
    "setup.py",
    "app.py",
    "research/trials.ipynb",
    "requirements.txt",
    "Dockerfile",
    "Dockerfile",
]))

for filepath in list_of_files:
    filepath = pathlib.Path(filepath)
    filedir = os.path.dirname(filepath)

    # Create directory if not exists
    if filedir and not os.path.exists(filedir):
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Created directory: {filedir}")

    # Create or verify file
    if not os.path.exists(filepath):
        with open(filepath, "w") as f:
            pass
        logging.info(f"Created new file: {filepath}")
    elif os.path.getsize(filepath) == 0:
        logging.info(f"File already exists but is empty: {filepath}")
    else:
        logging.info(f"File already exists and is not empty: {filepath}")
