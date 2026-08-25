"""Execute the training notebook and save execution outputs directly into the .ipynb file."""

import json
from pathlib import Path
import nbformat
from nbclient import NotebookClient

def run_notebook() -> None:
    nb_path = Path("notebooks/train_round_winner_models.ipynb")
    print(f"Reading notebook: {nb_path}")
    with nb_path.open("r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    client = NotebookClient(nb, timeout=3600, kernel_name="python3")
    print("Executing notebook cells...")
    client.execute()
    print("Notebook executed successfully. Saving output...")

    with nb_path.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print("Notebook successfully updated with execution outputs!")

if __name__ == "__main__":
    run_notebook()
