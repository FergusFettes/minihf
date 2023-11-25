#!/usr/bin/env python

import json
import zipfile
import random
from pathlib import Path

import typer

app = typer.Typer()


# Class to handle a zipped conversation dataset
class ZippedConversationsDataset:
    """
    Dataset class to lazily load conversations from a ZIP file.
    """

    def __init__(self, zip_file: Path, debug: bool = False):
        self.debug = debug
        self.training_items = []
        self.load_from_zip(zip_file)

    def load_from_zip(self, zip_file: Path):
        zip_ = zipfile.ZipFile(zip_file)
        for file_ in zip_.namelist():
            if self.debug:
                print(f"Loading file {file_}")
            if file_.endswith("/") or file_.startswith("__MACOSX"):
                continue
            with zip_.open(file_) as infile:
                training_item = self.process_file(file_, infile)
                if training_item:
                    self.training_items.append(training_item)
        random.shuffle(self.training_items)

    def process_file(self, file_name: str, file_content):
        """
        Processes a file within the ZIP file depending on the extension.
        """
        if file_name.endswith(".txt"):
            return file_content.read().decode('UTF-8')
        elif file_name.endswith(".json"):
            conversation = json.load(file_content)
            return self.extract_text_from_conversation(conversation)

    def extract_text_from_conversation(self, conversation: dict):
        """
        Extract and return text from conversation JSON dict.
        """
        for id_ in conversation["responseDict"]:
            branch = conversation["responseDict"][id_]
            if branch["rating"]:  # if True
                return branch["prompt"] + branch["text"]
        return None

    def __len__(self):
        return len(self.training_items)

    def __next__(self):
        return random.sample(self.training_items, 1)[0]


@app.command()
def main(
    dataset_path: Path = typer.Argument(
        ..., help="Path to the dataset ZIP file."
    ),
    show_n: int = typer.Option(
        10, help="Number of examples to show.", show_default=True
    ),
    # flag
    debug: bool = typer.Option(False, help="Show debug information.")
):
    """
    Opens the dataset and displays a few examples.
    """
    dataset = ZippedConversationsDataset(dataset_path, debug=debug)
    for _ in range(show_n):
        print(next(dataset))


if __name__ == "__main__":
    app()
