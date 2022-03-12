import json
import os
from pathlib import Path
from typing import Dict, List

from tftokenizers.file import save_json
from tftokenizers.types import TokenizerAttributes

DATABASE_PATH = Path("database.json")
TOKENIZER_ENTRY = "tokenizers"
TOKENIZER_NAME_ATTR = "name_or_path"
EXECUTE_PERMISSION = 0o0755


def create_database(database_path=DATABASE_PATH) -> None:
    """Save the files of a Huggingface tokenizer locally."""
    database: Dict[str, List[str]] = {TOKENIZER_ENTRY: []}
    with open(database_path, "w+", encoding="utf-8") as f:
        save_json(database, f)


def get_database():
    with open(DATABASE_PATH, "r+", encoding="utf-8") as f:
        data = json.load(f)
    return data


def insert_database_entry(attributes: TokenizerAttributes):
    database: dict = get_database()
    d = database[TOKENIZER_ENTRY]
    new_entry = attributes.dict()
    is_found = False

    for idx, entry in enumerate(d):
        if entry[TOKENIZER_NAME_ATTR] == new_entry[TOKENIZER_NAME_ATTR]:
            is_found = True
            database[TOKENIZER_ENTRY][idx] = new_entry
            break

    if not is_found:
        database[TOKENIZER_ENTRY].append(new_entry)

    return database


def save_tokenizer_attributes(attributes: TokenizerAttributes):
    if not os.path.isfile(DATABASE_PATH):
        create_database()

    with open(DATABASE_PATH, "r+", encoding="utf-8") as f:
        # data = json.load(f)
        data = insert_database_entry(attributes)
        f.seek(0)
        save_json(data, f)


def create_download_script():
    """Create bash script to download all tokenizers in database."""
    database = get_database()

    SCRIPT_NAME = "scripts/download_tokenizers.sh"
    MODULE = "tftokenizers"
    FILE = "download.py"

    with open(SCRIPT_NAME, "w") as f:
        f.writelines("#!/usr/bin/env bash\n\n")

        for entry in database[TOKENIZER_ENTRY]:
            tokenizer_name = entry[TOKENIZER_NAME_ATTR]
            f.writelines(f"python {MODULE}/{FILE} -n {tokenizer_name}\n")

    os.chmod(SCRIPT_NAME, EXECUTE_PERMISSION)
