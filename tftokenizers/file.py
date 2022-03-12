import glob
import json
import pathlib
from enum import Enum
from io import TextIOWrapper
from typing import List, Tuple


class FileTypes(str, Enum):
    TXT = "txt"
    JSON = "json"
    NONE = "None"


def get_filename_from_path(path: str, filename: str) -> str:
    """Find a file name prefix in a folder."""
    if filename in path:
        path_without_filename = "/".join(path.split("/")[:-1])
        path = path_without_filename

    for file_path in glob.glob(f"{path}/*"):
        file = file_path.split("/")[-1]

        if filename in file and filename not in path:
            filename = file
            break

    file_path = f"{path}/{filename}"
    return file_path


def get_filename_and_type_from_path(path: str, filename: str) -> Tuple[str, str]:
    """Find a file name prefix in a folder."""
    file_type = "None"

    if filename in path:
        path_without_filename = "/".join(path.split("/")[:-1])
        path = path_without_filename

    for file_path in glob.glob(f"{path}/*"):
        file = file_path.split("/")[-1]
        file_type = file.split(".")[-1]

        if filename in file and filename not in path:
            filename = file
            break

    file_path = f"{path}/{filename}"
    return file_path, file_type


def get_vocab_from_path(path: str) -> List[str]:

    vocab: List[str]
    filename = "vocab"
    file_path, filetype = get_filename_and_type_from_path(path, filename)

    if filetype == FileTypes.TXT:
        vocab = parse_and_load_txt_vocab(file_path)
    elif filetype == FileTypes.JSON:
        vocab = parse_and_load_json_vocab(file_path)
    else:
        raise ValueError(f"Could not parse the given file/path: {path}")
    return vocab


def load_txt(file_path: str):
    """Loads txt file from path."""
    data = pathlib.Path(file_path).read_text().splitlines()
    return data


def load_json(file_path: str):
    """Loads json file from path."""
    with open(file_path) as f:
        data = json.load(f)
    return data


def save_json(data: dict, f: TextIOWrapper):
    return json.dump(data, f, ensure_ascii=False, indent=4)


def parse_and_load_txt_vocab(file_path: str) -> List[str]:
    """Parses tokenizer vocab from a txt file."""
    return load_txt(file_path)


def parse_and_load_json_vocab(file_path: str) -> List[str]:
    """Parses tokenizer vocab from a json file."""
    data = load_json(file_path)
    return [key for key in data]
