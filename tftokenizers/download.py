import argparse
import logging

from rich.logging import RichHandler
from transformers import AutoTokenizer

from tftokenizers.database import save_tokenizer_attributes
from tftokenizers.types import TokenizerAttributes
from tftokenizers.utils import parse_args_to_keywords


def parse_args():
    parser = argparse.ArgumentParser(
        description="Save a huggingface tokenizer for conversion to a Tensorflow Text tokenizer."
    )
    parser.add_argument(
        "--name_or_path",
        "-n",
        type=str,
        help="Name to pretrained model or tokenizer from huggingface.co/models.",
        default="bert-base-uncased",
        # required=True
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="saved_tokenizers",
        help="Where to store the Huffingface tokenizer configs.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Display debug info if passed in as argument",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    logging.basicConfig(
        format="%(message)s",
        datefmt="[%X]",
        level="NOTSET" if args.verbose else logging.INFO,
        handlers=[RichHandler(rich_tracebacks=True)],
    )

    log = logging.getLogger("rich")
    log.info("Converting tokenizer...")

    name, path = parse_args_to_keywords(args)
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.save_pretrained(path)
    log.info(f"Saved tokenizer to: {path}")

    attributes = TokenizerAttributes(
        name_or_path=name,
        export_path=path,
    )
    save_tokenizer_attributes(attributes)
    log.info("Saving a download script")


if __name__ == "__main__":
    main()
