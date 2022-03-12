"""Help identify models, class type between a Huggingface tokenizer and Tensorflow implementation."""

import tensorflow as tf
import tensorflow_text as text
from transformers import PreTrainedTokenizer


def get_tokenizer_type(model_name_or_path, path, **kwargs):
    pass


# TODO: could be enum type
def find_tf_base_tokenizer(tokenizer_type, **kwargs):
    pass


def load_tokenizer(tokenizer_tf_base, tokenizer_config_params, **kwargs):
    pass


def detect_and_load_tokenizer(tokenizer: PreTrainedTokenizer, path: str, **kwargs):
    # TODO: Should load tokenizer based on `tokenizer.backend_tokenizer.model`
    """Load the corresponging base tokenizer from Huggingaface and map it to a Tensorflow model.

    We set lower_case=False, since we want the tokenizer vocabulary to be similarly cased to Huggingface
    """
    vocab_file = tokenizer.vocab_files_names["vocab_file"]
    vocab_path = f"{path}/{vocab_file}"

    """
    (vocab_lookup_table: Unknown,
    suffix_indicator: Unknown = "##",
    max_bytes_per_word: Unknown = 100,
    max_chars_per_token: Unknown = None,
    token_out_type: Unknown = dtypes.int64,
    unknown_token: Unknown = "[UNK]",
    split_unknown_characters: Unknown = False,
    lower_case: Unknown = False,
    keep_whitespace: Unknown = False,
    normalization_form: Unknown = None,
    preserve_unused_token: Unknown = False,
    basic_tokenizer_class: Unknown = BasicTokenizer) -> None
    """
    return text.BertTokenizer(
        vocab_path,
        token_out_type=tf.int64,
        lower_case=tokenizer.do_lower_case,
        **kwargs,
    )
