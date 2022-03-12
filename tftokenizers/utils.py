from typing import Any, Dict, List, Optional, Tuple

import tensorflow as tf

from tftokenizers.types import PaddingStrategies, TemplateType


def map_special_tokens_to_ids(
    vocab: List[str], special_tokens_or_config: Dict[str, str]
) -> Dict[str, tf.Tensor]:
    """Map special tokens ([SEP]) to their token id as a tensor."""
    special_tokens_map: Dict[str, tf.Tensor] = {}

    for name, symbol in special_tokens_or_config.items():
        if "_token" not in name or "_tokens" in name:
            continue
        if isinstance(symbol, str):
            _token_id = vocab.index(symbol)
            token_id: tf.Tensor = to_tensor(_token_id)
            special_tokens_map[symbol] = token_id

    return special_tokens_map


def num_special_tokens(proceesing_type: TemplateType, config: Dict[str, Any]):
    r"""Sets the number of special tokens when processing a sentence.

    Configuration is based on the `post_processor` template from \
        the configurations in Huggingface Tokenizers, \
        The template specifies which tokens and in what order \
        special tokens should be added to a single or pair os sentences.
    """

    assert proceesing_type in ["single", "pair"]

    template: List[Dict[str, Any]] = config["post_processor"][proceesing_type]
    return len([s for s in template if s.get("SpecialToken") is not None])


def parse_args_to_keywords(args) -> Tuple[str, str]:
    tokenizer_name: str = args.name_or_path
    save_name: str = tokenizer_name.replace("/", "_")
    path = f"{args.output_dir}/{save_name}"
    return tokenizer_name, path


def to_tensor(x: int) -> tf.Tensor:
    return tf.constant(x, dtype=tf.int64)


def list_to_tensor(lst: List[int]):
    return [to_tensor(num) for num in lst]


def set_valid_max_seq_length(
    desired=None, model_max_length=512, num_max_tokens_in_seq=510
) -> int:
    """Calculate and set the maximum allowed sequence length."""
    min_length = model_max_length - num_max_tokens_in_seq
    max_length = num_max_tokens_in_seq

    if desired is None or desired > max_length or desired <= min_length:
        desired = max_length
    return desired


def set_valid_padding(padding: Optional[PaddingStrategies]) -> PaddingStrategies:
    padding = padding if padding is not None else PaddingStrategies.LONGEST
    return padding
