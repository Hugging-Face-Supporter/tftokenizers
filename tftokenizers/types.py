from enum import Enum
from typing import Any, Dict, Tuple, Union

import tensorflow as tf
from pydantic import BaseModel

SPECIAL_TOKENS_DICT = Dict[str, Dict[str, Union[str, int, Any]]]
PROCESSING_STEP = Dict[str, Dict[str, str]]
TF_TENSORS = Union[tf.RaggedTensor, tf.SparseTensor, tf.IndexedSlices, tf.Tensor, Any]
TF_TOKENIZER_INPUT = Tuple[TF_TENSORS, TF_TENSORS]


class PaddingStrategies(str, Enum):
    """Define how input in the tokenizer should be padded."""

    MAX_LENGTH = "max_length"
    LONGEST = "longest"
    DO_NOT_PAD = "do_not_pad"
    NONE = "None"


class SpecialTokens(str, Enum):
    """Name of special tokens a particular tokenizer may have."""

    UNK_TOKEN = "unk_token"
    END_TOKEN = "eos_token"
    PAD_TOKEN = "pad_token"
    START_TOKEN = "bos_token"
    SEP_TOKEN = "sep_token"
    CLS_TOKEN = "cls_token"
    MASK_TOKEN = "mask_token"


class TemplateType(str, Enum):
    """Options for how to process a `post_processor` template."""

    SINGLE = "single"
    PAIR = "pair"


class TokenizerAttributes(BaseModel):
    name_or_path: str
    export_path: str


class NotValidPaddingStrategy(Exception):
    """Exception raised for not supported padding strategy.

    padding strategy for a tokenizer, currently supports strategies from `PaddingStrategies`:class
    """

    def __init__(
        self,
        padding,
        message="The padding strategy chosen is not supported. \
            Use one listed in the class `PaddingStrategies`.",
    ):
        self.strategy = padding
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"Chosen padding strategy: {self.strategy}\n{self.message}"
