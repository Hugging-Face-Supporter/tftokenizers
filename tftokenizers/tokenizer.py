import re
from typing import Any, Dict, List, Optional, Union

import tensorflow as tf
import tensorflow_text as text
from transformers import AutoTokenizer, PreTrainedTokenizer

from tftokenizers.detect import detect_and_load_tokenizer
from tftokenizers.file import get_vocab_from_path, load_json
from tftokenizers.types import (
    PROCESSING_STEP,
    TF_TENSORS,
    TF_TOKENIZER_INPUT,
    NotValidPaddingStrategy,
    PaddingStrategies,
    SpecialTokens,
    TemplateType,
)
from tftokenizers.utils import (
    list_to_tensor,
    map_special_tokens_to_ids,
    num_special_tokens,
    set_valid_max_seq_length,
    set_valid_padding,
)


def compare_tokenizers(pretrained_model_name: str, inputs: Union[str, List[str]]):
    """Helper function for comparing TF and HF tokenizers."""

    tf_tokenizer = TFAutoTokenizer.from_pretrained(pretrained_model_name)
    hf_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    tf_tokens = tf_tokenizer.batch_encode_plus(inputs)
    hf_tokens = hf_tokenizer.batch_encode_plus(
        inputs,
        padding=True,
        truncation=True,
        return_token_type_ids=False,
        return_tensors="tf",
    )

    return {"tf_tokens": tf_tokens, "hf_tokens": hf_tokens}


def save_and_load_tokenizer_to_path(
    tokenizer: PreTrainedTokenizer,
    model_name_or_path: str,
    output_dir="saved_tokenizers",
):
    path = model_name_or_path
    if output_dir not in model_name_or_path:
        path = f"{output_dir}/{model_name_or_path}"
        tokenizer.save_pretrained(path)
    return path


class TFAutoTokenizer(tf.Module):
    def __init__(
        self,
        model_name_or_path,
        max_length: Optional[int] = None,
        padding: Optional[PaddingStrategies] = None,
        output_dir="saved_tokenizers",
        tokenizer_config_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super(TFAutoTokenizer, self).__init__(**kwargs)
        config: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        path = save_and_load_tokenizer_to_path(config, model_name_or_path, output_dir)
        pipeline = load_json(f"{path}/tokenizer.json")

        self.tokenizer_file = config.vocab_files_names["tokenizer_file"]
        self.vocab_file = config.vocab_files_names["vocab_file"]
        # TODO: `vocab_file` Could be vocab or SentencePieice model
        vocab_path = f"{path}/{self.vocab_file}"
        self.vocab_path = vocab_path
        self.vocab = get_vocab_from_path(vocab_path)
        self.vocab_size = config.vocab_size
        self.vocab_dict = config.vocab
        self._vocab = tf.Variable(self.vocab)

        # TODO: Load in multiple steps
        self.tokenizer = detect_and_load_tokenizer(config, path)
        # self.tokenizer_type = get_tokenizer_type(model_name_or_path, path)
        # self.tokenizer_tf_base = find_tf_base_tokenizer(self.tokenizer_type)
        # self.tokenizer = load_tokenizer(
        #     self.tokenizer_tf_base, tokenizer_config_params, **kwargs
        # )

        self.special_tokens = map_special_tokens_to_ids(
            self.vocab, load_json(f"{path}/tokenizer_config.json")
        )
        self.all_special_tokens = config.all_special_tokens
        self.all_special_ids = config.all_special_ids
        self.all_special_ids_tf = list_to_tensor(config.all_special_ids)

        # TODO: Should be determined by the tokenizer config
        # POtential solution could be:
        # $ c = tokenizer.backend_tokenizer.post_processor
        # $ sentences_pair  = model_max_length - c.num_special_tokens_to_add(is_pair=False)
        # $ single_sentence = model_max_length - c.num_special_tokens_to_add(is_pair=True)
        self.max_len_sentences_pair = 509
        self.max_len_single_sentence = 510
        self.model_max_length = max_length if max_length is not None else 510

        self.model_input_names = config.model_input_names
        self.name_or_path = config.name_or_path
        self.saved_at_path = path

        # Tokenizer pipeline
        self.decoder = pipeline["decoder"]
        self.model = pipeline["model"]
        self.normalizer = pipeline["normalizer"]
        self.post_processor = pipeline["post_processor"]
        self.pre_tokenizer = pipeline["pre_tokenizer"]

        self.decoder_type = pipeline["decoder"]["type"]
        self.model_type = pipeline["model"]["type"]
        self.normalizer_type = pipeline["normalizer"]["type"]
        self.post_processor_type = pipeline["post_processor"]["type"]
        self.pre_tokenizer = pipeline["pre_tokenizer"]["type"]

        self.padding = padding if padding is not None else PaddingStrategies.LONGEST
        # self.max_length = (
        #    max_length if max_length is not None else self.model_max_length
        # )
        self.max_length = 512
        self._reserved_tokens = list(self.special_tokens.keys())
        self._vocab_path = tf.saved_model.Asset(self.vocab_path)

        # Include a tokenize signature for a batch of strings.
        self.tokenize.get_concrete_function(
            inputs=tf.TensorSpec(shape=[None], dtype=tf.string),
        )

        # Include `detokenize` and `lookup` signatures for:
        #   * `Tensors` with shapes [tokens] and [batch, tokens]
        #   * `RaggedTensors` with shape [batch, tokens]
        self.detokenize.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int64)
        )
        self.detokenize.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64)
        )

        self.lookup.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int64)
        )
        self.lookup.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64)
        )

        # These `get_*` methods take no arguments
        self.get_vocab_size.get_concrete_function()
        self.get_vocab_path.get_concrete_function()
        self.get_reserved_tokens.get_concrete_function()

    @classmethod
    def from_pretrained(cls, model_name_or_path, *model_args, **kwargs):
        """Load tokenizer in same expected behaviour as Huggingface tokenizers."""
        tokenizer = cls(model_name_or_path, *model_args, **kwargs)
        return tokenizer

    @tf.function
    def tokenize(
        self, inputs, max_length=None, padding: Optional[PaddingStrategies] = None
    ):
        self.max_length = set_valid_max_seq_length(
            max_length, self.model_max_length, self.max_len_single_sentence
        )
        self.padding = set_valid_padding(padding)

        input_ids = self.tokenizer.tokenize(inputs)
        # Merge the `word` and `word-piece` axes.
        input_ids = input_ids.merge_dims(-2, -1)
        input_ids = self.truncate(input_ids)
        input_ids = self.post_process(input_ids)
        input_ids, mask = self.pad(input_ids)
        return {"input_ids": input_ids, "attention_mask": mask}

    def truncate(self, sequence: tf.RaggedTensor):
        """Truncation removes a sequence up to a specified lenght.

        Special tokens marking the start and end of a sequence (if specified) \
            will also be included in the sentence.
        """
        max_length = self.max_length
        return sequence[:, :max_length]

    def pad(
        self,
        sequence: tf.RaggedTensor,
    ) -> TF_TOKENIZER_INPUT:

        r"""
        Add padding of a sequence and supports different strategies for doing so.

        ::note: passing in padding will have no effect when loading a saved tokenizer.

        `strategy` supports different options:
            - `max_length`: Pad all batches to tokenizer `max_seq_length`.
            - `longest`: Pad all to same length as the longest sequence in the batch.
            - `none`: Do not pad at all.
        """
        padding = self.padding
        max_length = self.max_length
        _input_ids, _attention_mask = TF_TENSORS, TF_TENSORS  # type: ignore

        if padding == PaddingStrategies.MAX_LENGTH:
            _input_ids, _attention_mask = text.pad_model_inputs(
                input=sequence, max_seq_length=max_length
            )
        elif padding == PaddingStrategies.LONGEST:
            seq_len_per_batch: tf.Tensor = sequence.nested_row_lengths()[0]
            longest_seq_in_batch = tf.reduce_max(seq_len_per_batch)
            _input_ids, _attention_mask = text.pad_model_inputs(
                input=sequence, max_seq_length=longest_seq_in_batch
            )
        elif padding == PaddingStrategies.NONE:
            _input_ids: tf.RaggedTensor = sequence
            _attention_mask: tf.RaggedTensor = tf.ones_like(sequence)
        else:
            raise NotValidPaddingStrategy(padding=padding)

        # Since Huggingface tensors are of same type
        input_ids = tf.cast(_input_ids, dtype=tf.int64)
        attention_mask = tf.cast(_attention_mask, dtype=tf.int64)
        return input_ids, attention_mask

    @tf.function
    def post_process(self, bs: tf.RaggedTensor) -> tf.RaggedTensor:
        r"""
        Specifies how a sentence should be constructed together with a set of special tokens.

        Huggingface tokenizers specify a template for how different post_processors should behave \
            when adding special tokens to a "single" or "pair" of sentences.

        An example template parsing instruction for BERT could look like:

            >>> from tftokenizers.load import load_json
            >>> config = load_json("saved_tokenizers/bert-base-uncased/tokenizer.json")
            >>> config["post_process"]["pair"]
            \"""
            [{'SpecialToken': {'id': '[CLS]', 'type_id': 0}},
            {'Sequence': {'id': 'A', 'type_id': 0}},
            {'SpecialToken': {'id': '[SEP]', 'type_id': 0}},
            {'Sequence': {'id': 'B', 'type_id': 1}},
            {'SpecialToken': {'id': '[SEP]', 'type_id': 1}}]
            \"""

        Each type of tokenizer can have different templates between the models. \
            This function parses the template into a simple abstract syntax tree \
            and combines the original sentence with the special tokens according \
            according to the template.
        """

        num_sentences = bs.bounding_shape()[0]
        template = self.post_processor
        processing_steps = template[TemplateType.SINGLE]
        num_steps = len(processing_steps)

        def get_token_id_from_current_step_in_template(
            instruction_step: PROCESSING_STEP,
        ) -> tf.Tensor:
            token_symbol: str = list(instruction_step.values())[0]["id"]
            token_id: tf.Tensor = self.special_tokens.get(token_symbol)
            # idx = self.all_special_tokens.index(token_symbol)
            # token_id = self.all_special_ids_tf[idx]
            return token_id

        # Build AST from template instruction
        for step in range(num_steps - 1):
            left: PROCESSING_STEP = processing_steps[step]
            right: PROCESSING_STEP = processing_steps[step + 1 - num_steps]
            left_token = get_token_id_from_current_step_in_template(left)
            right_token = get_token_id_from_current_step_in_template(right)

            if left != right:
                left_sentence = bs
                right_sentence = bs

                if "SpecialToken" in left:
                    left_sentence = tf.fill([num_sentences, 1], left_token)
                elif "SpecialToken" in right:
                    right_sentence = tf.fill([num_sentences, 1], right_token)
                bs = tf.concat([left_sentence, right_sentence], axis=1)
        return bs

    def cleanup_text(self, token_txt):
        bad_tokens = [
            re.escape(tok)
            for tok in self._reserved_tokens
            if tok != SpecialTokens.UNK_TOKEN.value
        ]
        bad_token_re = "|".join(bad_tokens)

        bad_cells = tf.strings.regex_full_match(token_txt, bad_token_re)
        result = tf.ragged.boolean_mask(token_txt, ~bad_cells)

        # Join them into strings.
        result = tf.strings.reduce_join(result, separator=" ", axis=-1)
        return result

    @tf.function
    def detokenize(self, tokenized):
        words = self.tokenizer.detokenize(tokenized)
        return self.cleanup_text(words)

    @tf.function
    def lookup(self, token_ids):
        return tf.gather(self._vocab, token_ids)

    @tf.function
    def get_vocab_size(self):
        return tf.shape(self._vocab)[0]

    @tf.function
    def get_vocab_path(self):
        return self._vocab_path

    @tf.function
    def get_reserved_tokens(self):
        return tf.constant(self._reserved_tokens)

    @tf.function
    def _set_valid_max_seq_length(
        self, desired=None, model_max_length=512, num_max_tokens_in_seq=510
    ) -> int:
        """Calculate and set the maximum allowed sequence length."""
        min_length = model_max_length - num_max_tokens_in_seq
        max_length = num_max_tokens_in_seq

        if desired is None or desired > max_length or desired <= min_length:
            desired = max_length
        return desired

    @tf.function
    def _set_valid_padding(self, padding: PaddingStrategies) -> PaddingStrategies:
        padding = padding if padding is not None else PaddingStrategies.LONGEST
        return padding

    @tf.function
    def get_max_length(self):
        return self.max_length

    # New methods with names similar to
    # The Huggingface Tokenizer interface
    @tf.function
    def encode(self, strings):
        return self.tokenize(strings)

    @tf.function
    def batch_encode(self, strings):
        return self.tokenize(strings)

    @tf.function
    def batch_encode_plus(self, strings):
        return self.tokenize(strings)

    @tf.function
    def decode(self, tokenized):
        return self.detokenize(tokenized)

    @tf.function
    def batch_decode(self, tokenized):
        return self.detokenize(tokenized)


class TFTokenizerBase(tf.Module):
    # TODO: Will move to deprecate this class
    # Was based on taking in a tokenizer to start
    def __init__(
        self,
        vocab_path: str,
        tokenizer_base: text.Tokenizer,
        hf_spec: Dict[str, Any],
        config: Dict[str, Any],
        max_length: Optional[int] = None,
        padding: Optional[PaddingStrategies] = None,
    ):
        super(TFTokenizerBase, self).__init__()

        self.tokenizer = tokenizer_base
        self.config = config
        vocab = get_vocab_from_path(vocab_path)
        self.vocab_size = len(vocab)
        self.vocab = tf.Variable(vocab)

        self.model_max_length = 510  # config["model_max_length"]
        self.max_length = (
            max_length if max_length is not None else self.model_max_length
        )
        self.padding = padding if padding is not None else PaddingStrategies.LONGEST
        self.tokenizer_spec = hf_spec
        self.model = hf_spec["model"]["type"]

        self.num_special_single = num_special_tokens(TemplateType.SINGLE, hf_spec)
        self.num_special_pair = num_special_tokens(TemplateType.PAIR, hf_spec)
        special_tokens = map_special_tokens_to_ids(vocab, config)
        self.special_tokens = special_tokens
        self._reserved_tokens = list(special_tokens.keys())
        self._vocab_path = tf.saved_model.Asset(vocab_path)

        # Include a tokenize signature for a batch of strings.
        self.tokenize.get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string)
        )

        # Include `detokenize` and `lookup` signatures for:
        #   * `Tensors` with shapes [tokens] and [batch, tokens]
        #   * `RaggedTensors` with shape [batch, tokens]
        self.detokenize.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int64)
        )
        self.detokenize.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64)
        )

        self.lookup.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int64)
        )
        self.lookup.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64)
        )

        # These `get_*` methods take no arguments
        self.get_vocab_size.get_concrete_function()
        self.get_vocab_path.get_concrete_function()
        self.get_reserved_tokens.get_concrete_function()

    @tf.function
    def tokenize(
        self, strings, max_length=None, padding: Optional[PaddingStrategies] = None
    ):
        input_ids = self.tokenizer.tokenize(strings)
        # Merge the `word` and `word-piece` axes.
        input_ids = input_ids.merge_dims(-2, -1)
        input_ids = self.truncate(input_ids)
        input_ids = self.post_process(input_ids)
        input_ids, mask = self.pad(input_ids, max_length=max_length, padding=padding)
        return {"input_ids": input_ids, "attention_mask": mask}

    def truncate(self, sequence: tf.RaggedTensor):
        """Truncation removes a sequence up to a specified lenght.

        Special tokens marking the start and end of a sequence (if specified) \
            will also be included in the sentence.
        """
        max_length = self.max_length  # TODO: update
        return sequence[:, :max_length]

    def pad(
        self,
        sequence: tf.RaggedTensor,
        max_length: int = None,
        padding: Optional[PaddingStrategies] = None,
    ) -> TF_TOKENIZER_INPUT:
        r"""
        Add padding of a sequence and supports different strategies for doing so.

        `strategy` supports different options:
            - `max_length`: Pad all batches to tokenizer `max_seq_length`.
            - `longest`: Pad all to same length as the longest sequence in the batch.
            - `none`: Do not pad at all.
        """
        _input_ids, _attention_mask = TF_TENSORS, TF_TENSORS  # type: ignore

        if padding is None:
            padding = (
                self.padding if self.padding is not None else PaddingStrategies.LONGEST
            )

        if max_length is None:
            max_length = (
                self.max_length
                if self.max_length is not None
                else self.model_max_length
            )

        if padding == PaddingStrategies.MAX_LENGTH:
            _input_ids, _attention_mask = text.pad_model_inputs(
                input=sequence, max_seq_length=max_length
            )
        elif padding == PaddingStrategies.LONGEST:
            seq_len_per_batch: tf.Tensor = sequence.nested_row_lengths()[0]
            longest_seq_in_batch = tf.reduce_max(seq_len_per_batch)
            _input_ids, _attention_mask = text.pad_model_inputs(
                input=sequence, max_seq_length=longest_seq_in_batch
            )
        elif padding == PaddingStrategies.NONE:
            _input_ids: tf.RaggedTensor = sequence
            _attention_mask: tf.RaggedTensor = tf.ones_like(sequence)
        else:
            raise NotValidPaddingStrategy(padding=padding)

        # Since Huggingface tensors are of same type
        input_ids = tf.cast(_input_ids, dtype=tf.int64)
        attention_mask = tf.cast(_attention_mask, dtype=tf.int64)
        return input_ids, attention_mask

    @tf.function
    def post_process(self, bs: tf.RaggedTensor) -> tf.RaggedTensor:
        r"""
        Specifies how a sentence should be constructed together with a set of special tokens.

        Huggingface tokenizers specify a template for how different post_processors should \
            behave when adding special tokens to a "single" or "pair" of sentences.

        An example template parsing instruction for BERT could look like:

            >>> from tftokenizers.load import load_json
            >>> config = load_json("saved_tokenizers/bert-base-uncased/tokenizer.json")
            >>> config["post_process"]["pair"]
            \"""
            [{'SpecialToken': {'id': '[CLS]', 'type_id': 0}},
            {'Sequence': {'id': 'A', 'type_id': 0}},
            {'SpecialToken': {'id': '[SEP]', 'type_id': 0}},
            {'Sequence': {'id': 'B', 'type_id': 1}},
            {'SpecialToken': {'id': '[SEP]', 'type_id': 1}}]
            \"""

        Each type of tokenizer can have different templates between the models. \
            This function parses the template into a simple abstract syntax tree \
            and combines the original sentence with the special tokens according \
            according to the template.
        """

        num_sentences = bs.bounding_shape()[0]
        template = self.tokenizer_spec["post_processor"]
        processing_steps = template[TemplateType.SINGLE]
        num_steps = len(processing_steps)

        def get_token_id_from_current_step_in_template(
            instruction_step: PROCESSING_STEP,
        ) -> tf.Tensor:
            token_symbol: str = list(instruction_step.values())[0]["id"]
            token_id: tf.Tensor = self.special_tokens.get(token_symbol)
            return token_id

        # Build AST from template instruction
        for step in range(num_steps - 1):
            left: PROCESSING_STEP = processing_steps[step]
            right: PROCESSING_STEP = processing_steps[step + 1 - num_steps]
            left_token = get_token_id_from_current_step_in_template(left)
            right_token = get_token_id_from_current_step_in_template(right)

            if left != right:
                left_sentence = bs
                right_sentence = bs

                if "SpecialToken" in left:
                    left_sentence = tf.fill([num_sentences, 1], left_token)
                elif "SpecialToken" in right:
                    right_sentence = tf.fill([num_sentences, 1], right_token)
                bs = tf.concat([left_sentence, right_sentence], axis=1)
        return bs

    def cleanup_text(self, token_txt):
        bad_tokens = [
            re.escape(tok)
            for tok in self._reserved_tokens
            if tok != SpecialTokens.UNK_TOKEN.value
        ]
        bad_token_re = "|".join(bad_tokens)

        bad_cells = tf.strings.regex_full_match(token_txt, bad_token_re)
        result = tf.ragged.boolean_mask(token_txt, ~bad_cells)

        # Join them into strings.
        result = tf.strings.reduce_join(result, separator=" ", axis=-1)
        return result

    @tf.function
    def get_max_length(self):
        return self.max_length

    @tf.function
    def detokenize(self, tokenized):
        words = self.tokenizer.detokenize(tokenized)
        return self.cleanup_text(words)

    @tf.function
    def lookup(self, token_ids):
        return tf.gather(self.vocab, token_ids)

    @tf.function
    def get_vocab_size(self):
        return tf.shape(self.vocab)[0]

    @tf.function
    def get_vocab_path(self):
        return self._vocab_path

    @tf.function
    def get_reserved_tokens(self):
        return tf.constant(self._reserved_tokens)

    # New methods with names similar to
    # The Huggingface Tokenizer interface
    @tf.function
    def encode(self, strings):
        return self.tokenize(strings)

    @tf.function
    def batch_encode(self, strings):
        return self.tokenize(strings)

    @tf.function
    def batch_encode_plus(self, strings):
        return self.tokenize(strings)

    @tf.function
    def decode(self, tokenized):
        return self.detokenize(tokenized)

    @tf.function
    def batch_decode(self, tokenized):
        return self.detokenize(tokenized)
