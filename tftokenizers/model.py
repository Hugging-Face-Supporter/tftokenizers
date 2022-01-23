from typing import Optional

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_hub as hub
import tensorflow_text as text
from transformers import AutoTokenizer, TFAutoModel
from transformers.utils.logging import set_verbosity_error

from tftokenizers.file import get_filename_from_path, get_vocab_from_path, load_json
from tftokenizers.tokenizer import TFAutoTokenizer, TFTokenizerBase
from tftokenizers.types import PaddingStrategies

set_verbosity_error()
tf.get_logger().setLevel("ERROR")


class TFModel(keras.layers.Layer):
    def __init__(self, model, tokenizer, **kwargs):
        super(TFModel, self).__init__(**kwargs)
        self.tokenizer = tokenizer
        self.model = model

    def call(
        self,
        inputs,
        verbose=False,
        training=True,
    ):
        tokens = self.tokenizer.tokenize(inputs=inputs)
        if verbose:
            print(tokens)

        # we only keep tensor 0 (last_hidden_state) for now
        embeddings = self.model(tokens, training=training)
        last_hidden_out = embeddings[0]
        return last_hidden_out


if __name__ == "__main__":
    PATH = "saved_tokenizers/bert-base-uncased"
    vocab = get_vocab_from_path(PATH)
    vocab_path = get_filename_from_path(PATH, "vocab")

    config = load_json(f"{PATH}/tokenizer_config.json")
    tokenizer_spec = load_json(f"{PATH}/tokenizer.json")
    special_tokens_map = load_json(f"{PATH}/special_tokens_map.json")

    tokenizer_base_params = dict(lower_case=True, token_out_type=tf.int64)
    tokenizer_base = text.BertTokenizer(vocab_path, **tokenizer_base_params)

    model = TFAutoModel.from_pretrained("bert-base-uncased")
    hf_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    s1 = "sponge bob squarepants is an avenger"
    s2 = "Huggingface to Tensorflow tokenizers"
    s3 = "HelLo!"

    # Define out custom tokenizer
    max_length = 512
    custom_tokenizer = TFTokenizerBase(
        vocab_path=vocab_path,
        tokenizer_base=tokenizer_base,
        hf_spec=tokenizer_spec,
        config=config,
    )
    custom_tokenizer = TFAutoTokenizer.from_pretrained("bert-base-uncased")

    # Building our custom TF model pipeline
    custom_model = TFModel(model, custom_tokenizer)
    output = custom_model(
        [s1, s2, s3],
    )

    # Compare against the Huggingface implementation
    hf_tokens = hf_tokenizer.batch_encode_plus(
        [s1, s2, s3],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="tf",
    )

    print("\n\nmodel output")
    hf_output = model(hf_tokens)[0]
    print(output)
    print(hf_output)

    # Save custom TF model and re-use it
    model_name = "bert_reusable_tf"
    tf.saved_model.save(custom_model, model_name)
    reloaded_model = tf.saved_model.load(model_name)
    output = custom_model([s1, s2, s3])
    print(output)
