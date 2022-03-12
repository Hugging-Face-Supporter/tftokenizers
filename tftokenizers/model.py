from typing import Dict, List, Union

import tensorflow as tf
import tensorflow.keras as keras
from transformers import AutoTokenizer, TFAutoModel
from transformers.utils.logging import set_verbosity_error

from tftokenizers.tokenizer import TFAutoTokenizer

set_verbosity_error()
tf.get_logger().setLevel("ERROR")


def compare_models(pretrained_model_name: str, inputs: Union[str, List[str]]):
    """Helper function for comparing TF and HF tokenizers."""

    model = TFAutoModel.from_pretrained(pretrained_model_name)

    tf_tokenizer = TFAutoTokenizer.from_pretrained(pretrained_model_name)
    tf_hub_model = TFModel(model=model, tokenizer=tf_tokenizer)
    tf_output = tf_hub_model(inputs, training=False)

    hf_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    hf_tokens: Dict[str, tf.Tensor] = hf_tokenizer.batch_encode_plus(
        inputs,
        padding=True,
        truncation=True,
        return_token_type_ids=False,
        return_tensors="tf",
    )
    hf_output = model(hf_tokens)[0]  # "last_hidden_state"

    return {"tf_output": tf_output, "hf_output": hf_output}


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
