import numpy as np
import pytest
import tensorflow as tf
import tensorflow_text as text
from tftokenizers.file import get_filename_from_path, load_json
from tftokenizers.model import TFModel
from tftokenizers.tokenizer import PaddingStrategies, TFTokenizerBase
from transformers import AutoTokenizer, TFAutoModel

PATH = "saved_tokenizers/bert-base-uncased"
VOCAB_PATH = get_filename_from_path(PATH, "vocab")

config = load_json(f"{PATH}/tokenizer_config.json")
tokenizer_spec = load_json(f"{PATH}/tokenizer.json")
special_tokens_map = load_json(f"{PATH}/special_tokens_map.json")

tokenizer_base_params = dict(lower_case=True, token_out_type=tf.int64)
tokenizer_base = text.BertTokenizer(VOCAB_PATH, **tokenizer_base_params)

s1 = "sponge bob squarepants is an avenger"
s2 = "Huggingface to Tensorflow tokenizers"
s3 = "HelLo!"


@pytest.fixture
def hf_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer


@pytest.fixture
def model():
    model = TFAutoModel.from_pretrained("bert-base-uncased")
    return model


def test_huggingface_model_with_custom_and_their_tokenizer(hf_tokenizer, model):
    # Define out custom tokenizer
    max_length = 512
    custom_tokenizer = TFTokenizerBase(
        vocab_path=VOCAB_PATH,
        tokenizer_base=tokenizer_base,
        hf_spec=tokenizer_spec,
        config=config,
    )

    # Building our custom TF model pipeline
    custom_model = TFModel(model=model, tokenizer=custom_tokenizer)
    tf_output = custom_model(
        [s1, s2, s3],
        padding=PaddingStrategies.MAX_LENGTH,
        max_length=max_length,
        training=False,
    )
    print(tf_output)  # with `shape=(3, 512, 768)`

    # Compare against the Huggingface implementation
    hf_tokens = hf_tokenizer.batch_encode_plus(
        [s1, s2, s3],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="tf",
    )
    hf_tokens.pop("token_type_ids")
    hf_output = model(hf_tokens)[0]
    print(hf_output)

    np.testing.assert_array_almost_equal(x=tf_output.numpy(), y=hf_output.numpy())
