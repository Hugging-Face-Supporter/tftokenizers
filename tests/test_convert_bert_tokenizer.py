import numpy as np
import pytest
import tensorflow as tf
import tensorflow_text as text
from tftokenizers.file import get_filename_from_path, load_json
from tftokenizers.tokenizer import TFTokenizerBase
from tftokenizers.types import PaddingStrategies
from transformers import AutoTokenizer, TFAutoModel

MODEL_NAME_OR_PATH = "bert-base-uncased"
ATTENTION_MASK = "attention_mask"
INPUT_IDS = "input_ids"
MAX_LENGTH = 512
PATH = "saved_tokenizers/bert-base-uncased"
VOCAB_PATH = get_filename_from_path(PATH, "vocab")

s1 = "sponge bob squarepants is an avenger"
s2 = "Svenska ordet smörgåsbord"

config = load_json(f"{PATH}/tokenizer_config.json")
tokenizer_spec = load_json(f"{PATH}/tokenizer.json")
tokenizer_base = text.BertTokenizer(VOCAB_PATH, lower_case=True)


@pytest.fixture
def tf_tokenizer():
    tokenizer = TFTokenizerBase(
        vocab_path=VOCAB_PATH,
        tokenizer_base=tokenizer_base,
        hf_spec=tokenizer_spec,
        config=config,
    )
    return tokenizer


@pytest.fixture
def tf_tokenizer_padding_to_max_length():
    tokenizer = TFTokenizerBase(
        vocab_path=VOCAB_PATH,
        tokenizer_base=tokenizer_base,
        hf_spec=tokenizer_spec,
        config=config,
    )
    return tokenizer


@pytest.fixture
def tf_tokenizer_padding_to_longest_in_batch():
    tokenizer = TFTokenizerBase(
        vocab_path=VOCAB_PATH,
        tokenizer_base=text.BertTokenizer(VOCAB_PATH, lower_case=True),
        hf_spec=tokenizer_spec,
        config=config,
    )
    return tokenizer


@pytest.fixture
def hf_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
    return tokenizer


@pytest.fixture
def model():
    model = TFAutoModel.from_pretrained(MODEL_NAME_OR_PATH)
    return model


def test_custom_tokenizer_one_sentence(tf_tokenizer):
    """Test that different input methods are tokenized the same and are supported."""

    token_id_tf = tf_tokenizer.tokenize(s1)["input_ids"]
    token_id_copy1_tf = tf_tokenizer.tokenize([s1])["input_ids"]
    token_id_copy2_tf = tf_tokenizer.tokenize(tf.constant(s1))["input_ids"]
    token_id_copy3_tf = tf_tokenizer.tokenize(tf.constant([s1]))["input_ids"]

    np.testing.assert_array_almost_equal(
        x=token_id_tf.numpy(), y=token_id_copy1_tf.numpy()
    )
    np.testing.assert_array_almost_equal(
        x=token_id_tf.numpy(), y=token_id_copy2_tf.numpy()
    )
    np.testing.assert_array_almost_equal(
        x=token_id_tf.numpy(), y=token_id_copy3_tf.numpy()
    )


def test_compare_huggingface_custom_tokenizer(hf_tokenizer, tf_tokenizer):
    # 1. Test simple English sentence
    tokens_tf = tf_tokenizer.tokenize(
        [s1], padding=PaddingStrategies.LONGEST, max_length=512
    )

    tokens_hf = hf_tokenizer.batch_encode_plus(
        [s1], return_tensors="tf", padding=True, truncation=True
    )
    np.testing.assert_almost_equal(
        tokens_tf["input_ids"].numpy(), tokens_hf["input_ids"].numpy()
    )
    np.testing.assert_almost_equal(
        tokens_tf["attention_mask"].numpy(), tokens_hf["attention_mask"].numpy()
    )

    # 2. Test simple Swedish sentence
    tokens_tf = tf_tokenizer.tokenize(
        [s2], padding=PaddingStrategies.LONGEST, max_length=512
    )
    tokens_hf = hf_tokenizer.batch_encode_plus(
        [s2], return_tensors="tf", padding=True, truncation=True
    )
    np.testing.assert_almost_equal(
        tokens_tf["input_ids"].numpy(), tokens_hf["input_ids"].numpy()
    )
    np.testing.assert_almost_equal(
        tokens_tf["attention_mask"].numpy(), tokens_hf["attention_mask"].numpy()
    )


def test_compare_huggingface_custom_model(hf_tokenizer, tf_tokenizer, model):
    tokens_tf = tf_tokenizer.tokenize(
        [s1], padding=PaddingStrategies.LONGEST, max_length=512
    )
    tokens_hf = hf_tokenizer.batch_encode_plus(
        [s1], return_tensors="tf", padding=True, truncation=True
    )

    out_tf = model(tokens_tf)
    out_hf = model(tokens_hf)

    last_hidden_tf = out_tf["last_hidden_state"]
    last_hidden_hf = out_hf["last_hidden_state"]
    pooler_tf = out_tf["pooler_output"]
    pooler_hf = out_hf["pooler_output"]

    np.testing.assert_almost_equal(last_hidden_tf.numpy(), last_hidden_hf.numpy())
    np.testing.assert_almost_equal(pooler_tf.numpy(), pooler_hf.numpy())


def test_padding_for_tokenizers(
    hf_tokenizer,
    tf_tokenizer_padding_to_max_length,
    tf_tokenizer_padding_to_longest_in_batch,
):
    # 1. Test pad and truncate to longest sequence in batch
    tokens_tf = tf_tokenizer_padding_to_longest_in_batch.tokenize(
        [s1, s2], padding=PaddingStrategies.LONGEST, max_length=512
    )["input_ids"]
    tokens_hf = hf_tokenizer.batch_encode_plus(
        [s1, s2], return_tensors="tf", padding=True, truncation=True
    )["input_ids"]
    assert list(tokens_tf.shape) == list(tokens_hf.shape)

    # 2. Pad and truncate all to max tokenizer length
    tokens_tf = tf_tokenizer_padding_to_max_length.tokenize(
        [s1, s2],
        max_length=512,
        padding=PaddingStrategies.MAX_LENGTH,
    )["input_ids"]
    tokens_hf = hf_tokenizer.batch_encode_plus(
        [s1, s2],
        return_tensors="tf",
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )["input_ids"]
    assert list(tokens_tf.shape) == list(tokens_hf.shape)
