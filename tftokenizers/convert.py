import tensorflow as tf
import tensorflow_text as text
from transformers import AutoTokenizer

from tftokenizers.file import get_filename_from_path, get_vocab_from_path, load_json
from tftokenizers.tokenizer import TFAutoTokenizer, TFTokenizerBase
from tftokenizers.types import PaddingStrategies

tf.get_logger().setLevel("ERROR")


if __name__ == "__main__":
    PATH = "saved_tokenizers/bert-base-uncased"
    vocab = get_vocab_from_path(PATH)
    vocab_path = get_filename_from_path(PATH, "vocab")

    config = load_json(f"{PATH}/tokenizer_config.json")
    tokenizer_spec = load_json(f"{PATH}/tokenizer.json")
    special_tokens_map = load_json(f"{PATH}/special_tokens_map.json")

    tokenizer_base_params = dict(lower_case=True, token_out_type=tf.int64)
    tokenizer_base = text.BertTokenizer(vocab_path, **tokenizer_base_params)

    # Define sentences
    s1 = "sponge bob squarepants is an avenger"
    s2 = "Huggingface to Tensorflow tokenizers"
    s3 = "HelLo!"

    tokenizer = TFTokenizerBase(
        vocab_path=vocab_path,
        tokenizer_base=tokenizer_base,
        hf_spec=tokenizer_spec,
        config=config,
    )
    hf_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # tokenizer = TFAutoTokenizer("bert-base-uncased")
    tokenizer = TFAutoTokenizer.from_pretrained("bert-base-uncased")

    # # Save tokenizer
    model_name = "exported_custom_bert_tokenizer_auto"
    tf.saved_model.save(tokenizer, model_name)

    # # Load tokenizer
    reloaded_tokenizer = tf.saved_model.load(model_name)

    tokens = tokenizer.tokenize(
        [s1, s2, s3],
        padding=PaddingStrategies.LONGEST,
        max_length=10,
    )["input_ids"]
    print("Token IDs:\n", tokens.numpy())

    tokens = hf_tokenizer.batch_encode_plus(
        [s1, s2, s3],
        padding=PaddingStrategies.LONGEST.value,
        max_length=10,
    )
    print("hf tokens\n", tokens)

    tokens = reloaded_tokenizer.tokenize([s1, s2, s3])["input_ids"]
    print("Token IDs:\n", tokens.numpy())
    print("")

    tokens = reloaded_tokenizer.tokenize(
        inputs=tf.constant([s1, s2, s3], dtype=tf.string, name="inputs"),
    )["input_ids"]
    print("Token IDs:\n", tokens.numpy())
