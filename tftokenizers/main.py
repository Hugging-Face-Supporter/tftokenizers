import tensorflow as tf
import tensorflow_text as text
from transformers import AutoTokenizer, TFAutoModel
from transformers.utils.logging import set_verbosity_error

from tftokenizers.file import get_filename_from_path, get_vocab_from_path, load_json
from tftokenizers.model import TFModel
from tftokenizers.tokenizer import TFAutoTokenizer, TFTokenizerBase

set_verbosity_error()
tf.get_logger().setLevel("ERROR")

if __name__ == "__main__":
    # Load base models from Huggingface
    pretrained_model_name = "bert-base-cased"
    pretrained_model_name = "distilbert-base-cased"
    pretrained_model_name = "distilbert-base-multilingual-cased"
    pretrained_model_name = "bert-base-multilingual-cased"
    pretrained_model_name = "KB/bert-base-swedish-cased"

    tf_tokenizer = TFAutoTokenizer.from_pretrained(pretrained_model_name)
    hf_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    model = TFAutoModel.from_pretrained(pretrained_model_name)

    # Three ways to load the TF tokenizer
    # a) by model_name
    tf_tokenizer = TFAutoTokenizer.from_pretrained(pretrained_model_name)

    # b) bundled with the model, similar to TFHub
    model = TFAutoModel.from_pretrained(pretrained_model_name)
    custom_model = TFModel(model=model, tokenizer=tf_tokenizer)

    # c) from vocab file
    PATH = "saved_tokenizers/bert-base-uncased"
    vocab = get_vocab_from_path(PATH)
    vocab_path = get_filename_from_path(PATH, "vocab")

    config = load_json(f"{PATH}/tokenizer_config.json")
    tokenizer_spec = load_json(f"{PATH}/tokenizer.json")
    special_tokens_map = load_json(f"{PATH}/special_tokens_map.json")

    tokenizer_base_params = dict(lower_case=True, token_out_type=tf.int64)
    tokenizer_base = text.BertTokenizer(vocab_path, **tokenizer_base_params)
    custom_tokenizer = TFTokenizerBase(
        vocab_path=vocab_path,
        tokenizer_base=tokenizer_base,
        hf_spec=tokenizer_spec,
        config=config,
    )

    # Usage
    # TF tokenizer have the same interface as HF transformers
    sentence = "Huggingface to Tensorflow tokenizers"
    print(tf_tokenizer.batch_encode_plus([sentence])["input_ids"])
    print(
        hf_tokenizer.batch_encode_plus(
            [sentence],
            padding=True,
            truncation=True,
            return_token_type_ids=False,
            return_tensors="tf",
        )["input_ids"]
    )

    # Create a Reusable SavedModel
    custom_model = TFModel(model=model, tokenizer=tf_tokenizer)

    # With the tokenizer in Tensorflow, it can handle Tensors and python string
    string_tf = tf.constant(["Hello from Tensorflow"])
    output = custom_model(string_tf)

    # And pass input for either inference (Default) or training
    output = custom_model(string_tf, training=False)  # inference
    output = custom_model(string_tf, training=True)  # training

    # Or pass a batch of sentences, similar to Huggingface
    s1 = "sponge bob squarepants is an avenger"
    s2 = "Huggingface to Tensorflow tokenizers"
    s3 = "HelLo!"
    output = custom_model(
        [s1, s2, s3],
    )
    print(output)
