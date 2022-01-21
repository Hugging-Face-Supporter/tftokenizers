import tensorflow as tf
from transformers import TFAutoModel

from tftokenizers.model import TFModel
from tftokenizers.tokenizer import TFAutoTokenizer

if __name__ == "__main__":
    # Load base models from Huggingface
    pretrained_model_name = "bert-base-cased"
    tokenizer = TFAutoTokenizer(pretrained_model_name)
    model = TFAutoModel.from_pretrained(pretrained_model_name)

    # Create a Reusable SavedModel
    custom_model = TFModel(model=model, tokenizer=tokenizer)

    # Use model in inference or training
    # With the tokenizer in Tensorflow, it can handle Tensors and python string
    string_tf = tf.constant(["Hello from Tensorflow"])
    s1 = "sponge bob squarepants is an avenger"
    s2 = "Huggingface to Tensorflow tokenizers"
    s3 = "HelLo!"

    output = custom_model(string_tf)
    output = custom_model([s1, s2, s3])

    # You can also pass arguments, similar how they are named from Huggingface
    output = custom_model(
        [s1, s2, s3],
        max_length=512,
        padding="max_length",
    )
    print(output)
