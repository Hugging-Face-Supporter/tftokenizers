# TFtftransformers
Converting Hugginface tokenizers to Tensorflow tokenizers. The main reason is to be able to bundle the tokenizer and model into one Reusable SavedModel.

<a href="https://badge.fury.io/py/tftokenizers"><img src="https://badge.fury.io/py/tftokenizers.svg" alt="PyPI version" height="18"></a>
---

**Source Code**: <a href="https://github.com/Huggingface-Supporters/tftftransformers" target="_blank">https://github.com/Hugging-Face-Supporter/tftokenizers</a>

---


## Example
This is an example of how one can use Huggingface model and tokenizers bundled together as a [Reusable SavedModel](https://www.tensorflow.org/hub/reusable_saved_models) and yields the same result as using the model and tokenizer from Huggingface 🤗


```python
import tensorflow as tf
from transformers import TFAutoModel
from tftokenizers import TFModel, TFAutoTokenizer

# Load base models from Huggingface
model_name = "bert-base-cased"
model = TFAutoModel.from_pretrained(model_name)

# Load converted TF tokenizer
tokenizer = TFAutoTokenizer.from_pretrained(model_name)

# Create a TF Reusable SavedModel
custom_model = TFModel(model=model, tokenizer=tokenizer)

# Tokenizer and model can handle `tf.Tensors` or regular strings
tf_string = tf.constant(["Hello from Tensorflow"])
s1 = "SponGE bob SQuarePants is an avenger"
s2 = "Huggingface to Tensorflow tokenizers"
s3 = "Hello, world!"

output = custom_model(tf_string)
output = custom_model([s1, s2, s3])

# We can now pass input as tensors
output = custom_model(
    inputs=tf.constant([s1, s2, s3], dtype=tf.string, name="inputs"),
)

# Save tokenizer
saved_name = "reusable_bert_tf"
tf.saved_model.save(custom_model, saved_name)

# # Load tokenizer
reloaded_model = tf.saved_model.load(saved_name)
output = reloaded_model([s1, s2, s3])
print(output)
```

## `Setup`
```bash
git clone https://github.com/Hugging-Face-Supporter/tftokenizers.git
cd tftokenizers
poetry install
poetry shell
```

## `Run`
To convert a Huggingface tokenizer to Tensorflow, first choose one from the models or tokenizers from the Huggingface hub to download.

**NOTE**
> Currently only BERT models work with the converter.

### `Download`
First download tokenizers from the hub by name. Either run the bash script do download multiple tokenizers or download a single tokenizer with the python script.

The idea is to eventually only to automatically download and convert

```bash
python tftokenizers/download.py -n bert-base-uncased
bash scripts/download_tokenizers.sh
```

### `Convert`
Convert downloaded tokenizer from Huggingface format to Tensorflow
```bash
python tftokenizers/convert.py
```

## `Before Commit`
```bash
make build
```



## WIP
- [x] Convert a BERT tokenizer from Huggingface to Tensorflow
- [x] Make a TF Reusabel SavedModel with Tokenizer and Model in the same class. Emulate how the TF Hub example for BERT works.
- [x] Find methods for identifying the base tokenizer model and map those settings and special tokens to new tokenizers
- [x] Extend the tokenizers to more tokenizer types and identify them from a huggingface model name
- [x] Document how others can use the library and document the different stages in the process
- [x] Improve the conversion pipeline (s.a. Download and export files if not passed in or available locally)
- [ ] Convert other tokenizers. Identify limitations
- [ ] Support encoding of two sentences at a time [Ref](https://www.tensorflow.org/text/guide/bert_preprocessing_guide)
- [ ] Allow the tokenizers to be used for Masking (MLM) [Ref](https://www.tensorflow.org/text/guide/bert_preprocessing_guide)
