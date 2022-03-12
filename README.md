# TFtftransformers

Converting Hugginface tokenizers to Tensorflow tokenizers. The main reason is to be able to bundle the tokenizer and model into one Reusable SavedModel, inspired by the [Tensorflow Official Guide on tokenizers](hhttps://www.tensorflow.org/text/guide/bert_preprocessing_guide)

## <a href="https://badge.fury.io/py/tftokenizers"><img src="https://badge.fury.io/py/tftokenizers.svg" alt="PyPI version" height="18"></a>

**Source Code**: <a href="https://github.com/Huggingface-Supporters/tftftransformers" target="_blank">https://github.com/Hugging-Face-Supporter/tftokenizers</a>

---

Models we know works:

```python
"bert-base-cased"
"bert-base-uncased"
"bert-base-multilingual-cased"
"bert-base-multilingual-uncased"
# Distilled
"distilbert-base-cased"
"distilbert-base-multilingual-cased"
"microsoft/MiniLM-L12-H384-uncased"
# Non-english
"KB/bert-base-swedish-cased"
"bert-base-chinese"
```

## Examples

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

# Load tokenizer
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

## FAQ

### How to know what tokenizer is used?
**TL;DR**
```python
from transformers import AutoTokenizer

name = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(name)

# IF the tokenizer is fast:
print(tokenizer.is_fast)
# Base tokenizer model
print(type(tokenizer.backend_tokenizer.model))
# Check if it is a SentencePiece tokenizer
# Should be `vocab.txt` or `vocab.json` if not SentencePiece tokenizer
# SencePiece if "vocab_file":
#   "sentencepiece.bpe.model"
print(tokenizer.vocab_files_names)

# Else
# Find if the model is a SentencePiece model with
print(vars(tokenizer).get("spm_file", None))
# print(vars(tokenizer).get("sp_model", None))
```

<details>
<summary>:memo: Read More:</summary>
And the components of the tokenizers described [here](https://huggingface.co/docs/tokenizers/python/latest/components.html) as:
- Normalizers
- Pre tokenizers
- [Models](https://huggingface.co/docs/tokenizers/python/latest/components.html#models)
- PostProcessor
- Decoders


When loading a tokenizer with Huggingface transformers, it maps the name of the model from the Huggingface Hub to the correct model and tokenizer available there, if not it will try to to find a folder on your local computer with that name.

Additionally, tokenizers from Huggingface are defined in multiple different steps using the Huggingface tokenizer library. For those interested, you can look into the different components of that library of how the composition of a tokenizer works [here](https://huggingface.co/docs/tokenizers/python/latest/). There is also a great guide documenting how composition of tokenizers are done in this [Medium article](https://towardsdatascience.com/designing-tokenizers-for-low-resource-languages-7faa4ab30ef4)
</details>

### What tokenizers are used by what models?
<details>
<summary>:memo: Read More:</summary>
As stated in the section above, you will need to look at each model to inspect the type of tokenizer it is using, but in general there are just a few "base tokenizers / models". See [Huggingface documentation](https://huggingface.co/docs/transformers/tokenizer_summary) for explanation on how these "base tokenizers" are defined

[Base Tokenizer Names](https://github.com/huggingface/tokenizers/blob/master/bindings/python/py_src/tokenizers/models/__init__.py)
[Model Implementations](https://github.com/huggingface/tokenizers/tree/master/bindings/python/py_src/tokenizers/implementations)

SentencePiece tokenizers can either be BPE (rare if the tokenizers is fast) or Unigram (all Unigram == SentencePiece)
#### BPE = tokenizers.models.BPE
- Implemented by

    [byte-pair BPE](https://github.com/huggingface/tokenizers/blob/master/bindings/python/py_src/tokenizers/implementations/byte_level_bpe.py), [char-level BPE](https://github.com/huggingface/tokenizers/blob/master/bindings/python/py_src/tokenizers/implementations/char_level_bpe.py), ([SentencePiece BPE](https://github.com/huggingface/tokenizers/blob/master/bindings/python/py_src/tokenizers/implementations/sentencepiece_bpe.py))

- Used by

    `GPT`, `XLNet`, `FlauBERT`, `RoBERTa`, `GPT-2`, `GPT-j`, `GPT-neo`, `BART`, `XLM-RoBERTa`
#### Unigram = tokenizers.models.Unigram
- Implemented by

    [SentencePiece Unicode](https://github.com/huggingface/tokenizers/blob/master/bindings/python/py_src/tokenizers/implementations/sentencepiece_unigram.py)

- Used by

    All `T5` models
#### WordPiece = tokenizers.models.WordPiece
- Implemented by

    [Bert WordPiece](https://github.com/huggingface/tokenizers/blob/master/bindings/python/py_src/tokenizers/implementations/bert_wordpiece.py)

- Used by

    `BERT`, `mBERT`, `miniLM`, distilled versions of BERT

#### SentencePiece
SentencePiece is a method for creating sub-word tokenizations.
It supports BPE and Unigram.

SentencePiece is a separate C++ implemented library with python and Tensorflow bindings.
The vocabulary is bundled into:

**For fast models**:

"vocab_file_names":

    `sentencepiece.bpe.model` for "BPE" and
    `spiece.model` for Unigram

**For slow models**:

"vocab_file_names":

    'source_spm': 'source.spm',
    'target_spm': 'target.spm',
    'vocab': 'vocab.json'

"spm_files":

    will be a single file or a list of files
    ...

- Used by:

    **Fast**: `T5` models
    **Slow**: `facebook/m2m100_418M`, `facebook/wmt19-en-de`
</details>

### How to implement the tokenizers from Huggingface to Tensorflow?
You will need to download the Huggingface tokenizer of your choice, determine the type of the tokenizer (`is_fast`, tokenizer type and `vocab_file_names`). Then map the tokenizer used to the Tensorflow supported equivalent:

https://github.com/tensorflow/text/issues/422

**BPE** and **Unigram**:
- All BPE implementations for Tensorflow is backed by SentencePiece
- [SentencePiece in TensorFlow](https://www.tensorflow.org/text/api_docs/python/text/SentencepieceTokenizer)
- [Official Answer 1](https://github.com/tensorflow/text/issues/415)
- [Official Answer 2](https://github.com/tensorflow/text/issues/763)
- [How to load a SentencePiece model](https://github.com/tensorflow/text/issues/215)
- [Input will need to be Tensors](https://github.com/tensorflow/text/issues/512)
- [How to load model from vocab](https://github.com/tensorflow/text/issues/452)

**WordPiece**:
- [BertTokenizer](https://www.tensorflow.org/text/api_docs/python/text/BertTokenizer) or
- [WordPiece](https://www.tensorflow.org/text/api_docs/python/text/FastWordpieceTokenizer) or
- [FastWordPiece](https://www.tensorflow.org/text/api_docs/python/text/FastWordpieceTokenizer)


https://github.com/tensorflow/text/issues/116
https://github.com/tensorflow/text/issues/414

### What other ways are there to convert a tokenizer?
<details>
<summary>:memo: Read More:</summary>
With `tfokenizers` there are three ways to use the package:

```python
import tensorflow as tf
import tensorflow_text as text
from transformers import AutoTokenizer, TFAutoModel
from transformers.utils.logging import set_verbosity_error

from tftokenizers.file import (
    get_filename_from_path,
    get_vocab_from_path,
    load_json
)
from tftokenizers.model import TFModel
from tftokenizers.tokenizer import TFAutoTokenizer, TFTokenizerBase

set_verbosity_error()
tf.get_logger().setLevel("ERROR")

pretrained_model_name = "bert-base-cased"


# a) by model_name
tf_tokenizer = TFAutoTokenizer.from_pretrained(pretrained_model_name)

# b) bundled with the model, similar to TFHub
model = TFAutoModel.from_pretrained(pretrained_model_name)
custom_model = TFModel(model=model, tokenizer=tf_tokenizer)

# c) from source, using the saved files of a transformers tokenizer
# Make sure you run download.py or the download script first
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
```
</details>


### How to save Huggingface Tokenizer files locally?
<details>
<summary>:memo: Read More:</summary>

To download the files used by Huggingface tokenizers, you can either download one by name
```
python tftokenizers/download.py -n KB/bert-base-swedish-cased
```
or download multiple
```
bash scrips/download_tokenizers.sh
```
</details>

## WIP

- [x] Convert a BERT tokenizer from Huggingface to Tensorflow
- [x] Make a TF Reusabel SavedModel with Tokenizer and Model in the same class. Emulate how the TF Hub example for BERT works.
- [x] Find methods for identifying the base tokenizer model and map those settings and special tokens to new tokenizers
- [x] Extend the tokenizers to more tokenizer types and identify them from a huggingface model name
- [x] Document how others can use the library and document the different stages in the process
- [x] Improve the conversion pipeline (s.a. Download and export files if not passed in or available locally)
- [ ] `model_max_length` should be regulated. However, some newer models have the max_lenght for tokenizers at 1000_000_000
- [ ] Support more tokenizers, starting with SentencePiece
- [ ] Identify tokenizer conversion limitations
- [ ] Support encoding of two sentences at a time [Ref](https://www.tensorflow.org/text/guide/bert_preprocessing_guide)
- [ ] Allow the tokenizers to be used for Masking (MLM) [Ref](https://www.tensorflow.org/text/guide/bert_preprocessing_guide)
