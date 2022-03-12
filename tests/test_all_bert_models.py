import numpy as np
import pytest
import tensorflow as tf
import tensorflow_text as text

from tftokenizers.tokenizer import compare_tokenizers

s1 = "sponge bob squarepants is an avenger"
s2 = "Huggingface to Tensorflow tokenizers"
s3 = "HelLo!"
swe = "Pippi Långstrump, en karaktär skapad av Astrid Lindgren"


def test_bert_base_cased():
    # Define out custom tokenizer
    pretrained_model_name = "bert-base-cased"
    out = compare_tokenizers(pretrained_model_name, [s1, s2, s3])
    np.testing.assert_array_almost_equal(
        x=out["tf_tokens"]["input_ids"].numpy(),
        y=out["hf_tokens"]["input_ids"].numpy(),
    )


def test_bert_base_multilingual_cased():
    # Define out custom tokenizer
    pretrained_model_name = "bert-base-multilingual-cased"
    out = compare_tokenizers(pretrained_model_name, [s1, s2, s3])
    np.testing.assert_array_almost_equal(
        x=out["tf_tokens"]["input_ids"].numpy(),
        y=out["hf_tokens"]["input_ids"].numpy(),
    )


def test_distilbert_base_cased():
    # Define out custom tokenizer
    pretrained_model_name = "distilbert-base-cased"
    out = compare_tokenizers(pretrained_model_name, [s1, s2, s3])
    np.testing.assert_array_almost_equal(
        x=out["tf_tokens"]["input_ids"].numpy(),
        y=out["hf_tokens"]["input_ids"].numpy(),
    )


def test_distilbert_base_multilingual_cased():
    # Define out custom tokenizer
    pretrained_model_name = "distilbert-base-multilingual-cased"
    out = compare_tokenizers(pretrained_model_name, [s1, s2, s3])
    np.testing.assert_array_almost_equal(
        x=out["tf_tokens"]["input_ids"].numpy(),
        y=out["hf_tokens"]["input_ids"].numpy(),
    )


def test_kb_bert_base_swedish_cased():
    # Define out custom tokenizer
    pretrained_model_name = "KB/bert-base-swedish-cased"
    out = compare_tokenizers(pretrained_model_name, [s1, s2, swe])
    np.testing.assert_array_almost_equal(
        x=out["tf_tokens"]["input_ids"].numpy(),
        y=out["hf_tokens"]["input_ids"].numpy(),
    )


def test_bert_base_chinese():
    # Define out custom tokenizer
    pretrained_model_name = "bert-base-chinese"
    out = compare_tokenizers(pretrained_model_name, [s1, s2, s3])
    np.testing.assert_array_almost_equal(
        x=out["tf_tokens"]["input_ids"].numpy(),
        y=out["hf_tokens"]["input_ids"].numpy(),
    )


def test_bert_base_uncased():
    pretrained_model_name = "bert-base-uncased"
    out = compare_tokenizers(pretrained_model_name, [s1, s2, s3])
    np.testing.assert_array_almost_equal(
        x=out["tf_tokens"]["input_ids"].numpy(),
        y=out["hf_tokens"]["input_ids"].numpy(),
    )


def test_bert_base_multilingual_uncased():
    pretrained_model_name = "bert-base-multilingual-uncased"
    out = compare_tokenizers(pretrained_model_name, [s1, s2, s3])
    np.testing.assert_array_almost_equal(
        x=out["tf_tokens"]["input_ids"].numpy(),
        y=out["hf_tokens"]["input_ids"].numpy(),
    )


def test_minilm_l12_uncased():
    pretrained_model_name = "microsoft/MiniLM-L12-H384-uncased"
    out = compare_tokenizers(pretrained_model_name, [s1, s2, s3])
    np.testing.assert_array_almost_equal(
        x=out["tf_tokens"]["input_ids"].numpy(),
        y=out["hf_tokens"]["input_ids"].numpy(),
    )


# def test_alberta_base():
#     pretrained_model_name = "albert-base-v2"
#     out = compare_tokenizers(pretrained_model_name, [s1, s2, s3])
#     np.testing.assert_array_almost_equal(
#         x=out["tf_tokens"]["input_ids"].numpy(),
#         y=out["hf_tokens"]["input_ids"].numpy(),
#     )
#
#
# def test_bio_clinical_bert():
#     # Trained on Medical MIMIC dataset
#     pretrained_model_name = "emilyalsentzer/Bio_ClinicalBERT"
#     out = compare_tokenizers(pretrained_model_name, [s1, s2, s3])
#     np.testing.assert_array_almost_equal(
#         x=out["tf_tokens"]["input_ids"].numpy(),
#         y=out["hf_tokens"]["input_ids"].numpy(),
#     )
#
#
# def test_bio_pubmed_base_uncased_abstract():
#     pretrained_model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
#     out = compare_tokenizers(pretrained_model_name, [s1, s2, s3])
#     np.testing.assert_array_almost_equal(
#         x=out["tf_tokens"]["input_ids"].numpy(),
#         y=out["hf_tokens"]["input_ids"].numpy(),
#     )
#
#
# def test_all_minilm_l12():
#     pretrained_model_name = "sentence-transformers/all-MiniLM-L12-v2"
#     out = compare_tokenizers(pretrained_model_name, [s1, s2, s3])
#     np.testing.assert_array_almost_equal(
#         x=out["tf_tokens"]["input_ids"].numpy(),
#         y=out["hf_tokens"]["input_ids"].numpy(),
#     )
