import numpy as np
import tensorflow as tf

from tftokenizers.file import get_vocab_from_path, load_json
from tftokenizers.types import PROCESSING_STEP, TemplateType
from tftokenizers.utils import map_special_tokens_to_ids

PATH = "saved_tokenizers/bert-base-uncased"
vocab = get_vocab_from_path(PATH)
config_huggingface = load_json(f"{PATH}/tokenizer.json")
config = load_json(f"{PATH}/tokenizer_config.json")


special_tokens = map_special_tokens_to_ids(vocab, config)


def test_parse_template_for_single_sentences():
    SPECIAL_TOKEN_ID_FOR_TEMPLATE = [101, 102]
    bs = tf.ragged.constant(
        [
            [1, 2, 102, 10, 20],
            [3, 4, 102, 30, 40, 50, 60, 70, 80],
            [5, 6, 7, 8, 9, 10, 70],
        ],
        tf.int64,
    )

    bs_after = tf.ragged.constant(
        [
            [101, 1, 2, 102, 10, 20, 102],
            [101, 3, 4, 102, 30, 40, 50, 60, 70, 80, 102],
            [101, 5, 6, 7, 8, 9, 10, 70, 102],
        ],
        tf.int64,
    )

    # Get instructions for how to parse the template
    num_sentences = bs.bounding_shape()[0]
    template = config_huggingface["post_processor"]
    processing_steps = template[TemplateType.SINGLE]
    num_steps = len(processing_steps)

    def get_token_id_from_current_step_in_template(
        instruction_step: PROCESSING_STEP,
    ) -> tf.Tensor:
        token_symbol: str = list(instruction_step.values())[0]["id"]
        token_id: tf.Tensor = special_tokens.get(token_symbol)
        return token_id

    # Build AST from template instruction
    for step in range(num_steps - 1):
        left: PROCESSING_STEP = processing_steps[step]
        right: PROCESSING_STEP = processing_steps[step + 1 - num_steps]
        left_token = get_token_id_from_current_step_in_template(left)
        right_token = get_token_id_from_current_step_in_template(right)

        if left != right:
            left_sentence = bs
            right_sentence = bs

            if "SpecialToken" in left:
                left_sentence = tf.fill([num_sentences, 1], left_token)
            elif "SpecialToken" in right:
                right_sentence = tf.fill([num_sentences, 1], right_token)
            bs = tf.concat([left_sentence, right_sentence], axis=1)

    # Only needed for numpy test framework
    bs = bs.to_tensor().numpy()
    bs_after = bs_after.to_tensor().numpy()

    np.testing.assert_array_almost_equal(x=bs, y=bs_after)
