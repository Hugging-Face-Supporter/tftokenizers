from tftokenizers.file import get_filename_and_type_from_path, get_vocab_from_path


def test_get_file_from_path():
    filename = "vocab"
    path_txt = "saved_tokenizers/bert-base-uncased"
    path_json = "saved_tokenizers/allenai_longformer-base-4096"

    # Find vocab file when as txt
    file_path, FILETYPE = get_filename_and_type_from_path(path_txt, filename)
    assert file_path == f"{path_txt}/{filename}.txt"
    assert FILETYPE == "txt"

    # Find vocab file when as json
    file_path, FILETYPE = get_filename_and_type_from_path(path_json, filename)
    assert file_path == f"{path_json}/{filename}.json"
    assert FILETYPE == "json"


def test_get_vocab_from_file_txt():
    path = "saved_tokenizers/bert-base-uncased"
    vocab = get_vocab_from_path(path)
    assert [vocab[:2], vocab[-2:]] == [["[PAD]", "[unused0]"], ["##？", "##～"]]


def test_get_vocab_from_file_json():
    path = "saved_tokenizers/allenai_longformer-base-4096"  # json vocab file
    vocab = get_vocab_from_path(path)
    assert [vocab[:2], vocab[-2:]] == [["<s>", "<pad>"], ["madeupword0002", "<mask>"]]


def get_vocab_from_config_file_json():
    pass
