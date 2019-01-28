import os
import tarfile
import urllib.request

import colorama

from constants import END_OF_SENTENCE_TOKEN, PADDING_TOKEN, MYPY, CLASS_TO_COLOUR, COLOUR_TEMPLATE

if MYPY:
    from dataloader import Encoder

colorama.init()


def coalesce(value, default_value):
    """https://en.wikipedia.org/wiki/Null_coalescing_operator
    Can't use 'or' cuz of the RuntimeError thrown by pytorch"""
    try:
        if value:
            return value
    except RuntimeError:  # bool(torch.tensor) throws RuntimeError for lists with len > 1
        return value
    return default_value


def read_lines_from_file(filepath, encoding='utf-8')-> list:
    """A wrapper around readlines function"""
    with open(filepath, 'r', encoding=encoding) as f:
        return f.readlines()


def read_lines_from_directory(dirpath)-> list:
    """Reads lines from all the files in the given directory"""
    lines = []
    for filename in os.listdir(dirpath):
        lines.extend(read_lines_from_file(os.path.join(dirpath, filename)))
    return lines


def print_colored_text(ids: list, classes: list, encoder: 'Encoder') -> None:
    """
    Colours the text(retrieved via 'ids') in the console according to the 'classes'
    :param ids: A list of token ids representing the text
    :param classes: A list of values ranging from 0 to 5 representing the colour to be used for ids
    :param encoder: Used to convert an id to token
    """
    assert len(ids) == len(classes), 'Length of ids is %d while length of classes is %d' % (len(ids), len(classes))
    for i, id_ in enumerate(ids):
        if id_ == encoder.get_id(END_OF_SENTENCE_TOKEN) or id_ == encoder.get_id(PADDING_TOKEN):
            break
        print(COLOUR_TEMPLATE.format(colour_code=CLASS_TO_COLOUR[classes[i]], token=encoder.get_token(id_)), end='')
    print('')


def download_file(url: str, file_path, flags: str) -> None:
    data = urllib.request.urlopen(url).read()
    with open(file_path, flags) as f:
        f.write(data)


def decompress_tar_gz(source, destination):
    tar = tarfile.open(source, 'r:gz')
    tar.extractall(path=destination)
    tar.close()
