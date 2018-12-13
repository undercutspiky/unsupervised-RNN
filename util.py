import os
import urllib.request
import tarfile

from constants import END_OF_SENTENCE_TOKEN, PADDING_TOKEN, COLOUR_TEMPLATE, MYPY

if MYPY:
    from dataloader import Encoder


def coalesce(value, default_value):
    if value is not None:
        return value
    return default_value


def read_lines_from_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.readlines()


def read_lines_from_directory(dirpath):
    lines = []
    for filename in os.listdir(dirpath):
        lines.extend(read_lines_from_file(os.path.join(dirpath, filename)))
    return lines


def print_colored_text(ids: list, classes: list, encoder: 'Encoder') -> None:
    color_codes = list(range(31, 37))
    assert len(ids) == len(classes), 'Length of ids is %d while length of classes is %d' % (len(ids), len(classes))
    for i, id_ in enumerate(ids):
        if id_ == encoder.get_id(END_OF_SENTENCE_TOKEN) or id_ == encoder.get_id(PADDING_TOKEN):
            break
        print(COLOUR_TEMPLATE.format(colour_code=color_codes[classes[i]], token=encoder.get_token(id_)), end='')
    # Reset the colour to black for the rest of the print statements
    print(COLOUR_TEMPLATE.format(colour_code=30, token='\n'))


def download_file(url: str, file_path, flags) -> None:
    data = urllib.request.urlopen(url).read()
    with open(file_path, flags) as f:
        f.write(data)


def decompress_tar_gz(source, destination):
    tar = tarfile.open(source, 'r:gz')
    tar.extractall(path=destination)
    tar.close()
