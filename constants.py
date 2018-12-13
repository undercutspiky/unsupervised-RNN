import platform

# Tokens
END_OF_SENTENCE_TOKEN = '<EOS>'
OUT_OF_VOCAB_TOKEN = '<OOV>'
PADDING_TOKEN = '<PAD>'

# Templates for printing colourful text on console
if platform.system() == 'Windows':
    COLOUR_TEMPLATE = "\033[{colour_code}m{token}"
else:
    COLOUR_TEMPLATE = "\033[0;{colour_code};0m{token}"

# For type hinting
MYPY = False
