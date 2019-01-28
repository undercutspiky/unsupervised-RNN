from colorama import Fore, Back

# Tokens
END_OF_SENTENCE_TOKEN = '<EOS>'
OUT_OF_VOCAB_TOKEN = '<OOV>'
PADDING_TOKEN = '<PAD>'

# Map from cluster numbers to colours
CLASS_TO_COLOUR = {0: Fore.RED, 1: Fore.GREEN, 2: Fore.CYAN, 3: Fore.YELLOW, 4: Fore.MAGENTA, 5: Fore.BLUE}
COLOUR_TEMPLATE = Back.BLACK + '{colour_code}{token}' + Fore.RESET + Back.RESET

# For type hinting
MYPY = False
