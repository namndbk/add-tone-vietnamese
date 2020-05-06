import re
import string


def remove_accent(text):
    return unidecode.unidecode(text.lower().strip())
