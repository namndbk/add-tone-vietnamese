import unidecode


def remove_accent(text):
    return unidecode.unidecode(text.lower().strip())
