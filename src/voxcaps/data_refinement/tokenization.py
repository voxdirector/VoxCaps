"""Small tokenization helpers used by pause restoration."""

import re

PUNC_G2P = [
    "…",
    "、",
    "。",
    "《",
    "》",
    "【",
    "】",
    "！",
    "＂",
    "＃",
    "＄",
    "％",
    "＇",
    "＇＇",
    "（",
    "）",
    "＊",
    "，",
    "：",
    "；",
    "？",
    "＼",
    "＾",
    "＿",
    "｀",
    "｛",
    "｝",
    "～",
]

PUNC = PUNC_G2P + list(".,?!~")


def is_english(text):
    return bool(re.fullmatch(r"^[A-Za-z]+$", text))


def isPUNC(c):
    for c_ in c:
        if c_.strip() and c_ in PUNC:
            return True
    return False


def merge_continuous_puncs(word_list, deduplicate_puncs=False):
    ret = []
    for i in range(len(word_list)):
        if len(ret) > 0 and isPUNC(ret[-1]) and isPUNC(word_list[i]):
            if (
                deduplicate_puncs and word_list[i] in ",，"
            ):  # 这个连续的标点符号的第二个很有可能是特殊符号被norm成了','，直接忽略
                continue
            else:
                ret[-1] = "".join([ret[-1], word_list[i]])
        else:
            ret.append(word_list[i])
    return ret


def get_word_list(text, deduplicate_puncs=False, del_unprintable=False):
    # cleaned_text = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fff\s]', '', text)
    cleaned_text = text

    result = []
    current_word = ""

    for c in cleaned_text:
        if del_unprintable and not c.isprintable():
            continue
        if c == " ":
            if current_word:
                result.append(current_word)
                current_word = ""
        elif c.isalnum() and not ("\u4e00" <= c <= "\u9fff"):
            current_word += c
        else:
            if current_word:
                result.append(current_word)
                current_word = ""
            result.append(c)

    if current_word:
        result.append(current_word)

    result = merge_continuous_puncs(result, deduplicate_puncs)

    return result
