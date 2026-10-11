"""Pause punctuation extracted from the production ASR pipeline."""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple, Union

from .tokenization import get_word_list, isPUNC

_SP_TOKEN = "<|sp|>"
_EN_CONTRACTION_SUFFIXES = {"s", "t", "m", "d", "ll", "re", "ve"}
_EN_APOSTROPHES = {"'", "’"}


def _looks_like_en_word(token: str) -> bool:
    return bool(re.search(r"[A-Za-z]", token or ""))


def _merge_english_contractions_with_str_after(
    words: List[str],
    punc_after: List[str],
) -> Tuple[List[str], List[str]]:
    merged_words: List[str] = []
    merged_after: List[str] = []
    i = 0
    while i < len(words):
        if (
            i + 2 < len(words)
            and words[i + 1] in _EN_APOSTROPHES
            and _looks_like_en_word(words[i])
            and words[i + 2].lower() in _EN_CONTRACTION_SUFFIXES
        ):
            merged_words.append(f"{words[i]}{words[i + 1]}{words[i + 2]}")
            merged_after.append(punc_after[i + 2])
            i += 3
            continue
        if (
            i + 1 < len(words)
            and punc_after[i] in _EN_APOSTROPHES
            and _looks_like_en_word(words[i])
            and words[i + 1].lower() in _EN_CONTRACTION_SUFFIXES
        ):
            merged_words.append(f"{words[i]}{punc_after[i]}{words[i + 1]}")
            merged_after.append(punc_after[i + 1])
            i += 2
            continue
        merged_words.append(words[i])
        merged_after.append(punc_after[i])
        i += 1
    return merged_words, merged_after


def _norm_lang(lang: str) -> str:
    lang = (lang or "").strip().lower()
    if lang in {"en", "english"}:
        return "en"
    if lang in {"zh", "chinese", "cn"}:
        return "zh"
    if "en" in lang and "zh" not in lang:
        return "en"
    if "zh" in lang:
        return "zh"
    return lang or "en"


def _normalize_token_for_match(token: str, lang: str) -> str:
    token = (token or "").strip()
    if not token:
        return ""

    if lang == "en":
        token = token.lower()
        token = re.sub(r"[^a-z0-9]+", "", token)
        return token

    return token


def _extract_words_and_punc_after(asr_text: str) -> Tuple[List[str], List[str]]:
    tokens = get_word_list(asr_text)

    words: List[str] = []
    punc_after: List[str] = []

    i = 0
    while i < len(tokens):
        t = tokens[i]

        if isPUNC(t):
            if len(punc_after) > 0:
                punc_after[-1] += t
            i += 1
            continue

        words.append(t)
        punc_after.append("")
        i += 1

        while i < len(tokens) and isPUNC(tokens[i]):
            punc_after[-1] += tokens[i]
            i += 1

    return words, punc_after


def _format_alignment_word_group(
    words: Sequence[str],
    punc_after: Sequence[str],
    start: int,
    end: int,
    *,
    lang: str,
) -> str:
    """
    Rebuild the original text-side token group that corresponds to one aligner token.

    The grouping decision is made from normalized text, but the emitted word should
    preserve the user's text as much as possible, e.g. can + ' + t -> can't.
    """
    if start == end:
        return words[start]

    lang = _norm_lang(lang)
    pieces: List[str] = []
    prev_norm = ""

    for idx in range(start, end + 1):
        word = words[idx]
        word_norm = _normalize_token_for_match(word, lang)

        if pieces and lang == "en" and not punc_after[idx - 1]:
            # If two real English word pieces were separated by whitespace in the
            # source text, get_word_list has already discarded that whitespace.
            # Reinsert a space unless an empty-normalized connector such as "'" is
            # between them.
            if prev_norm and word_norm:
                pieces.append(" ")

        pieces.append(word)
        if idx < end and punc_after[idx]:
            pieces.append(punc_after[idx])

        prev_norm = word_norm

    return "".join(pieces)


def _find_alignment_group_end(
    words: Sequence[str],
    start: int,
    target_norm: str,
    *,
    lang: str,
) -> Optional[int]:
    if not target_norm:
        return None

    accum = ""
    for end in range(start, len(words)):
        piece_norm = _normalize_token_for_match(words[end], lang)
        accum += piece_norm

        if accum == target_norm:
            return end

        # Empty-normalized connector tokens, e.g. apostrophes, are allowed to sit
        # inside a group without advancing accum.
        if accum and not target_norm.startswith(accum):
            return None

    return None


def _group_words_by_aligned_tokens(
    words: Sequence[str],
    punc_after: Sequence[str],
    aligned_words: Sequence[str],
    *,
    lang: str,
) -> Optional[Tuple[List[str], List[str]]]:
    """
    Convert text-side tokens to aligner token boundaries.

    This is intentionally driven by the aligner output rather than by English
    contraction heuristics. For example, if the aligner returns one token
    "can't", text tokens can + ' + t are grouped into one emitted token.
    """
    if not words or not aligned_words:
        return None

    grouped_words: List[str] = []
    grouped_punc_after: List[str] = []
    word_pos = 0

    for aligned_word in aligned_words:
        target_norm = _normalize_token_for_match(aligned_word, lang)
        end = _find_alignment_group_end(words, word_pos, target_norm, lang=lang)
        if end is None:
            return None

        grouped_words.append(
            _format_alignment_word_group(
                words,
                punc_after,
                word_pos,
                end,
                lang=lang,
            )
        )
        grouped_punc_after.append(punc_after[end])
        word_pos = end + 1

    if word_pos != len(words):
        return None

    return grouped_words, grouped_punc_after


def _choose_pause_punct(
    pause_s: float,
    lang: str,
    *,
    sp_min_s: float,
    comma_min_s: float,
    period_min_s: float,
    sp_token: str = _SP_TOKEN,
    enable_sp_token: bool = True,
) -> str:
    if pause_s is None:
        return ""

    pause_s = float(pause_s)

    # 如果不插入 sp：那么 sp 阈值及以下的停顿都忽略。
    # 等价于：pause < comma_min_s 时一律不加任何符号。
    if not enable_sp_token:
        if pause_s < comma_min_s:
            return ""

    if pause_s < sp_min_s:
        return ""

    if pause_s < comma_min_s:
        return sp_token

    if pause_s < period_min_s:
        return "，" if lang == "zh" else ","

    return "。" if lang == "zh" else "."


def _join_tokens(tokens: Sequence[str], lang: str, *, sp_token: str = _SP_TOKEN) -> str:
    lang = _norm_lang(lang)

    if lang == "zh":
        out: List[str] = []
        for tok in tokens:
            if tok:
                out.append(tok)
        return "".join(out)

    out: List[str] = []
    for i, tok in enumerate(tokens):
        if not tok:
            continue

        nxt = tokens[i + 1] if i + 1 < len(tokens) else None

        if tok == sp_token:
            if len(out) > 0 and not out[-1].endswith(" "):
                out.append(" ")
            out.append(tok)
            if nxt and (not isPUNC(nxt)):
                out.append(" ")
            continue

        if isPUNC(tok):
            if len(out) > 0 and out[-1].endswith(" "):
                out[-1] = out[-1].rstrip(" ")
            out.append(tok)
            if nxt and (nxt != sp_token) and (not isPUNC(nxt)):
                out.append(" ")
            continue

        if len(out) > 0 and (not out[-1].endswith(" ")):
            prev = out[-1]
            if prev and (not isPUNC(prev)) and prev != sp_token:
                out.append(" ")
        out.append(tok)

    return "".join(out).strip()


def _prefer_asr_punct_when_pause(
    existing: str,
    lang: str,
    insert_p: str,
    *,
    sp_token: str = _SP_TOKEN,
) -> str:
    """
    当 alignment/pause 已经判断“这里要插标点”(insert_p 非空) 后：

    - insert_p == <|sp|>：强制返回 <|sp|>（短停顿优先）
    - 否则如果 ASR 原本就有标点 existing：优先返回 existing（做一点中英文符号归一化）
    - 否则：回退用 insert_p（也就是按 pause bucket 推出来的逗号/句号）
    """
    lang = _norm_lang(lang)
    existing = (existing or "").strip()

    if not insert_p:
        return ""

    if insert_p == sp_token:
        return sp_token

    if not existing:
        return insert_p

    if lang == "zh":
        existing = (
            existing.replace(",", "，").replace(".", "。").replace("?", "？").replace("!", "！")
        )
        if "..." in existing and "…" not in existing:
            existing = existing.replace("...", "…")
        return existing

    existing = existing.replace("，", ",").replace("。", ".").replace("？", "?").replace("！", "!")
    if "…" in existing and "..." not in existing:
        existing = existing.replace("…", "...")
    if "、" in existing:
        existing = existing.replace("、", ",")
    return existing


def punctuate_by_alignment_pause(
    *,
    asr_text: str,
    align_items: Sequence,
    lang: str,
    sp_token: str = _SP_TOKEN,
    enable_sp_token: bool = True,
    sp_min_s: float = 0.08,
    comma_min_s: float = 0.18,
    period_min_s: float = 0.45,
    use_asr_punctuation: bool = True,
    require_positive_alignment_duration_for_pause: bool = False,
    keep_asr_punc_without_pause: bool = False,
    keep_final_asr_punc: bool = True,
    audio_duration_s: Union[float, None] = None,
) -> str:
    lang = _norm_lang(lang)
    if not asr_text:
        return asr_text

    raw_words, raw_punc_after = _extract_words_and_punc_after(asr_text)
    aligned_words = [getattr(it, "text", "") for it in align_items]
    grouped = _group_words_by_aligned_tokens(
        raw_words,
        raw_punc_after,
        aligned_words,
        lang=lang,
    )
    if grouped is not None:
        words, punc_after = grouped
    else:
        # Fallback keeps the previous forgiving behavior for imperfect aligner
        # output, but still avoids silently dropping unmatched text.
        words, punc_after = _merge_english_contractions_with_str_after(raw_words, raw_punc_after)

    if len(words) == len(aligned_words):
        idx_map = list(range(len(words)))
    else:
        idx_map = []
        w_i = 0
        for aw in aligned_words:
            aw_n = _normalize_token_for_match(aw, lang)
            while w_i < len(words) and _normalize_token_for_match(words[w_i], lang) != aw_n:
                w_i += 1
            if w_i >= len(words):
                break
            idx_map.append(w_i)
            w_i += 1

        if len(idx_map) != len(aligned_words):
            n = min(len(words), len(aligned_words))
            idx_map = list(range(n))
            align_items = list(align_items)[:n]

    out_tokens: List[str] = []
    last_emitted_word_idx = -1
    for i, it in enumerate(align_items):
        w_idx = idx_map[i]
        if w_idx > last_emitted_word_idx + 1:
            for missing_idx in range(last_emitted_word_idx + 1, w_idx):
                out_tokens.append(words[missing_idx])
                if use_asr_punctuation and punc_after[missing_idx]:
                    out_tokens.append(punc_after[missing_idx])

        out_tokens.append(words[w_idx])
        last_emitted_word_idx = w_idx

        # v1 clean-ASR fine-tuning can ignore all sentence-boundary punctuation
        # from ASR while still preserving lexical punctuation already rebuilt
        # inside a grouped word, e.g. the apostrophe in "can't".
        existing = punc_after[w_idx] if use_asr_punctuation else ""

        # ====== 句尾逻辑：用尾静音时长决定 ======
        if i + 1 >= len(align_items):
            if audio_duration_s is not None:
                last_end = float(getattr(it, "end_time", 0.0))
                tail_silence_s = float(audio_duration_s) - last_end
                if tail_silence_s < 0:
                    tail_silence_s = 0.0

                # 把“尾静音”当成最后一个 pause
                last_start = float(getattr(it, "start_time", 0.0))
                if require_positive_alignment_duration_for_pause and last_end <= last_start:
                    insert_p = ""
                else:
                    insert_p = _choose_pause_punct(
                        tail_silence_s,
                        lang,
                        sp_min_s=sp_min_s,
                        comma_min_s=comma_min_s,
                        period_min_s=period_min_s,
                        sp_token=sp_token,
                        enable_sp_token=enable_sp_token,
                    )

                # 是否有句尾标点由 insert_p 是否为空决定；
                # 具体符号尽量沿用 ASR 的 existing（除非 insert_p 是 <|sp|>）
                if insert_p:
                    out_tokens.append(
                        _prefer_asr_punct_when_pause(
                            existing,
                            lang,
                            insert_p,
                            sp_token=sp_token,
                        )
                    )
            else:
                # 拿不到音频时长：保底（不影响你“完全按尾静音”的主路径）
                if keep_final_asr_punc and existing:
                    out_tokens.append(existing)
            continue
        # ====== 句尾逻辑结束 ======

        next_it = align_items[i + 1]
        current_start = float(getattr(it, "start_time", 0.0))
        current_end = float(getattr(it, "end_time", 0.0))
        next_start = float(getattr(next_it, "start_time", 0.0))
        next_end = float(getattr(next_it, "end_time", 0.0))
        if require_positive_alignment_duration_for_pause and (
            current_end <= current_start or next_end <= next_start
        ):
            insert_p = ""
        else:
            pause_s = next_start - current_end
            insert_p = _choose_pause_punct(
                pause_s,
                lang,
                sp_min_s=sp_min_s,
                comma_min_s=comma_min_s,
                period_min_s=period_min_s,
                sp_token=sp_token,
                enable_sp_token=enable_sp_token,
            )

        if keep_asr_punc_without_pause:
            if existing:
                out_tokens.append(existing)
                continue
            if insert_p:
                out_tokens.append(insert_p)
            continue

        if insert_p:
            out_tokens.append(
                _prefer_asr_punct_when_pause(
                    existing,
                    lang,
                    insert_p,
                    sp_token=sp_token,
                )
            )

    for missing_idx in range(last_emitted_word_idx + 1, len(words)):
        out_tokens.append(words[missing_idx])
        if use_asr_punctuation and punc_after[missing_idx]:
            out_tokens.append(punc_after[missing_idx])

    return _join_tokens(out_tokens, lang, sp_token=sp_token)
