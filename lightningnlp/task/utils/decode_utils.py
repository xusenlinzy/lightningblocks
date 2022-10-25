from typing import List, Tuple


def get_entities(seq: List[str], *, suffix: bool = False) -> List[Tuple[str, int, int]]:
    """Gets entities from sequence.
    Args:
        seq: sequence of labels.
        suffix: if type as the suffix
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> get_entities(seq)
        [('PER', 0, 1), ('LOC', 3, 3)]
    """
    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        if suffix:
            tag = chunk[-1]
            type_ = chunk.split('-')[0]
        else:
            tag = chunk[0]
            type_ = chunk.split('-')[-1]

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i - 1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return chunks


def end_of_chunk(prev_tag: str, tag: str, prev_type: str, type_: str) -> bool:
    """Checks if a chunk ended between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.
    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag in {'B', 'I'} and tag in {'B', 'S', 'O'} or prev_tag not in ['B', 'I'] and prev_tag in {'E', 'S'}:
        chunk_end = True
    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag: str, tag: str, prev_type: str, type_: str) -> bool:
    """Checks if a chunk started between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.
    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag in {'B', 'S'}:
        chunk_start = True
    if prev_tag in {'E', 'S', 'O'} and tag in {'E', 'I'}:
        chunk_start = True
    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start


def is_overlapped(chunk1: tuple, chunk2: tuple):
    (_, s1, e1), (_, s2, e2) = chunk1, chunk2
    return s1 < e2 and s2 < e1


def is_nested(chunk1: tuple, chunk2: tuple):
    (_, s1, e1), (_, s2, e2) = chunk1, chunk2
    return (s1 <= s2 and e2 <= e1) or (s2 <= s1 and e1 <= e2)


def is_clashed(chunk1: tuple, chunk2: tuple, allow_nested: bool = True):
    if allow_nested:
        return is_overlapped(chunk1, chunk2) and not is_nested(chunk1, chunk2)
    else:
        return is_overlapped(chunk1, chunk2)


def filter_clashed_by_priority(chunks, allow_nested: bool = True):
    filtered_chunks = []
    for ck in chunks:
        if all(not is_clashed(ck, ex_ck, allow_nested=allow_nested) for ex_ck in filtered_chunks):
            filtered_chunks.append(ck)

    return filtered_chunks
