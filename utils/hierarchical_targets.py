import re
from typing import List
import torch

CODE_KEYWORDS = {
    "def", "return", "if", "else", "elif", "for", "while", "in",
    "try", "except", "with", "class", "and", "or", "not", "is",
    "None", "True", "False"
}

CODE_BUILTINS = {
    "range", "len", "enumerate", "zip", "sum", "min", "max",
    "sorted", "isinstance", "int", "str", "float", "abs", "round",
    "append", "split", "join"
}

CODE_OPERATORS = {
    "+", "-", "*", "/", "//", "%", "**",
    "=", "==", "!=", "<", "<=", ">", ">=",
    "+=", "-=", "*=", "/=", "%="
}

CODE_DELIMS = {"(", ")", "[", "]", "{", "}", ":", ",", "."}

SMALL_CONSTANTS = {
    "0", "1", "2", "3", "-1", "True", "False", "None"
}


def _norm_piece(piece: str) -> str:
    # Qwen/SentencePiece/BPE-ish cleanup
    return (
        piece.replace("Ġ", "")
             .replace("▁", "")
             .replace("Ċ", "\n")
             .strip()
    )


def is_code_anchor_piece(piece: str) -> bool:
    p = _norm_piece(piece)

    if p == "":
        return False

    if p in CODE_KEYWORDS:
        return True
    if p in CODE_BUILTINS:
        return True
    if p in CODE_OPERATORS:
        return True
    if p in CODE_DELIMS:
        return True
    if p in SMALL_CONSTANTS:
        return True

    # numeric constants
    if re.fullmatch(r"-?\d+(\.\d+)?", p):
        return True

    return False


def make_anchor_mask_from_token_ids(tokenizer, token_ids: List[int]) -> torch.Tensor:
    pieces = [tokenizer.decode([int(tid)], skip_special_tokens=False) for tid in token_ids]
    mask = [is_code_anchor_piece(p) for p in pieces]
    return torch.tensor(mask, dtype=torch.bool)


def make_batch_anchor_mask(tokenizer, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """
    labels: [B, T], where non-training positions may be -100.
    returns bool mask [B, T].
    """
    masks = []
    labels_cpu = labels.detach().cpu()

    for row in labels_cpu:
        row_mask = []
        for tid in row.tolist():
            if tid == ignore_index:
                row_mask.append(False)
            else:
                piece = tokenizer.decode([int(tid)], skip_special_tokens=False)
                row_mask.append(is_code_anchor_piece(piece))
        masks.append(row_mask)

    return torch.tensor(masks, dtype=torch.bool, device=labels.device)