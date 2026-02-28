#!/usr/bin/env python3

import re
import numpy as np
from typing import Dict, Tuple, List, Any, Optional

try:
    from rapidfuzz.distance import Levenshtein as Lev
except ImportError:
    raise ImportError("Please install rapidfuzz: pip install rapidfuzz")

# UNC markers
UNC_START = "<C>"
UNC_END = "</C>"
UNC_PATTERN = re.compile(re.escape(UNC_START) + r"(.*?)" + re.escape(UNC_END), re.DOTALL)

# Reward parameters
CORRECT_REWARD = 1.0
ERROR_PENALTY_OUT_UNC = 1.0
ERROR_PENALTY_IN_UNC = 0.3

def remove_unc_and_create_mask(text: str) -> Tuple[str, np.ndarray, List[Tuple[int, int]], bool]:
    """Remove UNC markers and create character-level mask"""
    has_format_error = False
    keep_chars = [True] * len(text)

    i = 0
    last_start_pos = -1

    while i < len(text):
        if text[i:i+len(UNC_START)] == UNC_START:
            if last_start_pos != -1:
                for j in range(last_start_pos, last_start_pos + len(UNC_START)):
                    keep_chars[j] = False
                has_format_error = True
            last_start_pos = i
            i += len(UNC_START)

        elif text[i:i+len(UNC_END)] == UNC_END:
            if last_start_pos == -1:
                for j in range(i, i + len(UNC_END)):
                    keep_chars[j] = False
                has_format_error = True
                i += len(UNC_END)
            else:
                next_pos = i + len(UNC_END)
                if next_pos + len(UNC_END) <= len(text) and text[next_pos:next_pos+len(UNC_END)] == UNC_END:
                    for j in range(next_pos, next_pos + len(UNC_END)):
                        keep_chars[j] = False
                    has_format_error = True
                last_start_pos = -1
                i += len(UNC_END)
        else:
            i += 1

    if last_start_pos != -1:
        for j in range(last_start_pos, last_start_pos + len(UNC_START)):
            if j < len(keep_chars):
                keep_chars[j] = False
        has_format_error = True

    cleaned_text = ""
    for i, keep in enumerate(keep_chars):
        if keep:
            cleaned_text += text[i]

    all_matches = list(UNC_PATTERN.finditer(cleaned_text))
    regions = []
    offset = 0
    for match in all_matches:
        start = match.start() - offset
        content = match.group(1)
        end = start + len(content)
        regions.append((start, end))
        offset += len(UNC_START) + len(UNC_END)

    clean_text = UNC_PATTERN.sub(r"\1", cleaned_text)
    mask = np.zeros(len(clean_text), dtype=bool)
    for start, end in regions:
        if start < len(mask) and end <= len(mask):
            mask[start:end] = True

    return clean_text, mask, regions, has_format_error

def compute_edit_operations_stats(ops: List, mask: np.ndarray) -> Dict[str, int]:
    """Compute detailed statistics of edit operations"""
    stats = {
        'substitutions': 0,
        'insertions': 0,
        'deletions': 0,
        'errors_in_unc': 0,
        'errors_out_unc': 0,
        'sub_in_unc': 0,
        'sub_out_unc': 0,
        'ins_in_unc': 0,
        'ins_out_unc': 0,
        'del_in_unc': 0,
        'del_out_unc': 0,
        'ins_chars_in_unc': 0,
        'ins_chars_out_unc': 0,
        'del_chars_in_unc': 0,
        'del_chars_out_unc': 0,
    }

    for op in ops:
        if op.tag == "replace":
            stats['substitutions'] += 1
            if op.src_pos < len(mask) and mask[op.src_pos]:
                stats['errors_in_unc'] += 1
                stats['sub_in_unc'] += 1
            else:
                stats['errors_out_unc'] += 1
                stats['sub_out_unc'] += 1

        elif op.tag == "insert":
            stats['insertions'] += 1
            is_in_unc = False

            if op.src_pos == 0:
                if len(mask) > 0 and mask[0]:
                    is_in_unc = True
            elif op.src_pos >= len(mask):
                if len(mask) > 0 and mask[-1]:
                    is_in_unc = True
            else:
                if mask[op.src_pos]:
                    is_in_unc = True
                elif op.src_pos > 0 and mask[op.src_pos - 1]:
                    is_in_unc = True

            if is_in_unc:
                stats['errors_in_unc'] += 1
                stats['ins_in_unc'] += 1
                stats['ins_chars_in_unc'] += 1
            else:
                stats['errors_out_unc'] += 1
                stats['ins_out_unc'] += 1
                stats['ins_chars_out_unc'] += 1

        elif op.tag == "delete":
            stats['deletions'] += 1
            if op.src_pos < len(mask) and mask[op.src_pos]:
                stats['errors_in_unc'] += 1
                stats['del_in_unc'] += 1
                stats['del_chars_in_unc'] += 1
            else:
                stats['errors_out_unc'] += 1
                stats['del_out_unc'] += 1
                stats['del_chars_out_unc'] += 1

    return stats

def normalize_text(text: str) -> str:
    """Normalize text"""
    import uuid
    placeholder_start = f"__PLACEHOLDER_START_{uuid.uuid4().hex.upper()}__"
    placeholder_end = f"__PLACEHOLDER_END_{uuid.uuid4().hex.upper()}__"

    text = text.replace('<C>', placeholder_start)
    text = text.replace('</C>', placeholder_end)
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = text.lower()
    text = ' '.join(text.split())
    text = text.replace(placeholder_start.lower(), '<C>')
    text = text.replace(placeholder_end.lower(), '</C>')

    return text

def tokenize_words(text: str) -> List[Tuple[str, int, int]]:
    """Tokenize text into words with positions"""
    words = []
    i = 0
    while i < len(text):
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text):
            break
        start = i
        while i < len(text) and not text[i].isspace():
            i += 1
        if start < i:
            words.append((text[start:i], start, i))
    return words

def compute_word_level_unc_metrics(pred_text: str, gt_text: str, mask: np.ndarray) -> Dict[str, Any]:
    """Compute word-level UNC metrics"""
    pred_words = tokenize_words(pred_text)
    gt_words = tokenize_words(gt_text)

    pred_word_texts = [w[0] for w in pred_words]
    gt_word_texts = [w[0] for w in gt_words]

    word_ops = Lev.editops(pred_word_texts, gt_word_texts)

    words_in_unc_correct = 0
    words_in_unc_error = 0
    words_out_unc_correct = 0
    words_out_unc_error = 0

    gt_word_status = []
    pred_used = [False] * len(pred_words)
    gt_used = [False] * len(gt_words)

    alignment = []
    word_opcodes = Lev.opcodes(pred_word_texts, gt_word_texts)

    for opcode in word_opcodes:
        if opcode.tag == "equal":
            for i in range(opcode.src_start, opcode.src_end):
                gt_idx = opcode.dest_start + (i - opcode.src_start)
                alignment.append((gt_idx, i, 'match', True))
                pred_used[i] = True
                gt_used[gt_idx] = True
        elif opcode.tag == "replace":
            for i in range(opcode.src_start, opcode.src_end):
                gt_idx = opcode.dest_start + (i - opcode.src_start)
                alignment.append((gt_idx, i, 'replace', False))
                pred_used[i] = True
                gt_used[gt_idx] = True
        elif opcode.tag == "insert":
            for gt_idx in range(opcode.dest_start, opcode.dest_end):
                alignment.append((gt_idx, -1, 'insert', False))
                gt_used[gt_idx] = True

    for gt_idx, pred_idx, operation, is_correct in alignment:
        if pred_idx == -1:
            if gt_idx == 0:
                estimated_pos = 0
            elif gt_idx >= len(pred_words):
                estimated_pos = pred_words[-1][2] if len(pred_words) > 0 else 0
            else:
                prev_pred_idx = -1
                for prev_gt_idx, prev_pred_idx_temp, _, _ in alignment:
                    if prev_gt_idx < gt_idx and prev_pred_idx_temp != -1:
                        prev_pred_idx = prev_pred_idx_temp
                estimated_pos = pred_words[prev_pred_idx][2] if prev_pred_idx != -1 and prev_pred_idx < len(pred_words) else 0
            pred_position = (estimated_pos, estimated_pos)
        else:
            pred_position = pred_words[pred_idx][1:3]

        is_in_unc = False
        if pred_position:
            pos = pred_position[0]
            if pred_idx == -1:
                if pos == 0:
                    if len(mask) > 0 and mask[0]:
                        is_in_unc = True
                elif pos >= len(mask):
                    if len(mask) > 0 and mask[-1]:
                        is_in_unc = True
                else:
                    check_forward = False
                    check_backward = False
                    for next_gt_idx, next_pred_idx, _, _ in alignment:
                        if next_gt_idx > gt_idx and next_pred_idx != -1:
                            if next_pred_idx < len(pred_words):
                                next_word_start = pred_words[next_pred_idx][1]
                                next_word_end = pred_words[next_pred_idx][2]
                                if next_word_end <= len(mask) and np.any(mask[next_word_start:next_word_end]):
                                    check_forward = True
                            break

                    for prev_gt_idx, prev_pred_idx, _, _ in reversed(alignment[:gt_idx]):
                        if prev_pred_idx != -1:
                            if prev_pred_idx < len(pred_words):
                                prev_word_start = pred_words[prev_pred_idx][1]
                                prev_word_end = pred_words[prev_pred_idx][2]
                                if prev_word_end <= len(mask) and np.any(mask[prev_word_start:prev_word_end]):
                                    check_backward = True
                            break

                    if check_forward or check_backward:
                        is_in_unc = True

                    if not is_in_unc:
                        if pos < len(mask) and mask[pos]:
                            is_in_unc = True
                        elif pos > 0 and pos - 1 < len(mask) and mask[pos - 1]:
                            is_in_unc = True
            else:
                if pred_position[1] <= len(mask):
                    is_in_unc = np.any(mask[pred_position[0]:pred_position[1]])

        gt_word_status.append((is_correct, is_in_unc))

    extra_words_in_unc = 0
    extra_words_out_unc = 0

    for pred_idx in range(len(pred_words)):
        if not pred_used[pred_idx]:
            pred_start, pred_end = pred_words[pred_idx][1:3]
            is_in_unc = False
            if pred_end <= len(mask):
                is_in_unc = np.any(mask[pred_start:pred_end])

            if is_in_unc:
                extra_words_in_unc += 1
            else:
                extra_words_out_unc += 1

    for is_correct, is_in_unc in gt_word_status:
        if is_in_unc:
            if is_correct:
                words_in_unc_correct += 1
            else:
                words_in_unc_error += 1
        else:
            if is_correct:
                words_out_unc_correct += 1
            else:
                words_out_unc_error += 1

    words_in_unc_error += extra_words_in_unc
    words_out_unc_error += extra_words_out_unc

    total_words_in_unc = words_in_unc_correct + words_in_unc_error
    total_words_out_unc = words_out_unc_correct + words_out_unc_error

    return {
        'words_in_unc_correct': words_in_unc_correct,
        'words_in_unc_error': words_in_unc_error,
        'total_words_in_unc': total_words_in_unc,
        'words_out_unc_correct': words_out_unc_correct,
        'words_out_unc_error': words_out_unc_error,
        'total_words_out_unc': total_words_out_unc,
        'total_pred_words': len(pred_words),
        'total_gt_words': len(gt_words),
        'total_word_ops': len(word_ops)
    }

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Compute OCR reward score"""

    if not solution_str:
        solution_str = "}"

    if not solution_str or not ground_truth:
        return {'score': 0.0}

    ground_truth = normalize_text(ground_truth)
    solution_str = normalize_text(solution_str)

    clean_pred, mask, regions, has_format_error = remove_unc_and_create_mask(solution_str)

    ops = Lev.editops(clean_pred, ground_truth)
    distance = len(ops)

    stats = compute_edit_operations_stats(ops, mask)

    gt_len = len(ground_truth)
    cer = distance / gt_len if gt_len > 0 else 0.0
    accuracy = 1.0 - cer

    unc_count = len(regions)
    has_unc = unc_count > 0

    unc_chars = int(mask.sum())
    non_unc_chars = len(mask) - unc_chars

    unc_chars_adjusted = unc_chars + stats['ins_chars_in_unc'] - stats['del_chars_in_unc']
    non_unc_chars_adjusted = non_unc_chars + stats['ins_chars_out_unc'] - stats['del_chars_out_unc']

    if has_unc and unc_chars_adjusted > 0:
        unc_error_rate = stats['errors_in_unc'] / max(1, unc_chars_adjusted)
        if non_unc_chars_adjusted > 0:
            non_unc_error_rate = stats['errors_out_unc'] / max(1, non_unc_chars_adjusted)
            unc_gap = unc_error_rate - non_unc_error_rate
        else:
            non_unc_error_rate = 0.0
            unc_gap = unc_error_rate
    else:
        unc_error_rate = 0.0
        non_unc_error_rate = stats['errors_out_unc'] / max(1, non_unc_chars_adjusted)
        unc_gap = 0.0

    unc_correct_use = 0
    if has_unc and not has_format_error:
        unc_correct_use = 1

    # UNC precision and recall
    unc_chars_adjusted_f1 = unc_chars + stats['ins_chars_in_unc']
    unc_precision = stats['errors_in_unc'] / max(1, unc_chars_adjusted_f1) if unc_chars_adjusted_f1 > 0 else 0
    unc_recall = stats['errors_in_unc'] / max(1, distance) if distance > 0 else 0
    unc_f1 = 2 * (unc_precision * unc_recall) / max(0.0001, 1 * unc_precision + unc_recall)

    # Word-level metrics
    word_metrics = {}
    if has_unc and len(mask) > 0:
        word_metrics = compute_word_level_unc_metrics(clean_pred, ground_truth, mask)
    else:
        pred_words = tokenize_words(clean_pred)
        gt_words = tokenize_words(ground_truth)
        pred_word_texts = [w[0] for w in pred_words]
        gt_word_texts = [w[0] for w in gt_words]
        word_ops = Lev.editops(pred_word_texts, gt_word_texts)

        word_opcodes = Lev.opcodes(pred_word_texts, gt_word_texts)
        total_correct_words = 0
        total_error_words = 0

        for opcode in word_opcodes:
            if opcode.tag == "equal":
                total_correct_words += (opcode.src_end - opcode.src_start)
            elif opcode.tag in ["replace", "delete"]:
                total_error_words += (opcode.src_end - opcode.src_start)

        word_metrics = {
            'words_in_unc_correct': 0,
            'words_in_unc_error': 0,
            'total_words_in_unc': 0,
            'words_out_unc_correct': total_correct_words,
            'words_out_unc_error': total_error_words,
            'total_words_out_unc': total_correct_words + total_error_words,
            'total_pred_words': len(pred_words),
            'total_gt_words': len(gt_words),
            'total_word_ops': len(word_ops)
        }

    # Word-level UNC metrics
    total_unc_words = word_metrics.get('words_in_unc_correct', 0) + word_metrics.get('words_in_unc_error', 0)
    word_unc_error_rate = word_metrics.get('words_in_unc_error', 0) / max(1, total_unc_words) if total_unc_words > 0 else 0
    word_non_unc_error_rate = word_metrics.get('words_out_unc_error', 0) / max(1, word_metrics.get('total_words_out_unc', 0)) if word_metrics.get('total_words_out_unc', 0) > 0 else 0
    word_unc_gap = word_unc_error_rate - word_non_unc_error_rate

    word_unc_precision = word_metrics.get('words_in_unc_error', 0) / max(1, total_unc_words) if total_unc_words > 0 else 0
    word_accuracy = 1 - word_metrics.get('total_word_ops', 0)/max(1, word_metrics.get('total_gt_words', 0))
    word_unc_recall = word_metrics.get('words_in_unc_error', 0) / max(1, word_metrics.get('words_in_unc_error', 0) + word_metrics.get('words_out_unc_error', 0)) if word_metrics.get('words_in_unc_error', 0) + word_metrics.get('words_out_unc_error', 0) > 0 else 0

    word_unc_f1 = 2 * (word_unc_precision * word_unc_recall) / max(0.0001, 1 * word_unc_precision + word_unc_recall)
    word_unc_f_want = 1.25 * (word_unc_precision * word_unc_recall) / max(0.0001, 0.25 * word_unc_precision + word_unc_recall)

    # Final word-level reward
    if word_metrics.get('total_pred_words', 0)/max(1, word_metrics.get('total_gt_words', 0)) > 1.3 or word_metrics.get('total_gt_words', 0)/max(1, word_metrics.get('total_pred_words', 0)) > 1.3:
        word_final_f1reward = 0.9*word_metrics.get('total_word_ops', 0)/word_metrics.get('total_gt_words', 0)*word_unc_f_want * 0.5 + word_accuracy
    else:
        word_final_f1reward = 0.9*word_metrics.get('total_word_ops', 0)/word_metrics.get('total_gt_words', 0)*word_unc_f_want + word_accuracy

    word_final_f1reward = max(0.0, word_final_f1reward)

    result = {
        'score': word_final_f1reward,
        'unc_f1': unc_f1,
        'unc_recall': unc_recall,
        'unc_precision': unc_precision,
        'accuracy': accuracy,
        'cer': cer,
        'total_errors': distance,
        'substitutions': stats['substitutions'],
        'insertions': stats['insertions'],
        'deletions': stats['deletions'],
        'errors_in_unc': stats['errors_in_unc'],
        'errors_out_unc': stats['errors_out_unc'],
        'sub_in_unc': stats['sub_in_unc'],
        'sub_out_unc': stats['sub_out_unc'],
        'ins_in_unc': stats['ins_in_unc'],
        'ins_out_unc': stats['ins_out_unc'],
        'del_in_unc': stats['del_in_unc'],
        'del_out_unc': stats['del_out_unc'],
        'unc_count': unc_count,
        'unc_correct_use': unc_correct_use,
        'has_unc': int(has_unc),
        'unc_chars': unc_chars,
        'non_unc_chars': non_unc_chars,
        'unc_chars_adjusted': unc_chars_adjusted,
        'non_unc_chars_adjusted': non_unc_chars_adjusted,
        'unc_error_rate': unc_error_rate,
        'non_unc_error_rate': non_unc_error_rate,
        'unc_gap': unc_gap,
        'has_format_error': int(has_format_error),
        'pred_len': len(clean_pred),
        'gt_len': len(ground_truth),
        'length_diff_gt-pre': len(ground_truth) - len(clean_pred),
        'words_in_unc_correct': word_metrics.get('words_in_unc_correct', 0),
        'words_in_unc_error': word_metrics.get('words_in_unc_error', 0),
        'total_words_in_unc': word_metrics.get('total_words_in_unc', 0),
        'words_out_unc_correct': word_metrics.get('words_out_unc_correct', 0),
        'words_out_unc_error': word_metrics.get('words_out_unc_error', 0),
        'total_words_out_unc': word_metrics.get('total_words_out_unc', 0),
        'word_unc_precision': word_unc_precision,
        'word_unc_recall': word_unc_recall,
        'word_f1': word_unc_f1,
        'word_accuracy': word_accuracy,
        'word_unc_gap': word_unc_gap,
        'word_error_rate': word_unc_error_rate,
        'word_non_unc_error_rate': word_non_unc_error_rate,
    }

    return result


if __name__ == "__main__":
    # Test cases
    test_cases = [
        ("Hello world", "Hello world", "Perfect match"),
        ("Hello world", "Hello world", "Perfect match"),
        ("Hello world", "Helo world", "Single character deletion"),
        ("Hello world", "Hello  world", "Extra space"),
        ("The quick brown fox", "The quik brown fox", "Misspelling"),
        ("<C>Hello</C> world", "Hello world", "UNC with correct content"),
        ("Hello <C>world</C>", "Hello world", "UNC at end"),
        ("<C>The quick</C> brown fox", "The quick brown fox", "UNC at beginning"),
        ("The <C>quick brown</C> fox", "The quick brown fox", "UNC in middle"),
        ("Hello", "Hello world", "Missing words"),
        ("Hello world extra", "Hello world", "Extra word"),
        ("<C>Test</C>", "Test", "Only UNC"),
        ("<C>Wrong</C> world", "Hello world", "UNC with wrong content"),
        ("", "Hello world", "Empty prediction"),
        ("Hello world", "", "Empty ground truth"),
        ("<C>The <C>nested</C> test</C>", "The nested test", "Nested UNC (format error)"),
        ("The quick brown fox jumps", "A quick brown fox jumped", "Multiple word differences"),
        ("<C>quick</C> brown fox", "quick brown fox", "UNC on correct word"),
        ("The <C>qick bron</C> fox", "The quick brown fox", "UNC on errors"),
        ("<C></C>Hello world", "Hello world", "Empty UNC tags"),
        ("Hello<C> </C>world", "Hello world", "UNC on space"),
    ]

    print("="*80)
    print("OCR Reward Word-Level Test Cases")
    print("="*80)

    for i, (pred, gt, description) in enumerate(test_cases, 1):
        print(f"\nTest {i}: {description}")
        print(f"Prediction: '{pred}'")
        print(f"Ground Truth: '{gt}'")

        try:
            result = compute_score(
                data_source="test",
                solution_str=pred,
                ground_truth=gt,
                extra_info={}
            )

            print(f"Score: {result['score']:.4f}")
            print(f"Word Accuracy: {result['word_accuracy']:.4f}")
            print(f"Word Precision: {result['word_unc_precision']:.4f}")
            print(f"Word Recall: {result['word_unc_recall']:.4f}")
            if result['has_unc']:
                print(f"UNC F1: {result['unc_f1']:.4f}")
                print(f"UNC Gap: {result['unc_gap']:.4f}")
                print(f"Word UNC Gap: {result['word_unc_gap']:.4f}")
            print(f"Has Format Error: {bool(result['has_format_error'])}")

        except Exception as e:
            print(f"Error: {e}")

        print("-"*40)