#!/usr/bin/env python3

from tqdm import tqdm
from typing import Dict, Any, List, Tuple
from rapidfuzz.distance import Levenshtein as Lev
import re

UNC_START = "<C>"
UNC_END = "</C>"

def normalize_text(text: str) -> str:
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = text.lower()
    text = ' '.join(text.split())
    return text

def normalize_whitespace(text: str) -> Tuple[str, List[int]]:
    normalized = []
    position_map = []

    i = 0
    while i < len(text):
        if text[i].isspace():
            start_pos = i
            while i < len(text) and text[i].isspace():
                i += 1
            normalized.append(' ')
            position_map.append(start_pos)
        else:
            normalized.append(text[i])
            position_map.append(i)
            i += 1

    return ''.join(normalized), position_map

def map_position_to_original(norm_pos: int, position_map: List[int], original_text: str) -> int:
    if norm_pos >= len(position_map):
        return len(original_text)
    return position_map[norm_pos]

def expand_regions_to_words(regions: List[Tuple[int, int]], text: str) -> List[Tuple[int, int]]:
    error_positions = set()
    for start, end in regions:
        for pos in range(start, end):
            error_positions.add(pos)

    word_error_positions = set()
    for pos in error_positions:
        if pos < len(text):
            word_start, word_end = find_word_boundaries(text, pos)
            for i in range(word_start, word_end):
                word_error_positions.add(i)

    if not word_error_positions:
        return []

    sorted_positions = sorted(word_error_positions)
    word_regions = []
    start = sorted_positions[0]
    end = start

    for pos in sorted_positions[1:]:
        if pos == end + 1:
            end = pos
        else:
            word_regions.append((start, end + 1))
            start = pos
            end = pos

    word_regions.append((start, end + 1))

    return word_regions

def is_punctuation_only(text: str) -> bool:
    text_no_space = ''.join(text.split())
    if not text_no_space:
        return True

    for char in text_no_space:
        if char.isalnum():
            return False

    return True

def expand_position_for_whitespace(pos: int, text: str) -> Tuple[int, int]:
    if pos >= len(text) or not text[pos].isspace():
        return pos, pos + 1

    start = pos
    while start > 0 and text[start - 1].isspace():
        start -= 1

    end = pos
    while end < len(text) and text[end].isspace():
        end += 1

    return start, end

def find_word_boundaries(text: str, pos: int) -> Tuple[int, int]:
    start = pos
    while start > 0 and not text[start - 1].isspace():
        start -= 1

    end = pos
    while end < len(text) and not text[end].isspace():
        end += 1

    return start, end

def mark_gt_errors(ocr_output: str, ground_truth: str,
                   verbose: bool = False, marking_level: str = "char") -> str:
    ocr_norm, ocr_map = normalize_whitespace(ocr_output)
    gt_norm, gt_map = normalize_whitespace(ground_truth)

    ops = Lev.editops(ocr_norm, gt_norm)

    if not ops:
        if verbose:
            print(f"OCR: {ocr_output}")
            print(f"GT:  {ground_truth}")
            print("No errors found")
        return ground_truth

    gt_error_positions = set()
    delete_word_positions = set()

    for op in ops:
        if op.tag == "replace":
            orig_pos = map_position_to_original(op.dest_pos, gt_map, ground_truth)

            if orig_pos < len(ground_truth) and ground_truth[orig_pos].isspace():
                start, end = expand_position_for_whitespace(orig_pos, ground_truth)
                for pos in range(start, end):
                    gt_error_positions.add(pos)
            else:
                gt_error_positions.add(orig_pos)

        elif op.tag == "insert":
            orig_pos = map_position_to_original(op.dest_pos, gt_map, ground_truth)
            gt_error_positions.add(orig_pos)

        elif op.tag == "delete":
            orig_pos = map_position_to_original(op.dest_pos, gt_map, ground_truth)
            if orig_pos < len(ground_truth):
                delete_word_positions.add(orig_pos)
            elif orig_pos > 0:
                delete_word_positions.add(orig_pos - 1)

    for pos in delete_word_positions:
        if pos < len(ground_truth):
            word_start, word_end = find_word_boundaries(ground_truth, pos)
            for i in range(word_start, word_end):
                gt_error_positions.add(i)

    if not gt_error_positions:
        return ground_truth

    sorted_positions = sorted(gt_error_positions)
    regions = []
    start = sorted_positions[0]
    end = start

    for pos in sorted_positions[1:]:
        if pos == end + 1:
            end = pos
        else:
            regions.append((start, end + 1))
            start = pos
            end = pos

    regions.append((start, end + 1))

    if marking_level == "word":
        regions = expand_regions_to_words(regions, ground_truth)

    filtered_regions = []
    for start, end in regions:
        region_text = ground_truth[start:end]
        if not is_punctuation_only(region_text):
            filtered_regions.append((start, end))

    regions = filtered_regions

    result = []
    last_pos = 0

    for start, end in regions:
        if start > last_pos:
            result.append(ground_truth[last_pos:start])

        result.append(UNC_START)
        result.append(ground_truth[start:end])
        result.append(UNC_END)

        last_pos = end

    if last_pos < len(ground_truth):
        result.append(ground_truth[last_pos:])

    marked_text = ''.join(result)

    pattern = re.escape(UNC_END) + r'(\s+)' + re.escape(UNC_START)
    marked_text = re.sub(pattern, r'\1', marked_text)

    if verbose:
        print(f"OCR: {ocr_output}")
        print(f"GT:  {ground_truth}")
        print(f"GT Marked: {marked_text}")

    return marked_text

def mark_ocr_errors(ocr_output: str, ground_truth: str,
                    verbose: bool = False, marking_level: str = "char") -> str:
    ocr_norm, ocr_map = normalize_whitespace(ocr_output)
    gt_norm, gt_map = normalize_whitespace(ground_truth)

    ops = Lev.editops(ocr_norm, gt_norm)

    if not ops:
        if verbose:
            print(f"OCR: {ocr_output}")
            print(f"GT:  {ground_truth}")
            print("No errors found")
        return ocr_output

    insert_positions = set()
    other_positions = set()

    for op in ops:
        if op.tag == "replace":
            orig_pos = map_position_to_original(op.src_pos, ocr_map, ocr_output)

            if orig_pos < len(ocr_output) and ocr_output[orig_pos].isspace():
                start, end = expand_position_for_whitespace(orig_pos, ocr_output)
                for pos in range(start, end):
                    other_positions.add(pos)
            else:
                other_positions.add(orig_pos)

        elif op.tag == "insert":
            orig_pos = map_position_to_original(op.src_pos, ocr_map, ocr_output)
            insert_positions.add(orig_pos)

        elif op.tag == "delete":
            orig_pos = map_position_to_original(op.src_pos, ocr_map, ocr_output)

            if orig_pos < len(ocr_output) and ocr_output[orig_pos].isspace():
                start, end = expand_position_for_whitespace(orig_pos, ocr_output)
                for pos in range(start, end):
                    other_positions.add(pos)
            else:
                other_positions.add(orig_pos)

    word_positions = set()
    for pos in insert_positions:
        if pos <= len(ocr_output):
            if pos == 0:
                word_pos = 0
            elif pos >= len(ocr_output):
                word_pos = len(ocr_output) - 1
            else:
                word_pos = pos if pos < len(ocr_output) else pos - 1
            word_start, word_end = find_word_boundaries(ocr_output, word_pos)
            for i in range(word_start, word_end):
                word_positions.add(i)

    all_error_positions = other_positions | word_positions

    if not all_error_positions:
        return ocr_output

    sorted_positions = sorted(all_error_positions)
    regions = []
    start = sorted_positions[0]
    end = start

    for pos in sorted_positions[1:]:
        if pos == end + 1:
            end = pos
        else:
            regions.append((start, end + 1))
            start = pos
            end = pos

    regions.append((start, end + 1))

    if marking_level == "word":
        regions = expand_regions_to_words(regions, ocr_output)

    filtered_regions = []
    for start, end in regions:
        region_text = ocr_output[start:end]
        if not is_punctuation_only(region_text):
            filtered_regions.append((start, end))

    regions = filtered_regions

    result = []
    last_pos = 0

    for start, end in regions:
        if start > last_pos:
            result.append(ocr_output[last_pos:start])

        result.append(UNC_START)
        result.append(ocr_output[start:end])
        result.append(UNC_END)

        last_pos = end

    if last_pos < len(ocr_output):
        result.append(ocr_output[last_pos:])

    marked_text = ''.join(result)

    pattern = re.escape(UNC_END) + r'(\s+)' + re.escape(UNC_START)
    marked_text = re.sub(pattern, r'\1', marked_text)

    return marked_text

def test_word_level_marking():
    print("="*60)
    print("Word-Level Marking Test Cases")
    print("="*60)

    print("\nTest 1: Character vs Word level comparison")
    print("-"*40)
    ocr1 = "hello wlrld 123"
    gt1 = "hello world"
    print("OCR: ", ocr1)
    print("GT:  ", gt1)
    print("Character level:")
    result1_char = mark_ocr_errors(ocr1, gt1, verbose=True, marking_level="char")
    print(result1_char)
    print("Word level:")
    result1_word = mark_ocr_errors(ocr1, gt1, verbose=True, marking_level="word")
    print(result1_word)

def process_text_files(ocr_file_path: str, gt_file_path: str,
                       output_file_path: str, mark_target: str = "ocr",
                       marking_level: str = "char", encoding: str = "utf-8",
                       silent: bool = False) -> None:
    try:
        with open(ocr_file_path, 'r', encoding=encoding) as f:
            ocr_content = f.read()

        with open(gt_file_path, 'r', encoding=encoding) as f:
            gt_content = f.read()

        if mark_target.lower() == "ocr":
            marked_result = mark_ocr_errors(ocr_content, gt_content, verbose=False, marking_level=marking_level)
        elif mark_target.lower() == "gt":
            marked_result = mark_gt_errors(ocr_content, gt_content, verbose=False, marking_level=marking_level)
        else:
            raise ValueError("mark_target must be 'ocr' or 'gt'")

        if not marking_level in ["char", "word"]:
            raise ValueError("marking_level must be 'char' or 'word'")

        with open(output_file_path, 'w', encoding=encoding) as f:
            f.write(marked_result)

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except Exception as e:
        print(f"Error processing files: {e}")

def batch_process_files(file_pairs: list, output_dir: str = "output",
                       mark_target: str = "ocr", marking_level: str = "char",
                       encoding: str = "utf-8") -> None:
    import os

    os.makedirs(output_dir, exist_ok=True)

    for i, (ocr_file, gt_file, output_name) in enumerate(file_pairs, 1):
        print(f"\n{'='*60}")
        print(f"Processing pair {i}/{len(file_pairs)}: {output_name}")
        print(f"{'='*60}")

        output_path = os.path.join(output_dir, f"{output_name}_marked.txt")
        process_text_files(ocr_file, gt_file, output_path, mark_target, marking_level, encoding)

if __name__ == "__main__":
    test_word_level_marking()