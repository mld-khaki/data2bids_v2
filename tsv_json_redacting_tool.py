#!/usr/bin/env python3
# TSV_JSON_redacting_tool_optimized.py
# Author: Dr. Milad Khaki (Updated by ChatGPT)
# Date: 2025-02-24
# Description: Optimized script redacts names from TSV and JSON files using caching, combined Aho–Corasick search, and optimized I/O.
# Usage: python TSV_JSON_redacting_tool_optimized.py <excel_path> <folder_path> <backup_folder> <backup_folder_upd>
# License: MIT License

import os
import json
import pandas as pd
import shutil
import csv
import re
import time
import argparse
from functools import lru_cache
import ahocorasick  # Requires: pip install pyahocorasick

SEPARATORS = r"[ _\.,\|;\-]+"  # Extended regex pattern for names with separators

ignore_list = ["obscur", "please", "clean", "leans", "polyspik", "adjustin", "against", 
    "covering", "fluttering", "leaving", "technician", "LIAN+", "max 2", "max 3", "max 4",
    "max 5", "max 6", "max 7", "max 8", "max 9", "max 0", "max L", "Max L", "clear", 
    "polys", "piano", "todd's", "todds","quivering","ering","POLYSPIK","against","leaves",
    "Todds","Todd's","sparkling","Clear","unpleasant","leading","PLEASE","variant"," IAn",
    "maximum","Maximum","MAXIMUM", " max ", "LIAn"]

def load_names_from_excel(excel_path):
    """Load names from an Excel file and generate variations."""
    df = pd.read_excel(excel_path, usecols=["LastName", "FirstName"], dtype=str)
    df.dropna(subset=["LastName", "FirstName"], inplace=True)

    last_names = set(df["LastName"].str.strip().tolist())
    first_names = set(df["FirstName"].str.strip().tolist())

    full_names = set()
    reverse_full_names = set()

    for _, row in df.iterrows():
        first, last = row["FirstName"].strip(), row["LastName"].strip()
        full_names.add(f"{last}{first[0]}")
        full_names.add(f"{first} {last}")
        reverse_full_names.add(f"{last} {first}")

        for sep in ["","_", ",", ".", "|", ";", "-", "  ", ", "]:
            full_names.add(f"{last}{sep}{first[0]}")
            full_names.add(f"{first}{sep}{last}")
            reverse_full_names.add(f"{last}{sep}{first}")

    return last_names, first_names, full_names, reverse_full_names

def prompt_user_for_replacement(line, name, file):
    pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, ignore_list)) + r')\b', re.IGNORECASE)
    tmp_line = pattern.sub(" ", line)
    if not any(name.lower() in word for word in tmp_line.lower().split()):
        return False
        
    """Prompt user before replacing a name."""
    for qCtr in range(80):
        print("=", end="", sep="")
    print(f"\nFound match upd: {tmp_line.strip()}, in file = <{file}>")
    print(f"\nFound match: {line.strip()}")
    response = input(f"Replace '{name}' with '.X.'? (y or enter/n): ").strip().lower()
    return response in ["y", ""]

@lru_cache(maxsize=1024)
def get_compiled_pattern(name):
    """Cache compiled regex pattern for a name."""
    return re.compile(rf"\b{re.escape(name)}\b|{re.escape(name).replace(' ', SEPARATORS)}", re.IGNORECASE)

def replace_with_case_preserved(text, name):
    """Replace whole word occurrences while preserving case."""
    def replacement(match):
        return ".X." if match.group(0).istitle() else ".x."
    
    pattern = get_compiled_pattern(name)
    return pattern.sub(replacement, text)

def build_automaton(names):
    """Build an Aho–Corasick automaton for fast string matching."""
    A = ahocorasick.Automaton()
    for name in names:
        A.add_word(name.lower(), name)
    A.make_automaton()
    return A

def find_matches(text, automaton, last_names, first_names, full_names, reverse_full_names):
    """Find all unique name matches in text, prioritizing longer matches."""
    all_matches = []
    lower_text = text.lower()
    
    # Collect all matches with their positions
    for end_index, original in automaton.iter(lower_text):
        start_index = end_index - len(original) + 1
        all_matches.append((start_index, end_index, original))
    
    # Sort matches by position (start index ascending, end index descending to prioritize longer matches)
    all_matches.sort(key=lambda x: (x[0], -x[1]))
    
    # Filter out overlapping matches, prioritizing longer ones
    filtered_matches = []
    if all_matches:
        # Initialize with the first match
        current_match = all_matches[0]
        filtered_matches.append(current_match[2])
        
        for match in all_matches[1:]:
            # If this match starts after the current one ends, it's non-overlapping
            if match[0] > current_match[1]:
                current_match = match
                filtered_matches.append(match[2])
            # If this is a longer match at the same starting position, replace the current one
            elif match[0] == current_match[0] and match[1] > current_match[1]:
                filtered_matches.pop()  # Remove the previous shorter match
                filtered_matches.append(match[2])
                current_match = match
    
    # Prioritize full patterns (reverse_full_names and full_names) over individual names
    prioritized_matches = set()
    for match in filtered_matches:
        prioritized_matches.add(match)
    
    return prioritized_matches

def move_to_backup(original_path, input_folder, backup_folder_org):
    """Move the original file to a backup folder while maintaining the structure."""
    rel_path = os.path.relpath(original_path, input_folder)
    backup_path_org = os.path.join(backup_folder_org, rel_path)
    if os.path.exists(backup_path_org):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        backup_path_org = f"{backup_path_org}_{timestamp}"
    os.makedirs(os.path.dirname(backup_path_org), exist_ok=True)
    shutil.move(original_path, backup_path_org)
    return backup_path_org

def process_tsv(file_path, args, automaton, last_names, first_names, full_names, reverse_full_names):
    """Process and redact TSV files."""
    changed = False
    temp_file_path = file_path + ".tmp"

    with open(file_path, "r", encoding="utf-8") as infile, open(temp_file_path, "w", encoding="utf-8", newline='') as outfile:
        reader = csv.reader(infile, delimiter='\t')
        writer = csv.writer(outfile, delimiter='\t')

        for row in reader:
            new_row = []
            for cell in row:
                original_cell = cell
                matches = sorted(find_matches(cell, automaton, last_names, first_names, full_names, reverse_full_names), 
                                key=len, reverse=True)  # Sort matches by length, longest first
                for name in matches:
                    if prompt_user_for_replacement(cell, name, file_path):
                        cell = replace_with_case_preserved(cell, name)
                        break  # Stop after the first successful replacement
                new_row.append(cell)
                if cell != original_cell:
                    changed = True
            writer.writerow(new_row)
            changed |= new_row != row

    if changed:
        rel_path = os.path.relpath(file_path, args.input_folder)
        backup_path_upd = os.path.join(args.backup_folder_upd, rel_path)
        os.makedirs(os.path.dirname(backup_path_upd), exist_ok=True)
        shutil.copyfile(temp_file_path, backup_path_upd)

        backup_path = move_to_backup(file_path, args.input_folder, args.backup_folder_org)

        os.replace(temp_file_path, file_path)
        print(f" - Redacted TSV file. Backup moved to {backup_path}, org moved to {backup_path_upd}")
    else:
        os.remove(temp_file_path)
        print(" - No changes needed in TSV file.")

    return changed

def process_json(file_path, args, automaton, last_names, first_names, full_names, reverse_full_names):
    """Process and redact JSON files."""
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(" - Error: Invalid JSON file.")
            return False

    def redact(obj):
        changed = False
        if isinstance(obj, dict):
            new_dict = {}
            for k, v in obj.items():
                new_v, was_changed = redact(v) if isinstance(v, (dict, list, str)) else (v, False)
                new_dict[k] = new_v
                changed |= was_changed  # Ensure change is tracked
            return new_dict, changed
        elif isinstance(obj, list):
            new_list = []
            for v in obj:
                new_v, was_changed = redact(v) if isinstance(v, (dict, list, str)) else (v, False)
                new_list.append(new_v)
                changed |= was_changed
            return new_list, changed
        elif isinstance(obj, str):
            # original_obj = obj
            matches = find_matches(obj, automaton, last_names, first_names, full_names, reverse_full_names)
            for name in matches:
                if prompt_user_for_replacement(obj, name, file_path):
                    obj = replace_with_case_preserved(obj, name)
                    changed = True
            return obj, changed
        return obj, changed

    modified_data, changed = redact(data)

    if changed:
        rel_path = os.path.relpath(file_path, args.input_folder)
        backup_path_upd = os.path.join(args.backup_folder_upd, rel_path)
        os.makedirs(os.path.dirname(backup_path_upd), exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(modified_data, f, indent=4, ensure_ascii=False)

        shutil.copyfile(file_path, backup_path_upd)

        backup_path_org = move_to_backup(file_path, args.input_folder, args.backup_folder_org)

        print(f" - Redacted JSON file. Backup moved to {backup_path_org}, org moved to {backup_path_upd}")        
    else:
        print(" - No changes needed in JSON file.")

    return changed

from functools import lru_cache

def redact_names(text, names_to_redact, preserve_case=True, replacement_text=".X."):
    """
    Redacts names from text, prioritizing longer matches.
    
    Args:
        text (str): The text to redact names from
        names_to_redact (list): List of names to redact
        preserve_case (bool): If True, preserves case of replacement (.X. for title case, .x. for lowercase)
        replacement_text (str): Text to replace names with (default: ".X.")
        
    Returns:
        str: The redacted text
    """
    # Build Aho-Corasick automaton for efficient string matching
    automaton = ahocorasick.Automaton()
    for name in names_to_redact:
        if name and isinstance(name, str):
            automaton.add_word(name.lower(), name)
    automaton.make_automaton()
    
    # Find matches
    all_matches = []
    lower_text = text.lower()
    
    # Collect all matches with their positions
    for end_index, original in automaton.iter(lower_text):
        start_index = end_index - len(original) + 1
        all_matches.append((start_index, end_index, original))
    
    # Sort matches by position and length (prioritize longer matches)
    all_matches.sort(key=lambda x: (x[0], -x[1]))
    
    # Filter out overlapping matches, keeping longer ones
    filtered_matches = []
    if all_matches:
        current_match = all_matches[0]
        filtered_matches.append(current_match)
        
        for match in all_matches[1:]:
            # If this match starts after the current one ends, it's non-overlapping
            if match[0] > current_match[1]:
                current_match = match
                filtered_matches.append(match)
            # If this is a longer match at the same starting position, replace the current one
            elif match[0] == current_match[0] and match[1] > current_match[1]:
                filtered_matches.pop()  # Remove the previous shorter match
                filtered_matches.append(match)
                current_match = match
    
    # No matches found, return original text
    if not filtered_matches:
        return text
    
    # Apply replacements from end to start to avoid offset issues
    result = text
    for start_idx, end_idx, original in sorted(filtered_matches, key=lambda x: -x[0]):
        match_in_original_text = result[start_idx:end_idx+1]
        if preserve_case:
            replacement = replacement_text if match_in_original_text[0].isupper() else replacement_text.lower()
        else:
            replacement = replacement_text
        result = result[:start_idx] + replacement + result[end_idx+1:]
    
    return result

# Example usage:
# names = ["John Smith", "Jane Doe", "Smith", "J Smith"]
# text = "John Smith and Jane Doe were discussing with Smith and J Smith."
# redacted = redact_names(text, names)
# print(redacted)

def search_and_process_files(args, automaton, last_names, first_names, full_names, reverse_full_names):
    """Search for files and process them."""
    file_extensions = {".tsv": process_tsv, ".json": process_json}
    total_changed = 0

    for root, _, files in os.walk(args.input_folder):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()

            if ext in file_extensions:
                print(f"Processing {file_path}...")
                changed = file_extensions[ext](file_path, args, automaton, last_names, first_names, full_names, reverse_full_names)
                if changed:
                    total_changed += 1

    print(f"Total files modified: {total_changed}")

def main():
    """Main function to get user input and run the script."""
    parser = argparse.ArgumentParser(description="Redact names from TSV and JSON files.")
    parser.add_argument("excel_path", nargs="?", help="Path to the Excel file")
    parser.add_argument("input_folder", nargs="?", help="Folder containing TSV/JSON files")
    parser.add_argument("backup_folder_org", nargs="?", help="Folder to store original files")
    parser.add_argument("backup_folder_upd", nargs="?", help="Folder2 to store newly generated files")

    args = parser.parse_args()

    # Prompt user if any argument is missing
    if not args.excel_path:
        args.excel_path = input("Enter path to Excel file (default: e:/iEEG_Demographics.xlsx): ") or "e:/iEEG_Demographics.xlsx"
    if not args.input_folder:
        args.input_folder = input("Enter input folder path (default: c:/tmp/all_tsv/): ") or "c:/tmp/all_tsv/"
    if not args.backup_folder_org:
        args.backup_folder_org = input("Enter backup folder for original files' path (default: c:/tmp/backup/org/): ") or "c:/tmp/backup/org/"
    if not args.backup_folder_upd:
        args.backup_folder_upd = input("Enter backup folder for newly gen files' path (default: c:/tmp/backup2/upd): ") or "c:/tmp/backup2/upd/"

    start_time = time.time()

    print(f"Loading names from {args.excel_path}...")
    last_names, first_names, full_names, reverse_full_names = load_names_from_excel(args.excel_path)

    automaton = build_automaton(set().union(last_names, first_names, full_names, reverse_full_names))

    print(f"Scanning {args.input_folder}...")
    search_and_process_files(args, automaton, last_names, first_names, full_names, reverse_full_names)

    print(f"Total execution time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()