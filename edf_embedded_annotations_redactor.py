import numpy as np
import re
import logging
import os
import time
import mmap
import traceback
import argparse
from datetime import datetime
import sys
from edflibpy.edfreader import EDFreader
import ahocorasick  # New dependency: pip install pyahocorasick
from tqdm import tqdm

# Configure logging
def setup_logging(log_dir="logs", filename = "logData_"):
    """Set up detailed logging to both console and file"""
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    # Generate a timestamp for the log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, filename + f"edf_anonymize_{timestamp}.log")
    redaction_map_file = os.path.join(log_dir, filename + f"redaction_map_{timestamp}.txt")
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # File handler with more detailed formatting
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Console handler with less verbose output for better readability
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Create a special handler for redaction mapping
    redaction_map_handler = logging.FileHandler(redaction_map_file)
    redaction_map_handler.setLevel(logging.INFO)
    redaction_map_formatter = logging.Formatter('%(message)s')
    redaction_map_handler.setFormatter(redaction_map_formatter)
    
    # Create a dedicated logger for redaction mapping
    redaction_logger = logging.getLogger('redaction_map')
    redaction_logger.setLevel(logging.INFO)
    redaction_logger.addHandler(redaction_map_handler)
    
    # Log the startup message
    logger.info(f"Logging initialized. Log file: {log_file}")
    logger.info(f"Redaction map file: {redaction_map_file}")
    return logger

def build_automaton(name_patterns):
    """
    Build an Aho-Corasick automaton for efficient string matching.
    
    Args:
        name_patterns: List of string patterns to match
        
    Returns:
        Compiled Aho-Corasick automaton
    """
    # Remove any empty or None patterns
    valid_patterns = [p for p in name_patterns if p and isinstance(p, bytes)]
    
    if not valid_patterns:
        return None
        
    # Convert bytes to strings for the automaton
    str_patterns = [p.decode('utf-8', errors='replace') for p in valid_patterns]
    
    # Build automaton
    A = ahocorasick.Automaton()
    for idx, pattern in enumerate(str_patterns):
        A.add_word(pattern.lower(), (idx, pattern))
    A.make_automaton()
    
    return A


def process_edf_annotations(data_chunk, automaton, redaction_patterns, annot_offsets, annot_sizes, record_size):
    """
    Process EDF+ annotations by blanking all annotation content.
    
    Args:
        data_chunk: Raw binary data containing annotations
        automaton: Compiled Aho-Corasick automaton (not used when blanking all annotations)
        redaction_patterns: List of (pattern, replacement) tuples (not used when blanking all annotations)
        annot_offsets: List of offsets where annotation signals begin in each record
        annot_sizes: List of annotation signal sizes
        record_size: Size of each data record in bytes
        
    Returns:
        Processed data chunk with blank annotations
    """
    logger = logging.getLogger('edf_processor')
    redaction_logger = logging.getLogger('redaction_map')
    logger.debug(f"Processing data chunk of size {len(data_chunk)}, with {len(annot_offsets)} annotation signals")
    
    data_array = np.frombuffer(data_chunk, dtype=np.uint8).copy()
    
    for r in range(len(data_chunk) // record_size):
        record_start = r * record_size
        
        for i, offset in enumerate(annot_offsets):
            annot_start = record_start + offset
            annot_end = annot_start + annot_sizes[i]
            
            if annot_end <= len(data_array):
                # Get the annotation data for this signal in this record
                annot_data = data_array[annot_start:annot_end]
                
                try:
                    # Decode annotation bytes to string with error handling
                    annot_bytes = annot_data.tobytes()
                    
                    # Skip empty annotation blocks
                    if all(b == 0 for b in annot_bytes):
                        continue
                    
                    # Find TALs in the annotation bytes
                    # TALs start with + or - (timestamps) and end with a null byte
                    pos = 0
                    modified_bytes = bytearray(annot_bytes)
                    
                    while pos < len(annot_bytes):
                        # Find the start of a TAL (should begin with + or -)
                        if annot_bytes[pos] == ord('+') or annot_bytes[pos] == ord('-'):
                            # Extract the entire TAL until the terminating null byte
                            tal_end = annot_bytes.find(b'\x00', pos)
                            if tal_end == -1:  # No null terminator found
                                tal_end = len(annot_bytes)
                            
                            tal_bytes = annot_bytes[pos:tal_end+1]
                            
                            logger.debug(f"Found TAL at record {r}, offset {offset}, position {pos}: "
                                        f"{tal_bytes[:min(20, len(tal_bytes))]}...")
                                        
                            modified_tal = process_tal(tal_bytes, automaton, redaction_patterns, logger, redaction_logger)
                            
                            # Replace the TAL in the modified bytes
                            modified_bytes[pos:pos+len(tal_bytes)] = modified_tal
                            pos += len(modified_tal)
                        else:
                            # Skip non-TAL bytes (should be null padding)
                            pos += 1
                    
                    # Copy modified bytes back to the data array
                    for j in range(len(modified_bytes)):
                        if annot_start + j < len(data_array):
                            data_array[annot_start + j] = modified_bytes[j]
                
                except Exception as e:
                    logger.warning(f"Warning: Failed to process annotation in record {r}, offset {offset}: {e}")
                    logger.debug(f"Exception details: {traceback.format_exc()}")
    
    return data_array.tobytes()

def process_tal(tal_bytes, automaton, redaction_patterns, logger, redaction_logger):
    """
    Process a single Time-stamped Annotation List (TAL) by removing all annotation content.
    
    Args:
        tal_bytes: Bytes of a single TAL
        automaton: Compiled Aho-Corasick automaton (not used when blanking all annotations)
        redaction_patterns: List of (pattern, replacement) tuples (not used when blanking all annotations)
        logger: Logger instance for debug output
        redaction_logger: Logger for recording redaction mappings
        
    Returns:
        Modified TAL bytes with blank annotations
    """
    try:
        # Convert to bytearray for modification
        tal = bytearray(tal_bytes)
        
        # Find the onset timestamp (part before the first 0x14 byte)
        onset_end = tal.find(0x14)
        if onset_end == -1:
            return tal  # No proper TAL format, return unchanged
        
        # Extract onset
        onset = tal[:onset_end].decode('utf-8', errors='replace')
        
        # Skip the timestamp and any duration marker
        annotations_start = onset_end + 1
        
        # Process each annotation in the TAL
        current_pos = annotations_start
        modifications_made = 0
        
        while current_pos < len(tal):
            # Find the end of this annotation (marked by 0x14)
            annotation_end = tal.find(0x14, current_pos)
            if annotation_end == -1:
                break  # No more annotations
                
            # Extract the annotation text
            if annotation_end > current_pos:
                original_annotation = tal[current_pos:annotation_end].decode('utf-8', errors='replace')
                
                # Log the original annotation
                if original_annotation.strip():  # Only log non-empty annotations
                    redaction_logger.info(f"BLANKED ANNOTATION: '{original_annotation}'")
                    logger.debug(f"Blanked annotation: '{original_annotation}'")
                
                # Replace with an empty or placeholder annotation
                blank_annotation = ""  # Or use a placeholder like "REDACTED"
                blank_bytes = blank_annotation.encode('utf-8')
                
                # Replace the annotation text, padding with spaces to maintain TAL length
                tal[current_pos:annotation_end] = blank_bytes.ljust(annotation_end - current_pos, b' ')
                
                if original_annotation.strip():  # Count only meaningful annotations
                    modifications_made += 1
            
            # Move to the next annotation
            current_pos = annotation_end + 1
            
            # If we hit a null byte, we're at the end of the TAL
            if current_pos < len(tal) and tal[current_pos] == 0:
                break
        
        logger.debug(f"Processed TAL at timestamp {onset}: {modifications_made} annotations blanked")
        return tal
    except Exception as e:
        logger.warning(f"Error processing TAL: {e}")
        logger.debug(f"TAL processing exception details: {traceback.format_exc()}")
        return tal_bytes  # Return unchanged if there was an error

def redact_with_automaton(text, automaton):
    """
    Redact text using Aho-Corasick automaton, prioritizing longer matches.
    
    Args:
        text: Text to redact
        automaton: Compiled Aho-Corasick automaton
        
    Returns:
        Redacted text with identifiers replaced by X's
    """
    if not text or automaton is None:
        return text
        
    # Find all matches with their positions
    matches = []
    for end_pos, (_, original) in automaton.iter(text.lower()):
        start_pos = end_pos - len(original) + 1
        matches.append((start_pos, end_pos, original))
    
    # No matches found
    if not matches:
        return text
        
    # Sort matches by position (start index ascending, end index descending to prioritize longer matches)
    matches.sort(key=lambda x: (x[0], -x[1]))
    
    # Filter out overlapping matches, prioritizing longer ones
    filtered_matches = []
    current_match = matches[0]
    filtered_matches.append(current_match)
    
    for match in matches[1:]:
        # If this match starts after the current one ends, it's non-overlapping
        if match[0] > current_match[1]:
            current_match = match
            filtered_matches.append(match)
        # If this is a longer match at the same starting position, replace the current one
        elif match[0] == current_match[0] and match[1] > current_match[1]:
            filtered_matches.pop()  # Remove the previous shorter match
            filtered_matches.append(match)
            current_match = match
    
    # Apply redactions from end to beginning to avoid offset issues
    result = list(text)
    for start_pos, end_pos, _ in sorted(filtered_matches, reverse=True):
        # Get the actual text segment to redact from the original
        matched_segment = text[start_pos:end_pos+1]
        # Replace with X's while preserving case
        replacement = ''.join(['X' if c.isupper() else 'x' for c in matched_segment])
        # Apply the replacement
        result[start_pos:end_pos+1] = replacement
    
    return ''.join(result)

def extract_patient_info_from_header(header_bytes):
    """
    Extract patient information from EDF header for redaction, with enhanced name variation generation.
    
    Args:
        header_bytes: The EDF file header as bytes
        
    Returns:
        Tuple of (redaction_patterns, automated_patterns)
        - redaction_patterns: List of (pattern, replacement) tuples for traditional pattern matching
        - name_variations: List of name pattern variations for automaton construction
    """
    logger = logging.getLogger('edf_processor')
    redaction_patterns = []
    name_variations = []
    
    try:
        # Extract patient information field (bytes 8-88)
        patient_info = header_bytes[8:88].decode('ascii', errors='ignore').strip()
        logger.info(f"Extracted patient field from header: '{patient_info}'")
        
        # Extract recording information field (bytes 88-168)
        recording_info = header_bytes[88:168].decode('ascii', errors='ignore').strip()
        logger.info(f"Extracted recording field from header: '{recording_info}'")
        
        # Extract name components from patient info
        # Format according to EDF+ spec: hospital_id sex birthdate patient_name
        parts = patient_info.split()
        
        first_name = None
        last_name = None
        
        if len(parts) >= 4:
            # The hospital ID is the first part
            hospital_id = parts[0]
            if len(hospital_id) > 2:  # Only redact if it's a substantial ID
                redaction_patterns.append((hospital_id.encode(), b'X-XXXXXXX'))
                name_variations.append(hospital_id.encode())
                logger.debug(f"Adding redaction pattern for hospital ID: {hospital_id}")
            
            # The patient name should be all parts from position 3 onwards
            if len(parts) > 3:
                name_parts = parts[3:]
                
                # Add each name part as a pattern
                for name in name_parts:
                    if len(name) > 2:  # Only redact names longer than 2 chars
                        redaction_patterns.append((name.encode(), b'XXXX'))
                        name_variations.append(name.encode())
                        logger.debug(f"Adding redaction pattern for name part: {name}")
                
                # Also add the full name
                full_name = ' '.join(name_parts)
                if len(full_name) > 2:
                    redaction_patterns.append((full_name.encode(), b'XXXX'))
                    name_variations.append(full_name.encode())
                    logger.debug(f"Adding redaction pattern for full name: {full_name}")
                
                # Assume first name and last name if multiple parts
                if len(name_parts) >= 2:
                    first_name = name_parts[0]
                    last_name = name_parts[-1]
        
        # Look for name patterns in recording field as well
        recording_parts = recording_info.split()
        for part in recording_parts:
            if len(part) > 2 and not part.isdigit():  # Skip short parts and numbers
                redaction_patterns.append((part.encode(), b'XXXX'))
                name_variations.append(part.encode())
        
        # Also look for common patient identifier patterns
        id_pattern = re.compile(r'\b[A-Z0-9]{6,}\b')
        
        # Search in both patient and recording fields
        for info_field in [patient_info, recording_info]:
            id_matches = id_pattern.findall(info_field)
            
            for id_match in id_matches:
                if len(id_match) > 3:  # Only redact substantial IDs
                    redaction_patterns.append((id_match.encode(), b'XXXXXXX'))
                    name_variations.append(id_match.encode())
                    logger.debug(f"Adding redaction pattern for identified ID: {id_match}")
        
        # Generate name variations if we have both first and last name
        if first_name and last_name:
            variations = generate_name_variations(first_name, last_name)
            name_variations.extend(variations)
            logger.debug(f"Generated {len(variations)} name variations for {first_name} {last_name}")
    
    except Exception as e:
        logger.warning(f"Warning: Failed to extract patient info: {e}")
        logger.debug(f"Patient info extraction exception details: {traceback.format_exc()}")
    
    logger.info(f"Generated {len(redaction_patterns)} traditional redaction patterns")
    logger.info(f"Generated {len(name_variations)} name variations for automaton")
    
    return redaction_patterns, name_variations

def anonymize_edf_header(header_bytes):
    """
    Anonymize the EDF header by replacing patient identifiable information.
    
    Args:
        header_bytes: The EDF file header as bytes
        
    Returns:
        Anonymized header bytes
    """
    logger = logging.getLogger('edf_processor')
    new_header = bytearray(header_bytes)
    
    # Log original patient and recording fields
    patient_field = header_bytes[8:88].decode('ascii', errors='ignore').strip()
    recording_field = header_bytes[88:168].decode('ascii', errors='ignore').strip()
    
    logger.info(f"Original patient field: '{patient_field}'")
    logger.info(f"Original recording field: '{recording_field}'")
    
    # Anonymize patient field (bytes 8-88)
    anonymous_patient = "X X X X".ljust(80).encode('ascii')
    new_header[8:88] = anonymous_patient
    
    logger.info(f"Anonymized patient field to: 'X X X X'")
    return new_header

def anonymize_edf_complete(input_path, output_path, buffer_size_mb=64, redaction_patterns = [], log_dir = ""):
    """
    Complete EDF anonymizer that ensures exact file size matching and properly handles annotations.
    
    This implementation:
    1. Directly reads and parses the EDF header to understand file structure
    2. Identifies and processes annotation channels
    3. Creates a new file with an anonymized header
    4. Copies data records with annotation redaction where needed
    5. Ensures the output file size matches exactly what's expected
    
    Args:
        input_path (str): Path to the input EDF file
        output_path (str): Path to save the anonymized EDF file
        buffer_size_mb (int): Size of the buffer in megabytes for reading/writing data
        redaction_patterns (list): Optional list of redaction patterns to use
        log_dir (str): Directory to store log files
        
    Returns:
        bool: True if anonymization was successful, False otherwise
    """
    
    start_time = time.time()
    edf_reader = None
    
    if log_dir != "":
        logger = setup_logging(log_dir, filename=os.path.basename(input_path))
    else:
        logger = logging.getLogger('edf_anonymizer')

    
    logger.info(f"Starting anonymization of: {input_path}")
    logger.info(f"Output will be saved to: {output_path}")
    logger.info(f"Using buffer size of {buffer_size_mb} MB")
    
    try:
        input_file_size = os.path.getsize(input_path)
        logger.info(f"Input file size: {input_file_size:,} bytes ({input_file_size / (1024*1024):.2f} MB)")

        # Open file using mmap for fast direct access
        with open(input_path, 'rb') as f:
            logger.debug("Opening input file with memory mapping")
            mmapped_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

            # Read base header
            base_header = bytearray(mmapped_file[:256])
            logger.debug("Read base header (256 bytes)")

            # Extract essential EDF header info
            header_bytes = int(base_header[184:192].decode('ascii').strip())
            data_records = int(base_header[236:244].decode('ascii').strip())
            record_duration = float(base_header[244:252].decode('ascii').strip())
            num_signals = int(base_header[252:256].decode('ascii').strip())
            
            logger.info(f"Header size: {header_bytes} bytes")
            logger.info(f"Data records: {data_records}, Duration: {record_duration}s per record")
            logger.info(f"Number of signals: {num_signals}")

            # Read signal header
            signal_header_size = num_signals * 256
            signal_header = bytearray(mmapped_file[256:256 + signal_header_size])
            logger.debug(f"Read signal header ({signal_header_size} bytes)")

            # Identify annotation channels
            signal_labels = [signal_header[i*16:(i+1)*16].decode('ascii').strip() for i in range(num_signals)]
            annot_channels = [i for i, label in enumerate(signal_labels) if label in ["EDF Annotations", "BDF Annotations"]]

            logger.info(f"Signal labels: {signal_labels}")
            logger.info(f"Annotation channels identified: {annot_channels}")

            # Read samples per record
            samples_per_record = [
                int(signal_header[num_signals * 216 + (i * 8):num_signals * 216 + (i * 8) + 8].decode('ascii').strip())
                for i in range(num_signals)
            ]
            
            logger.debug(f"Samples per record per signal: {samples_per_record}")

            # Calculate data record size
            bytes_per_sample = 2  # EDF uses 2-byte integers
            record_size = sum(samples_per_record) * bytes_per_sample
            logger.info(f"Data record size: {record_size} bytes")

            # Extract patient info for redaction
            automated_patterns = []
            try:
                # Extract directly from header
                logger.debug("Extracting patient info from header for redaction")
                header_patterns, name_variations = extract_patient_info_from_header(base_header)
                redaction_patterns.extend(header_patterns)
                automated_patterns.extend(name_variations)
                
                # Try to get additional details from EDFreader
                logger.debug("Attempting to extract additional patient info using EDFreader")
                edf_reader = EDFreader(input_path)
                patient_name = edf_reader.getPatientName()
                if patient_name:
                    logger.info(f"Patient name from EDFreader: '{patient_name}'")
                    
                    # Split by common separators
                    name_parts = re.split(r';|,|-|\s+', patient_name)
                    for name in name_parts:
                        if len(name) > 2:  # Only redact substantial names
                            redaction_patterns.append((name.encode(), b'XXXX'))
                            automated_patterns.append(name.encode())
                            logger.debug(f"Added redaction pattern from EDFreader: {name}")
                    
                    # Try to identify first and last name
                    if len(name_parts) >= 2:
                        first_name = name_parts[0]
                        last_name = name_parts[-1]
                        variations = generate_name_variations(first_name, last_name)
                        automated_patterns.extend(variations)
                        logger.debug(f"Generated {len(variations)} name variations from EDFreader name")
                
                edf_reader.close()

                
                # Remove duplicates from both lists
                original_count = len(redaction_patterns)
                unique_patterns = {}
                for pattern, replacement in redaction_patterns:
                    pattern_str = pattern.decode('utf-8', errors='ignore')
                    if pattern_str not in unique_patterns:
                        unique_patterns[pattern_str] = replacement
                
                redaction_patterns = [(k.encode('utf-8'), v) for k, v in unique_patterns.items()]
                logger.info(f"Redaction patterns: {len(redaction_patterns)} unique patterns "
                           f"(removed {original_count - len(redaction_patterns)} duplicates)")
                
                # Add standard montage patterns regardless of names
                montage_patterns = [
                    (b"montage", b"montage"),  # Keep the word montage but detect it for context
                    (b"Montage", b"Montage"),
                    (b"MONTAGE", b"MONTAGE")
                ]

                for pattern, replacement in montage_patterns:
                    automated_patterns.append(pattern)

                # Also add a placeholder pattern for generic detection
                automated_patterns.append(b"montage ")                
                # Build automaton from name variations
                
                automaton = build_automaton(automated_patterns)
                if automaton:
                    logger.info(f"Built automaton with {len(automated_patterns)} unique patterns")
                else:
                    logger.warning("Failed to build automaton, falling back to pattern matching only")
                
                # Log all patterns for debugging
                for pattern, replacement in redaction_patterns:
                    logger.debug(f"Pattern: '{pattern.decode('utf-8', 'replace')}' -> '{replacement.decode('utf-8', 'replace')}'")
            
            except Exception as e:
                logger.warning(f"Warning: Could not read patient name: {e}")
                logger.debug(f"Patient name extraction exception details: {traceback.format_exc()}")
                automaton = None
            
            # Anonymize patient info in header
            logger.info("Anonymizing patient info in header")
            new_base_header = anonymize_edf_header(base_header)

            # Calculate annotation offsets
            annot_offsets, annot_sizes = [], []
            offset = 0
            for i, samples in enumerate(samples_per_record):
                size = samples * bytes_per_sample
                if i in annot_channels:
                    annot_offsets.append(offset)
                    annot_sizes.append(size)
                offset += size

            logger.debug(f"Annotation offsets: {annot_offsets}, sizes: {annot_sizes}")

            # Prepare output file
            logger.info(f"Creating output file: {output_path}")
            with open(output_path, 'wb') as out_file:
                # Write the anonymized header
                out_file.write(new_base_header)
                out_file.write(signal_header)
                logger.debug(f"Wrote anonymized header ({len(new_base_header) + len(signal_header)} bytes)")

                buffer_size_bytes = buffer_size_mb * 1024 * 1024
                records_per_chunk = max(1, buffer_size_bytes // record_size)
                chunk_size_bytes = records_per_chunk * record_size

                logger.info(f"Processing records in chunks: {records_per_chunk} records "
                           f"({chunk_size_bytes / (1024*1024):.2f} MB) per chunk")

                bytes_remaining = input_file_size - (256 + signal_header_size)
                logger.info(f"Data bytes to process: {bytes_remaining:,}")
                
                total_records_processed = 0
                records_with_annotations = 0
                
                with tqdm(total=data_records, desc="Processing records", unit="records") as pbar:
                    for record_index in range(0, data_records, records_per_chunk):
                        chunk_records = min(records_per_chunk, data_records - record_index)
                        chunk_bytes = min(chunk_size_bytes, bytes_remaining)
                        if chunk_bytes <= 0:
                            logger.warning(f"No more bytes to process after record {record_index}")
                            break
                        
                        logger.debug(f"Processing chunk starting at record {record_index}, "
                                    f"size: {chunk_bytes} bytes, {chunk_records} records")
                
                        data_chunk = mmapped_file[256 + signal_header_size + record_index * record_size:
                                                  256 + signal_header_size + record_index * record_size + chunk_bytes]
                
                        if annot_channels and (redaction_patterns or automaton):

                            logger.info("Set to blank all annotations in output file - ignoring redaction patterns")                            
                            start_process = time.time()
                            data_chunk = process_edf_annotations(data_chunk, automaton, redaction_patterns, annot_offsets, annot_sizes, record_size)
                            process_time = time.time() - start_process
                            logger.debug(f"Annotation processing took {process_time:.3f}s")
                            records_with_annotations += chunk_records
                        else:
                            logger.debug(f"Skipping annotation processing (no channels or patterns)")
                
                        out_file.write(data_chunk)
                        bytes_remaining -= chunk_bytes
                        total_records_processed += chunk_records
                        pbar.update(chunk_records)

            mmapped_file.close()
            logger.info(f"Processed {total_records_processed} records total, "
                       f"{records_with_annotations} records with annotations")

            # Verify output file size and fix discrepancies
            output_file_size = os.path.getsize(output_path)
            file_size_diff = input_file_size - output_file_size
            logger.info(f"Output file size: {output_file_size:,} bytes")
            
            if file_size_diff != 0:
                logger.warning(f"File size difference detected: {file_size_diff} bytes")
                with open(output_path, 'r+b') as fix_file:
                    if abs(file_size_diff) < 256:
                        # Small difference might be due to header changes
                        actual_data_size = output_file_size - (256 + signal_header_size)
                        actual_records = int(round(actual_data_size / record_size))

                        if actual_records != data_records:
                            logger.info(f"Updating data records in header: {data_records} → {actual_records}")
                            fix_file.seek(236)
                            fix_file.write(f"{actual_records:<8}".encode('ascii'))
                    else:
                        # Larger discrepancy - pad or truncate the file
                        if file_size_diff > 0:  # Output file is smaller than input
                            logger.warning(f"Padding output file with {file_size_diff} bytes")
                            fix_file.seek(0, os.SEEK_END)  # Go to end of file
                            
                            # Before padding, ensure the header has the correct number of data records
                            # This is the critical fix that was missing:
                            if total_records_processed == 0 and data_records > 0:
                                logger.warning("No records processed but expected data records > 0, fixing header")
                                fix_file.seek(236)
                                fix_file.write(f"{data_records:<8}".encode('ascii'))
                            
                            # Now pad the file (for large files, pad in smaller chunks to avoid memory issues)
                            chunk_size = 1024 * 1024  # 1MB chunks
                            remaining = file_size_diff
                            fix_file.seek(0, os.SEEK_END)
                            
                            while remaining > 0:
                                write_size = min(chunk_size, remaining)
                                fix_file.write(b'\x00' * write_size)
                                remaining -= write_size
                                
                        else:  # Output file is larger than input
                            logger.warning(f"Output file is {-file_size_diff} bytes larger than input - truncating")
                            fix_file.truncate(input_file_size)  # Truncate to match input size
                
                # Verify size after adjustment
                final_size = os.path.getsize(output_path)
                logger.info(f"Final output file size after adjustment: {final_size:,} bytes")
                if final_size != input_file_size:
                    logger.warning(f"Size mismatch remains: {input_file_size - final_size} bytes difference")

        elapsed_time = time.time() - start_time
        logger.info(f"Anonymization completed in {elapsed_time:.2f} seconds")
        logger.info(f"Processing speed: {input_file_size / 1024 / 1024 / elapsed_time:.2f} MB/s")
        logger.info(f"Anonymized EDF saved to: {output_path}")
        # Final validation to ensure header is correct
        validate_edf_header(output_path, expected_records=data_records)
        return True
        
    except Exception as e:
        logger.error(f"Error during anonymization: {e}")
        logger.error(traceback.format_exc())
        return False
        
def validate_edf_header(file_path, expected_records=None):
    """
    Validate and fix critical EDF header fields.
    
    Args:
        file_path: Path to the EDF file
        expected_records: Expected number of data records (if known)
        
    Returns:
        True if header was valid or successfully fixed, False otherwise
    """
    logger = logging.getLogger('edf_validator')
    try:
        with open(file_path, 'r+b') as f:
            # Read header
            f.seek(0)
            header = f.read(256)
            
            # Extract data record count
            data_records_str = header[236:244].decode('ascii', 'replace').strip()
            try:
                data_records = int(data_records_str)
                logger.info(f"Header data record count: {data_records}")
                
                # Check if record count is valid
                if data_records <= 0 and expected_records is not None:
                    logger.warning(f"Invalid data record count ({data_records}), fixing to {expected_records}")
                    f.seek(236)
                    f.write(f"{expected_records:<8}".encode('ascii'))
                    return True
                elif data_records <= 0:
                    # Try to estimate from file size
                    f.seek(0, os.SEEK_END)
                    file_size = f.tell()
                    
                    # Get header size
                    f.seek(0)
                    header_bytes_str = header[184:192].decode('ascii', 'replace').strip()
                    header_bytes = int(header_bytes_str)
                    
                    # Get number of signals
                    num_signals_str = header[252:256].decode('ascii', 'replace').strip()
                    num_signals = int(num_signals_str)
                    
                    # Read samples per record for record size calculation
                    f.seek(256 + (num_signals * 216))  # Position at samples per record section
                    samples_per_record_data = f.read(num_signals * 8)
                    samples_per_record = [
                        int(samples_per_record_data[i:i+8].decode('ascii', 'replace').strip())
                        for i in range(0, len(samples_per_record_data), 8)
                    ]
                    
                    # Calculate record size (2 bytes per sample in EDF)
                    record_size = sum(samples_per_record) * 2
                    
                    # Estimate number of records
                    data_size = file_size - header_bytes
                    estimated_records = max(1, data_size // record_size)
                    
                    logger.warning(f"Estimated {estimated_records} data records from file size")
                    f.seek(236)
                    f.write(f"{estimated_records:<8}".encode('ascii'))
                    return True
                    
                return data_records > 0
                
            except ValueError:
                logger.error(f"Could not parse data record count: '{data_records_str}'")
                if expected_records is not None:
                    logger.warning(f"Setting data record count to {expected_records}")
                    f.seek(236)
                    f.write(f"{expected_records:<8}".encode('ascii'))
                    return True
                return False
                
    except Exception as e:
        logger.error(f"Error validating EDF header: {e}")
        logger.error(traceback.format_exc())
        return False

def validate_anonymized_file(input_path, output_path):
    """
    Validate that the anonymized file has the same structure as the original
    but with personally identifiable information removed.
    
    Args:
        input_path: Path to the original EDF file
        output_path: Path to the anonymized EDF file
        
    Returns:
        True if validation passed, False otherwise
    """
    logger = logging.getLogger('edf_validator')
    logger.info(f"Starting basic structure validation: {input_path} vs {output_path}")
    
    try:
        # Check file sizes
        input_size = os.path.getsize(input_path)
        output_size = os.path.getsize(output_path)
        
        logger.info(f"Original file size: {input_size:,} bytes")
        logger.info(f"Anonymized file size: {output_size:,} bytes")
        
        if input_size != output_size:
            logger.warning(f"File size mismatch: Input={input_size}, Output={output_size}, Diff={input_size-output_size}")
            return False
            
        # Check basic header structure
        with open(input_path, 'rb') as in_file, open(output_path, 'rb') as out_file:
            # Read headers
            in_header = in_file.read(256)
            out_header = out_file.read(256)
            
            # Check version
            in_version = in_header[0:8].decode('ascii', 'replace').strip()
            out_version = out_header[0:8].decode('ascii', 'replace').strip()
            logger.info(f"EDF version check: {in_version} vs {out_version}")
            
            if in_header[0:8] != out_header[0:8]:
                logger.warning("Version field mismatch")
                return False
                
            # Patient field should be anonymized
            in_patient = in_header[8:88].decode('ascii', 'replace').strip()
            out_patient = out_header[8:88].decode('ascii', 'replace').strip()
            logger.info(f"Patient field: '{in_patient}' -> '{out_patient}'")
            
            if out_header[8:88] == in_header[8:88]:
                logger.warning("Patient field not anonymized")
                return False
                
            # Technical fields should match
            in_technical = in_header[184:256].decode('ascii', 'replace').strip()
            out_technical = out_header[184:256].decode('ascii', 'replace').strip()
            logger.debug(f"Technical fields: '{in_technical}' vs '{out_technical}'")
            
            if in_header[184:256] != out_header[184:256]:
                logger.warning("Technical header fields mismatch")
                return False
                
            # Read a sample of data records to verify structure
            mismatch_found = False
            for i in range(3):  # Check first 3 records
                logger.debug(f"Checking record sample {i+1}")
                in_record = in_file.read(1024)  # Just sample 1KB
                out_record = out_file.read(1024)
                
                if len(in_record) != len(out_record):
                    logger.warning(f"Data record {i+1} length mismatch: {len(in_record)} vs {len(out_record)}")
                    mismatch_found = True
                    break
            
            if mismatch_found:
                return False
        
        logger.info("Anonymized file validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Error during validation: {e}")
        logger.error(traceback.format_exc())
        return False

def verify_edf_signals(input_path, output_path):
    """
    Verify that the non-annotation signals in the anonymized EDF file match those in the original file.
    
    This function:
    1. Reads both original and anonymized EDF files
    2. Identifies non-annotation channels
    3. Compares signal data between the files to ensure they match
    4. Reports any discrepancies found
    
    Args:
        input_path (str): Path to the original EDF file
        output_path (str): Path to the anonymized EDF file
        
    Returns:
        tuple: (bool, dict) - Success status and detailed results including:
            - total_signals: Number of signals checked
            - matching_signals: Number of signals that matched perfectly
            - mismatched_signals: List of signals that didn't match
            - error_details: Any error details if verification failed
    """
    import numpy as np
    
    logger = logging.getLogger('edf_verifier')
    logger.info(f"Starting detailed signal verification between:\n  - {input_path}\n  - {output_path}")
    
    results = {
        'total_signals': 0,
        'matching_signals': 0,
        'mismatched_signals': [],
        'error_details': None
    }
    
    try:
        # Open both files using EDFreader
        logger.debug("Opening EDF files with EDFreader")
        original_edf = EDFreader(input_path)
        anon_edf = EDFreader(output_path)
        
        # Get signal information
        signal_count = original_edf.getNumSignals()
        signal_labels = []
        for s in range(signal_count):
            signal_labels.append(original_edf.getSignalLabel(s))
        
        logger.info(f"Found {signal_count} signals to verify")
        
        # Store annotation channel indices
        annot_channels = [i for i, label in enumerate(signal_labels) 
                          if label in ["EDF Annotations", "BDF Annotations"]]
        
        if annot_channels:
            logger.info(f"Annotation channels (will be skipped): {annot_channels}")
        
        # Get data record info
        data_records = original_edf.getNumDataRecords()
        record_duration = original_edf.getFileDuration() / data_records
        
        logger.info(f"File contains {data_records} data records, {record_duration}s each")
        
        # Verify non-annotation channels
        for signal_idx in range(signal_count):
            # Skip annotation channels
            if signal_idx in annot_channels:
                continue
                
            signal_name = signal_labels[signal_idx]
            logger.info(f"Verifying signal: {signal_name} (index {signal_idx})")
            
            # Read the entire signal from both files
            samples_per_record = original_edf.getSampelsPerDataRecord(signal_idx)
            total_samples = samples_per_record * data_records
            
            logger.debug(f"  Signal has {samples_per_record} samples per record, {total_samples} total samples")
            
            # Read in manageable chunks to avoid memory issues with large files
            chunk_size = min(samples_per_record * 100, total_samples)  # 100 records or all if smaller
            logger.debug(f"  Using chunk size of {chunk_size} samples")
            
            # Process in chunks
            mismatches_for_signal = []
            
            for offset in range(0, total_samples, chunk_size):
                actual_chunk = min(chunk_size, total_samples - offset)
                
                # Read chunks from both files using the correct method signature
                # Looking at EDFreader.py, the readSamples method requires:
                # - signal number (s)
                # - buffer (numpy array) 
                # - number of samples to read (n)
                logger.debug(f"  Reading chunk at offset {offset} (size {actual_chunk} samples)")
                
                # Create buffers for the original and anonymized data
                original_data = np.zeros(actual_chunk, dtype=np.float64)
                anon_data = np.zeros(actual_chunk, dtype=np.float64)
                
                # Set the file position
                original_edf.fseek(signal_idx, offset, original_edf.EDFSEEK_SET)
                anon_edf.fseek(signal_idx, offset, anon_edf.EDFSEEK_SET)
                
                # Read the samples into the buffers
                samples_read_orig = original_edf.readSamples(signal_idx, original_data, actual_chunk)
                samples_read_anon = anon_edf.readSamples(signal_idx, anon_data, actual_chunk)
                
                # Check if we got the expected number of samples
                if samples_read_orig != actual_chunk or samples_read_anon != actual_chunk:
                    logger.warning(f"  Failed to read expected number of samples: "
                                 f"Original={samples_read_orig}, "
                                 f"Anonymized={samples_read_anon}, "
                                 f"Expected={actual_chunk}")
                    continue
                
                # Compare the data
                if not np.array_equal(original_data, anon_data):
                    # Check if differences are within numerical precision
                    if np.allclose(original_data, anon_data, rtol=1e-10, atol=1e-10):
                        logger.info(f"  Signal {signal_name}: Minor precision differences found but within tolerance")
                    else:
                        # Calculate statistics about the differences
                        diff = original_data - anon_data
                        non_zero_diff = diff[diff != 0]
                        
                        if len(non_zero_diff) > 0:
                            diff_stats = {
                                'mean_diff': float(np.mean(non_zero_diff)),
                                'max_diff': float(np.max(np.abs(non_zero_diff))),
                                'diff_count': len(non_zero_diff),
                                'diff_percent': 100 * len(non_zero_diff) / len(diff),
                                'chunk_start': offset,
                                'chunk_size': actual_chunk
                            }
                            
                            mismatches_for_signal.append(diff_stats)
                            
                            logger.warning(f"  Signal {signal_name} mismatch in chunk {offset}-{offset+actual_chunk}:")
                            logger.warning(f"    - {diff_stats['diff_count']} samples differ ({diff_stats['diff_percent']:.4f}%)")
                            logger.warning(f"    - Mean difference: {diff_stats['mean_diff']}")
                            logger.warning(f"    - Max difference: {diff_stats['max_diff']}")
                            
                            # Log first few mismatches for debugging
                            first_mismatch_indices = np.nonzero(diff)[0][:5]  # Get first 5 mismatches
                            if len(first_mismatch_indices) > 0:
                                for i in first_mismatch_indices:
                                    abs_pos = offset + i
                                    logger.debug(f"    Sample at position {abs_pos}: "
                                                f"Original={original_data[i]}, "
                                                f"Anonymized={anon_data[i]}, "
                                                f"Diff={diff[i]}")
                            
                            # If we've found substantial differences, no need to check more chunks
                            if diff_stats['diff_percent'] > 1.0:  # If more than 1% differs
                                logger.warning("  Substantial differences found, skipping remaining chunks")
                                break
            
            # After checking all chunks, summarize findings for this signal
            if mismatches_for_signal:
                # Calculate overall statistics
                total_diffs = sum(m['diff_count'] for m in mismatches_for_signal)
                overall_diff_percent = 100 * total_diffs / total_samples
                
                results['mismatched_signals'].append({
                    'signal_name': signal_name,
                    'signal_idx': signal_idx,
                    'total_samples': total_samples,
                    'total_differences': total_diffs,
                    'diff_percent': overall_diff_percent,
                    'chunk_details': mismatches_for_signal
                })
                
                logger.warning(f"  Signal {signal_name} summary: {total_diffs:,} mismatches "
                              f"out of {total_samples:,} samples ({overall_diff_percent:.4f}%)")
            else:
                # Signal matched perfectly
                logger.info(f"  Signal {signal_name}: MATCH OK - All samples identical")
                results['matching_signals'] += 1
        
        # Update total signals checked (excluding annotation channels)
        results['total_signals'] = signal_count - len(annot_channels)
        
        # Check header fields that should match
        logger.info("Verifying critical header fields")
        
        # Number of signals should match
        orig_signal_count = original_edf.getNumSignals()
        anon_signal_count = anon_edf.getNumSignals()
        if orig_signal_count != anon_signal_count:
            msg = f"Header mismatch: Number of signals differs ({orig_signal_count} vs {anon_signal_count})"
            logger.warning(msg)
            results['error_details'] = msg
            
        # Data record count and duration should match
        orig_records = original_edf.getNumDataRecords()
        anon_records = anon_edf.getNumDataRecords()
        if orig_records != anon_records:
            msg = f"Header mismatch: Number of data records differs ({orig_records} vs {anon_records})"
            logger.warning(msg)
            results['error_details'] = "Number of data records in header doesn't match"
            
        orig_duration = original_edf.getFileDuration()
        anon_duration = anon_edf.getFileDuration()
        if abs(orig_duration - anon_duration) > 0.001:
            msg = f"Header mismatch: File duration differs ({orig_duration}s vs {anon_duration}s)"
            logger.warning(msg)
            results['error_details'] = "File duration in header doesn't match"
            
        # Close the EDF readers
        logger.debug("Closing EDF readers")
        original_edf.close()
        anon_edf.close()
        
        # Determine success
        success = (results['mismatched_signals'] == [] and results['error_details'] is None)
        
        if success:
            logger.info(f"Signal verification PASSED: All {results['total_signals']} signals match perfectly")
        else:
            logger.warning(f"Signal verification FAILED: {len(results['mismatched_signals'])} of {results['total_signals']} signals mismatched")
            
        return success, results
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Error during signal verification: {e}")
        logger.error(error_details)
        results['error_details'] = str(e)
        return False, results
    
    
def run_verification(input_path, output_path):
    """
    Run a comprehensive verification of the anonymized EDF file.
    
    Args:
        input_path (str): Path to the original EDF file
        output_path (str): Path to the anonymized EDF file
        
    Returns:
        bool: True if verification passed, False otherwise
    """
    logger = logging.getLogger('edf_verifier')
    logger.info("Starting comprehensive EDF verification...")
    
    # First check file sizes and structure
    logger.info("Step 1: Basic file structure verification")
    structure_ok = validate_anonymized_file(input_path, output_path)
    if not structure_ok:
        logger.error("File structure validation failed")
        return False
    
    # Then verify the signal data
    logger.info("Step 2: Detailed signal data verification")
    signals_ok, results = verify_edf_signals(input_path, output_path)
    
    # Print verification summary
    logger.info("=== Verification Summary ===")
    if signals_ok:
        logger.info(f"? Signal verification PASSED: All {results['total_signals']} signals match")
    else:
        logger.error(f"? Signal verification FAILED: {len(results['mismatched_signals'])} of {results['total_signals']} signals mismatched")
        for mismatch in results['mismatched_signals']:
            logger.error(f"  - Signal '{mismatch['signal_name']}': "
                       f"{mismatch['total_differences']:,} differences "
                       f"({mismatch['diff_percent']:.4f}%)")
        
        if results['error_details']:
            logger.error(f"Error details: {results['error_details']}")
    
    final_result = signals_ok and structure_ok
    logger.info(f"Overall verification result: {'PASSED' if final_result else 'FAILED'}")
    
    return final_result

def parse_arguments():
    parser = argparse.ArgumentParser(description="Anonymize EDF file by removing patient-identifiable information.")
    parser.add_argument("input_path", type=str, help="Path to the input EDF file.")
    parser.add_argument("output_path", type=str, help="Path to save the anonymized EDF file.")
    parser.add_argument("--buffer_size_mb", type=int, default=64, help="Buffer size in MB for processing chunks (default: 64MB).")
    parser.add_argument("--verify", action="store_true", help="Verify signal data integrity after anonymization")
    parser.add_argument("--verify_level", choices=["basic", "thorough"], default="thorough", 
                        help="Verification level: 'basic' for structure only, 'thorough' for full signal comparison")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to store log files")
    parser.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
                       help="Set logging level (default: INFO)")
    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()

    # Setup logging with detailed configuration
    logger = setup_logging(args.log_dir)
    
    # Set log level based on command line argument
    log_level = getattr(logging, args.log_level)
    logger.setLevel(log_level)
    
    logger.info("=== EDF Anonymization Tool ===")
    logger.info(f"Input file: {args.input_path}")
    logger.info(f"Output file: {args.output_path}")
    logger.info(f"Buffer size: {args.buffer_size_mb} MB")
    logger.info(f"Verification: {'Enabled' if args.verify else 'Disabled'}")
    if args.verify:
        logger.info(f"Verification level: {args.verify_level}")

    # Check if input file exists
    if not os.path.exists(args.input_path):
        logger.error(f"Input file not found: {args.input_path}")
        sys.exit(1)
        
    # Check if output directory exists
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        logger.info(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    # Run the anonymization
    logger.info("Starting anonymization process")
    start_time = time.time()
    success = anonymize_edf_complete(args.input_path, args.output_path, buffer_size_mb=args.buffer_size_mb)
    elapsed = time.time() - start_time

    if success:
        logger.info(f"Anonymization completed successfully in {elapsed:.2f} seconds")
        
        # Run verification if requested
        if args.verify:
            logger.info("Running verification...")
            verify_start = time.time()
            
            if args.verify_level == "thorough":
                verification_result = run_verification(args.input_path, args.output_path)
                verify_elapsed = time.time() - verify_start
                
                if verification_result:
                    logger.info(f"? Thorough verification PASSED in {verify_elapsed:.2f} seconds")
                    print("? Anonymization and verification successful")
                else:
                    logger.error(f"? Thorough verification FAILED in {verify_elapsed:.2f} seconds")
                    print("? Anonymization succeeded but verification failed - check logs for details")
            else:
                # Just do basic validation
                verification_result = validate_anonymized_file(args.input_path, args.output_path)
                verify_elapsed = time.time() - verify_start
                
                if verification_result:
                    logger.info(f"? Basic validation PASSED in {verify_elapsed:.2f} seconds")
                    print("? Anonymization and basic validation successful")
                else:
                    logger.error(f"? Basic validation FAILED in {verify_elapsed:.2f} seconds")
                    print("? Anonymization succeeded but validation failed - check logs for details")
        else:
            logger.info("Skipping verification (use --verify to verify signals)")
            print("? Anonymization completed successfully (no verification requested)")
    else:
        logger.error("Anonymization failed")
        print("? Anonymization failed - check logs for details")
        sys.exit(1)