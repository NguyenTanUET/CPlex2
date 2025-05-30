#!/usr/bin/env python3
"""
batch_convert_sm.py

Script to convert multiple RCPSP .sm files to .data format with optimal bounds
for use with rcpsp.py.

Usage:
    python batch_convert_sm.py [source_dir] [bounds_csv] [output_dir]

If no arguments are provided, defaults to:
    - source_dir: ./raw_data
    - bounds_csv: ./bound_j30.sm.csv
    - output_dir: ./data
"""
import sys
import os
import re
import csv
import traceback
from pathlib import Path


def parse_sm(sm_path):
    """
    Parse a .sm file and extract relevant data:
    - number of jobs
    - number of resources
    - resource capacities
    - durations, demands, and successors for each task
    """
    try:
        # Read file content
        with open(sm_path, 'r') as f:
            content = f.read()

        # Split into lines and filter out comments and empty lines
        lines = [line.strip() for line in content.splitlines()
                 if line.strip() and not line.strip().startswith('*')]

        # Extract number of jobs
        n_jobs = None
        for line in lines:
            if 'jobs (incl. supersource/sink' in line.lower():
                match = re.search(r':\s*(\d+)', line)
                if match:
                    n_jobs = int(match.group(1))
                    break

        if n_jobs is None:
            raise ValueError(f"Could not find number of jobs in {sm_path}")

        # Extract number of resources
        m_resources = None
        for line in lines:
            if 'renewable' in line.lower():
                match = re.search(r':\s*(\d+)', line)
                if match:
                    m_resources = int(match.group(1))
                    break

        if m_resources is None:
            raise ValueError(f"Could not find number of resources in {sm_path}")

        # Extract resource capacities
        capacities = []
        resource_section = False
        for i, line in enumerate(lines):
            if "RESOURCEAVAILABILITIES" in line.upper():
                resource_section = True
                # Look for capacity values in the next few lines
                for j in range(i + 1, min(i + 10, len(lines))):
                    tokens = lines[j].split()
                    # Check if this line contains just numbers matching the resource count
                    if tokens and all(token.isdigit() for token in tokens):
                        if m_resources == len(tokens):
                            capacities = list(map(int, tokens))
                            break
                break

        if not capacities:
            raise ValueError(f"Could not find resource capacities in {sm_path}")

        # Extract task durations and demands
        durations = {}
        demands = {}

        # Find the REQUESTS/DURATIONS section
        req_start_idx = -1
        for i, line in enumerate(lines):
            if "REQUESTS/DURATIONS" in line.upper():
                req_start_idx = i
                break

        if req_start_idx == -1:
            raise ValueError(f"Could not find REQUESTS/DURATIONS section in {sm_path}")

        # Skip header lines (usually 2-3 lines)
        current_idx = req_start_idx + 1
        while current_idx < len(lines):
            line = lines[current_idx]
            if re.match(r'^\s*\d+\s+\d+', line):  # Line starts with numbers
                break
            current_idx += 1

        # Parse task data
        for i in range(current_idx, len(lines)):
            line = lines[i]
            if not line or line.upper().startswith("PRECEDENCE") or line.upper().startswith("*"):
                break

            parts = line.split()
            if len(parts) < 3 + m_resources:  # Task ID, mode, duration, demands
                continue

            try:
                task_id = int(parts[0])
                duration = int(parts[2])
                demand = [int(parts[3 + j]) for j in range(m_resources)]

                durations[task_id] = duration
                demands[task_id] = demand
            except (ValueError, IndexError) as e:
                print(f"Warning: Error parsing task data in line {i + 1}: {line}")
                print(f"Error details: {e}")

        # Find precedence relations
        successors = {}
        prec_start_idx = -1

        for i, line in enumerate(lines):
            if "PRECEDENCE RELATIONS" in line.upper():
                prec_start_idx = i
                break

        if prec_start_idx != -1:
            # Skip header lines
            current_idx = prec_start_idx + 1
            while current_idx < len(lines):
                line = lines[current_idx]
                if re.match(r'^\s*\d+\s+\d+', line):  # Line starts with numbers
                    break
                current_idx += 1

            # Parse precedence data
            for i in range(current_idx, len(lines)):
                line = lines[i]
                if not line or line.upper().startswith("REQUESTS") or line.upper().startswith("*"):
                    break

                parts = line.split()
                if len(parts) < 3:  # Need at least task ID, mode, and successor count
                    continue

                try:
                    task_id = int(parts[0])
                    succ_count = int(parts[2])

                    if len(parts) >= 3 + succ_count:
                        successors[task_id] = [int(parts[3 + j]) for j in range(succ_count)]
                    else:
                        successors[task_id] = []
                except (ValueError, IndexError) as e:
                    print(f"Warning: Error parsing precedence data in line {i + 1}: {line}")
                    print(f"Error details: {e}")

        # Build tasks list
        tasks = []
        for i in range(1, n_jobs + 1):
            duration = durations.get(i, 0)  # Default to 0 if missing
            demand = demands.get(i, [0] * m_resources)  # Default to zeros if missing
            succ = successors.get(i, [])

            tasks.append({
                'duration': duration,
                'demands': demand,
                'successors': succ
            })

        return {
            'n_jobs': n_jobs,
            'n_resources': m_resources,
            'capacities': capacities,
            'tasks': tasks
        }

    except Exception as e:
        print(f"Error parsing {sm_path}: {e}")
        traceback.print_exc()
        raise


def load_bounds(csv_path):
    """
    Load bounds from CSV file with format: instance_name, lower_bound, upper_bound
    Returns a dictionary mapping instance names to bound value (using only the first bound).
    """
    bounds = {}
    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:  # Need at least filename and one bound
                    continue

                instance = row[0].strip()
                try:
                    # Only use the first bound (lower bound)
                    bound = int(row[1].strip())

                    # Store with multiple key formats for easier matching
                    base_name = Path(instance).stem
                    if base_name.endswith('.sm'):
                        base_name = base_name[:-3]

                    bounds[instance] = bound
                    bounds[base_name] = bound
                    bounds[base_name.replace('_', '')] = bound
                except ValueError:
                    continue

        return bounds
    except Exception as e:
        print(f"Error loading bounds from {csv_path}: {e}")
        return {}


def write_data_file(sm_file, output_dir, sm_data, bound=None):
    """
    Write the data in .data format to the specified output directory.
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Determine filename
        base_name = Path(sm_file).stem
        out_file = Path(output_dir) / f"{base_name}.data"

        with open(out_file, 'w') as f:
            # First line: number of tasks, number of resources, [optimal bound]
            if bound is not None:
                f.write(f"{sm_data['n_jobs']} {sm_data['n_resources']} {bound}\n")
            else:
                f.write(f"{sm_data['n_jobs']} {sm_data['n_resources']}\n")

            # Second line: resource capacities
            f.write(" ".join(map(str, sm_data['capacities'])) + "\n")

            # Remaining lines: one per task
            for task in sm_data['tasks']:
                # Format: duration demand1 demand2 ... demandN numSuccessors successor1 successor2 ...
                line_parts = [str(task['duration'])]
                line_parts.extend(map(str, task['demands']))
                line_parts.append(str(len(task['successors'])))
                line_parts.extend(map(str, task['successors']))
                f.write(" ".join(line_parts) + "\n")

        return out_file
    except Exception as e:
        print(f"Error writing to {output_dir}: {e}")
        raise


def find_matching_bound(filename, bounds):
    """
    Try various combinations to find a matching bound for the given filename.
    """
    base_name = Path(filename).stem

    # Try direct matches
    if filename in bounds:
        return bounds[filename]
    if base_name in bounds:
        return bounds[base_name]

    # Try without underscore
    no_underscore = base_name.replace('_', '')
    if no_underscore in bounds:
        return bounds[no_underscore]

    # Try with .sm extension
    if f"{base_name}.sm" in bounds:
        return bounds[f"{base_name}.sm"]

    # Pattern matching
    for key in bounds:
        # Match j301_5 with j3015 or similar patterns
        if re.search(r'j30(\d+)[_]?(\d+)', base_name) and re.search(r'j30(\d+)[_]?(\d+)', key):
            base_match = re.search(r'j30(\d+)[_]?(\d+)', base_name)
            key_match = re.search(r'j30(\d+)[_]?(\d+)', key)

            if base_match and key_match and base_match.group(1) == key_match.group(1) and base_match.group(
                    2) == key_match.group(2):
                return bounds[key]

    # If no match found, try a more general search
    base_number = re.search(r'j30(\d+)', base_name)
    if base_number:
        base_num = base_number.group(1)
        for key in bounds:
            if f"j30{base_num}" in key:
                return bounds[key]

    return None


def main():
    # Parse command-line arguments
    args = sys.argv[1:]

    if len(args) >= 3:
        source_dir = args[0]
        bounds_csv = args[1]
        output_dir = args[2]
    elif len(args) == 2:
        source_dir = args[0]
        bounds_csv = args[1]
        output_dir = "Convert/data/j120"
    elif len(args) == 1:
        source_dir = args[0]
        bounds_csv = "Convert/bounds/bound_j120.sm.csv"
        output_dir = "Convert/data/j120"
    else:
        source_dir = "Convert/raw_data/j120"
        bounds_csv = "Convert/bounds/bound_j120.sm.csv"
        output_dir = "Convert/data/j120"

    # Ensure paths are Path objects
    source_dir = Path(source_dir)
    bounds_csv = Path(bounds_csv)
    output_dir = Path(output_dir)

    # Check if source directory exists
    if not source_dir.exists() or not source_dir.is_dir():
        print(f"Error: Source directory '{source_dir}' does not exist or is not a directory")
        sys.exit(1)

    # Check if bounds CSV exists
    if not bounds_csv.exists():
        print(f"Warning: Bounds CSV '{bounds_csv}' does not exist. Proceeding without bounds.")
        bounds = {}
    else:
        # Load bounds
        bounds = load_bounds(bounds_csv)
        print(f"Loaded {len(bounds)} bounds from {bounds_csv}")

    # Process all .sm files in the source directory
    sm_files = list(source_dir.glob("*.sm"))
    print(f"Found {len(sm_files)} .sm files to process")

    successful = 0
    failed = 0

    for sm_file in sm_files:
        try:
            print(f"Processing {sm_file.name}...", end="", flush=True)

            # Parse .sm file
            sm_data = parse_sm(sm_file)

            # Find matching bound
            bound = find_matching_bound(sm_file.name, bounds)
            if bound is not None:
                print(f" found bound: {bound}", end="", flush=True)
                # Use the first bound value directly
                optimal_bound = bound
            else:
                print(f" no bound found", end="", flush=True)
                optimal_bound = None

            # Write .data file
            out_file = write_data_file(sm_file, output_dir, sm_data, optimal_bound)
            print(f" -> written to {out_file}")

            successful += 1
        except Exception as e:
            print(f" ERROR: {str(e)}")
            failed += 1

    print(f"\nConversion complete: {successful} files converted successfully, {failed} failed")
    print(f"Output files are in: {output_dir}")


if __name__ == "__main__":
    main()