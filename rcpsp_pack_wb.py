
#!/usr/bin/env python3
"""
RCPSP solver using linear search from upper bound down to lower bound.
This approach tries each makespan value sequentially to find the optimal solution.
No time limit per makespan test - only overall 900s limit.
"""

"TODO: Tìm cách add trực tiếp bounds thành các constraint, ví dụ: Max(e1, e2,..., et) = 60"

from docplex.cp.model import *
import os
import csv
import time
from pathlib import Path
from google.cloud import storage
import os

# Thời gian tối đa cho mỗi instance
TIME_PER_INSTANCE = 900


def solve_rcpsp_with_direct_bounds(data_file, lower_bound, upper_bound, time_limit):
    """
    Solve RCPSP with bounds added directly to max expression
    """
    try:
        # Read data file (same as before)
        with open(data_file, 'r') as file:
            first_line = file.readline().split()
            NB_TASKS, NB_RESOURCES = int(first_line[0]), int(first_line[1])
            CAPACITIES = [int(v) for v in file.readline().split()]
            TASKS = [[int(v) for v in file.readline().split()] for i in range(NB_TASKS)]

        DURATIONS = [TASKS[t][0] for t in range(NB_TASKS)]
        DEMANDS = [TASKS[t][1:NB_RESOURCES + 1] for t in range(NB_TASKS)]
        SUCCESSORS = [TASKS[t][NB_RESOURCES + 2:] for t in range(NB_TASKS)]

        # Create CP model
        mdl = CpoModel()

        # Create interval variables
        tasks = [interval_var(name=f'T{i + 1}', size=DURATIONS[i]) for i in range(NB_TASKS)]

        # Add precedence constraints
        for t in range(NB_TASKS):
            for s in SUCCESSORS[t]:
                if s > 0:
                    mdl.add(end_before_start(tasks[t], tasks[s - 1]))

        # Add resource constraints
        for r in range(NB_RESOURCES):
            resource_usage = [pulse(tasks[t], DEMANDS[t][r]) for t in range(NB_TASKS) if DEMANDS[t][r] > 0]
            if resource_usage:
                mdl.add(sum(resource_usage) <= CAPACITIES[r])

        # CREATE MAKESPAN VARIABLE
        makespan = integer_var(lower_bound, upper_bound, name="makespan")

        # ADD BOUNDS DIRECTLY TO MAX EXPRESSION
        mdl.add(makespan == max(end_of(t) for t in tasks))
        mdl.add(makespan >= lower_bound)
        mdl.add(makespan <= upper_bound)

        # ADD OBJECTIVE TO MINIMIZE MAKESPAN
        mdl.add(minimize(makespan))

        # Solve
        res = mdl.solve(TimeLimit=time_limit, LogVerbosity="Quiet")

        if res and res.is_solution():
            actual_makespan = res.get_value(makespan)
            return actual_makespan
        else:
            return None

    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def solve_rcpsp_linear_search(data_file):
    """
    Solve RCPSP using linear search from upper bound down to lower bound
    Only overall TIME_PER_INSTANCE time limit - no limit per makespan test
    """
    start_time = time.time()

    try:
        # Read bounds from data file
        with open(data_file, 'r') as file:
            first_line = file.readline().split()
            NB_TASKS, NB_RESOURCES = int(first_line[0]), int(first_line[1])

            # Read bounds
            LOWER_BOUND = UPPER_BOUND = None
            if len(first_line) >= 4:
                LOWER_BOUND = int(first_line[2])
                UPPER_BOUND = int(first_line[3])
                print(f"Bounds from file: LB={LOWER_BOUND}, UB={UPPER_BOUND}")
            else:
                print("No bounds specified in file")
                return (None, None, None, "infeasible", time.time() - start_time)

        if LOWER_BOUND is None or UPPER_BOUND is None:
            print("Invalid bounds")
            return (None, None, None, "infeasible", time.time() - start_time)

        # Linear search from upper bound down to lower bound
        print(f"Starting linear search from {UPPER_BOUND} down to {LOWER_BOUND}")
        print(f"Total time limit: {TIME_PER_INSTANCE}s")

        optimal_makespan = None
        attempts = 0
        timeout_occurred = False

        for makespan in range(UPPER_BOUND, LOWER_BOUND - 1, -1):
            attempts += 1
            elapsed = time.time() - start_time
            time_remaining = TIME_PER_INSTANCE - elapsed

            # Check total time limit
            if time_remaining <= 0:
                print(f"Total time limit exceeded after {attempts} attempts")
                timeout_occurred = True
                break

            print(f"  Attempt {attempts}: Testing makespan = {makespan}")
            print(f"    Elapsed: {elapsed:.1f}s, Remaining: {time_remaining:.1f}s")

            # Test if this makespan is feasible
            attempt_start = time.time()
            is_feasible = solve_rcpsp_with_direct_bounds(data_file, LOWER_BOUND, UPPER_BOUND, time_remaining)
            attempt_time = time.time() - attempt_start

            print(f"    Attempt took: {attempt_time:.1f}s")

            if is_feasible:
                optimal_makespan = makespan
                print(f"  ✓ Makespan {makespan} is FEASIBLE")

                # Continue searching for better (smaller) makespan
                continue
            else:
                print(f"  ✗ Makespan {makespan} is INFEASIBLE")

                # If current makespan is infeasible, then optimal_makespan
                # (if found) is the best we can do
                break

        solve_time = time.time() - start_time

        if optimal_makespan is not None:
            # Determine status based on timeout and optimality
            if solve_time > TIME_PER_INSTANCE or timeout_occurred:
                # Nếu chạy quá thời gian cho phép thì status là feasible
                status = "feasible"
                print(f"✓ Found FEASIBLE solution (timeout): {optimal_makespan}")
            elif optimal_makespan == LOWER_BOUND:
                # Nếu tìm được lower bound và không timeout thì optimal
                status = "optimal"
                print(f"✓ Found OPTIMAL solution: {optimal_makespan} (matches lower bound)")
            else:
                # Nếu không timeout nhưng chưa đạt lower bound thì feasible
                status = "feasible"
                print(f"✓ Found FEASIBLE solution: {optimal_makespan}")

            print(f"Linear search completed: tested {attempts} values in {solve_time:.2f}s")
            return (LOWER_BOUND, UPPER_BOUND, optimal_makespan, status, solve_time)
        else:
            print(f"✗ No feasible solution found in range [{LOWER_BOUND}, {UPPER_BOUND}]")
            return (LOWER_BOUND, UPPER_BOUND, None, "infeasible", solve_time)

    except Exception as e:
        solve_time = time.time() - start_time
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return (None, None, None, "infeasible", solve_time)


def main():
    # Define directories
    data_dir = Path("data")
    result_dir = Path("result")
    output_file = result_dir / "pack_with_bound_900s.csv"

    # Create result directory if it doesn't exist
    os.makedirs(result_dir, exist_ok=True)

    # Find all .data files in the data directory
    all_data_files = list(data_dir.glob("*.data"))
    if not all_data_files:
        print(f"Warning: No .data files found in {data_dir}")
        print(f"Current directory: {os.getcwd()}")
        print("Directory contents:")
        for item in os.listdir():
            print(f"  {item}")
        return

    # Sort files to ensure consistent order
    all_data_files.sort()

    # Process ALL files instead of a specific range
    data_files = all_data_files

    print(f"Found {len(all_data_files)} total .data files")
    print(f"Processing ALL {len(data_files)} files in the data directory")
    print(f"Using {TIME_PER_INSTANCE} seconds time limit per instance")
    print("Strategy: Linear search from upper bound down to lower bound")

    # Process files
    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Updated CSV header as requested
        csv_writer.writerow(["File name", "LB", "UB", "Makespan", "Status", "Solve time (second)"])

        for i, data_file in enumerate(data_files, 1):
            file_name = data_file.name
            print(f"\n{'=' * 60}")
            print(f"[{i}/{len(data_files)}] Processing {file_name}")
            print(f"{'=' * 60}")

            try:
                lb, ub, makespan, status, solve_time = solve_rcpsp_linear_search(data_file)

                # Format values for CSV
                lb_str = str(lb) if lb is not None else "N/A"
                ub_str = str(ub) if ub is not None else "N/A"
                makespan_str = str(makespan) if makespan is not None else "N/A"

                csv_writer.writerow([
                    file_name,
                    lb_str,
                    ub_str,
                    makespan_str,
                    status,
                    f"{solve_time:.2f}"
                ])
                csvfile.flush()

                print(f"\nFINAL RESULT:")
                print(f"  File: {file_name}")
                print(f"  LB: {lb_str}")
                print(f"  UB: {ub_str}")
                print(f"  Makespan: {makespan_str}")
                print(f"  Status: {status}")
                print(f"  Time: {solve_time:.2f}s")

            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")
                csv_writer.writerow([file_name, "N/A", "N/A", "N/A", "infeasible", "0.00"])
                csvfile.flush()

    print(f"\n{'=' * 60}")
    print(f"ALL PROCESSING COMPLETE")
    print(f"Results saved to: {output_file}")
    print(f"{'=' * 60}")

    # Tên bucket mà bạn đã tạo
    bucket_name = "rcpsp-with-bounds-results-bucket"
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    local_path = "result/pack_with_bound_900s.csv"
    blob_name = f"results/{os.path.basename(local_path)}"  # ví dụ "results/pack_with_bound_900s.csv"

    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    print(f"Uploaded {local_path} to gs://{bucket_name}/{blob_name}")

if __name__ == "__main__":
    main()