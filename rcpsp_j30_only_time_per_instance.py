#!/usr/bin/env python3
"""
RCPSP solver using linear search from upper bound down to lower bound.
This approach tries each makespan value sequentially to find the optimal solution.
No time limit per makespan test - only overall 900s limit.
"""
from docplex.cp.model import *
import os
import csv
import time
from pathlib import Path


def solve_rcpsp_with_makespan_bound(data_file, target_makespan, time_remaining):
    """
    Solve RCPSP with fixed makespan upper bound constraint
    Returns True if feasible, False if infeasible
    No individual time limit - runs until solution found or time_remaining expires
    """
    try:
        # Read data file
        with open(data_file, 'r') as file:
            first_line = file.readline().split()
            NB_TASKS, NB_RESOURCES = int(first_line[0]), int(first_line[1])

            CAPACITIES = [int(v) for v in file.readline().split()]
            TASKS = [[int(v) for v in file.readline().split()] for i in range(NB_TASKS)]

        # Extract data
        DURATIONS = [TASKS[t][0] for t in range(NB_TASKS)]
        DEMANDS = [TASKS[t][1:NB_RESOURCES + 1] for t in range(NB_TASKS)]
        SUCCESSORS = [TASKS[t][NB_RESOURCES + 2:] for t in range(NB_TASKS)]

        # Create CP model
        mdl = CpoModel()

        # Create interval variables for tasks
        tasks = [interval_var(name=f'T{i + 1}', size=DURATIONS[i]) for i in range(NB_TASKS)]

        # Add precedence constraints
        for t in range(NB_TASKS):
            for s in SUCCESSORS[t]:
                if s > 0:  # Valid successor
                    mdl.add(end_before_start(tasks[t], tasks[s - 1]))

        # Add resource capacity constraints
        for r in range(NB_RESOURCES):
            resource_usage = [pulse(tasks[t], DEMANDS[t][r]) for t in range(NB_TASKS) if DEMANDS[t][r] > 0]
            if resource_usage:
                mdl.add(sum(resource_usage) <= CAPACITIES[r])

        # Create makespan and add FIXED bound constraint
        makespan = max(end_of(t) for t in tasks)
        mdl.add(makespan <= target_makespan)

        # Solve with remaining time - removed Workers and FailLimit
        time_to_use = max(1, time_remaining)

        res = mdl.solve(
            TimeLimit=time_to_use,
            LogVerbosity="Quiet"
        )

        return res is not None

    except Exception as e:
        print(f"Error solving with makespan {target_makespan}: {str(e)}")
        return False


def solve_rcpsp_linear_search(data_file):
    """
    Solve RCPSP using linear search from upper bound down to lower bound
    Only overall 900s time limit - no limit per makespan test
    """
    TOTAL_TIME_LIMIT = 900
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
            elif len(first_line) == 3:
                LOWER_BOUND = UPPER_BOUND = int(first_line[2])
                print(f"Single bound from file: {LOWER_BOUND}")
            else:
                print("No bounds specified in file")
                return (None, None, None, "infeasible", time.time() - start_time)

        if LOWER_BOUND is None or UPPER_BOUND is None:
            print("Invalid bounds")
            return (None, None, None, "infeasible", time.time() - start_time)

        # Linear search from upper bound down to lower bound
        print(f"Starting linear search from {UPPER_BOUND} down to {LOWER_BOUND}")
        print(f"Total time limit: {TOTAL_TIME_LIMIT}s")

        optimal_makespan = None
        attempts = 0

        for makespan in range(UPPER_BOUND, LOWER_BOUND - 1, -1):
            attempts += 1
            elapsed = time.time() - start_time
            time_remaining = TOTAL_TIME_LIMIT - elapsed

            # Check total time limit
            if time_remaining <= 0:
                print(f"Total time limit exceeded after {attempts} attempts")
                break

            print(f"  Attempt {attempts}: Testing makespan = {makespan}")
            print(f"    Elapsed: {elapsed:.1f}s, Remaining: {time_remaining:.1f}s")

            # Test if this makespan is feasible
            attempt_start = time.time()
            is_feasible = solve_rcpsp_with_makespan_bound(data_file, makespan, time_remaining)
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
            # Determine simplified status
            if optimal_makespan == LOWER_BOUND:
                status = "optimal"
                print(f"✓ Found OPTIMAL solution: {optimal_makespan} (matches lower bound)")
            else:
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
    # Directories
    data_dir = Path("Convert/data/j120")
    result_dir = Path("result")
    output_file = result_dir / "j120_linear_search.csv"

    os.makedirs(result_dir, exist_ok=True)

    # Find data files
    data_files = list(data_dir.glob("*.data"))
    if not data_files:
        print(f"No .data files found in {data_dir}")
        return

    print(f"Found {len(data_files)} files to process")
    print("Strategy: Linear search from upper bound down to lower bound")
    print("No time limit per makespan test - only 900s total limit per instance")

    # Process files
    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Updated CSV header as requested
        csv_writer.writerow(["File name", "LB", "UB", "Makespan", "Status", "Solve time (second)"])

        for i, data_file in enumerate(sorted(data_files), 1):
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


if __name__ == "__main__":
    main()