# !/usr/bin/env python3
"""
RCPSP solver that processes all .data files and outputs results to CSV.
Each instance gets a fixed 20 seconds of solve time.

Usage:
    python rcpsp_j60_only_time_per_instance.py

This script:
1. Finds all .data files in the Convert/data directory
2. Solves each RCPSP instance using the CP Optimizer with a 20-second limit per instance
3. Records results in result/j60.csv with columns:
   - file name (just the filename, not the path)
   - Model constraint (makespan found)
   - Status (optimal/feasible/unknown)
   - Solve time (in seconds)
"""
from docplex.cp.model import *
import os
import sys
import csv
import time
from pathlib import Path


def solve_rcpsp(data_file):
    """
    Solve the RCPSP problem for the given data file with fixed 20-second time limit
    Returns tuple: (makespan, status, solve_time)
    """
    start_time = time.time()

    # Fixed time limit per instance (in seconds)
    TIME_PER_INSTANCE = 900

    try:
        # Read the input data file
        with open(data_file, 'r') as file:
            first_line = file.readline().split()
            NB_TASKS, NB_RESOURCES = int(first_line[0]), int(first_line[1])

            # Check if there's a bound specified
            OPTIMAL_BOUND = None
            if len(first_line) > 2:
                OPTIMAL_BOUND = int(first_line[2])
                print(f"Using bound: {OPTIMAL_BOUND} from file {data_file.name}")

            CAPACITIES = [int(v) for v in file.readline().split()]
            TASKS = [[int(v) for v in file.readline().split()] for i in range(NB_TASKS)]

        # Extract data
        DURATIONS = [TASKS[t][0] for t in range(NB_TASKS)]
        DEMANDS = [TASKS[t][1:NB_RESOURCES + 1] for t in range(NB_TASKS)]
        SUCCESSORS = [TASKS[t][NB_RESOURCES + 2:] for t in range(NB_TASKS)]

        # Create model
        mdl = CpoModel()

        # Create task interval variables
        tasks = [interval_var(name=f'T{i + 1}', size=DURATIONS[i]) for i in range(NB_TASKS)]

        # Add precedence constraints
        mdl.add(end_before_start(tasks[t], tasks[s - 1]) for t in range(NB_TASKS) for s in SUCCESSORS[t])

        # Constrain capacity of resources
        mdl.add(
            sum(pulse(tasks[t], DEMANDS[t][r]) for t in range(NB_TASKS) if DEMANDS[t][r] > 0) <= CAPACITIES[r] for r in
            range(NB_RESOURCES))

        # Create makespan variable
        makespan = max(end_of(t) for t in tasks)

        # Always minimize the makespan
        mdl.add(minimize(makespan))

        # If an optimal bound is provided, add it as an upper bound constraint
        if OPTIMAL_BOUND is not None:
            mdl.add(makespan <= OPTIMAL_BOUND)

        # Solve model with fixed time limit
        print(f"Solving model for {data_file.name} with {TIME_PER_INSTANCE} seconds limit...")
        res = mdl.solve(TimeLimit=TIME_PER_INSTANCE, LogVerbosity="Quiet")

        solve_time = time.time() - start_time

        if res:
            # Solution found - check status
            solve_status = res.get_solve_status()

            # Get the objective value (makespan)
            objective_values = res.get_objective_values()
            objective_value = objective_values[0] if objective_values else None

            if solve_status == "Optimal":
                status = "optimal"
                print(f"Optimal solution found for {data_file.name}")
            else:
                status = "feasible"
                print(f"Feasible solution found for {data_file.name}")

            if objective_value is not None:
                print(f"Makespan = {objective_value}")
            else:
                # This shouldn't happen, but just in case
                print(f"Warning: Solution found but no objective value for {data_file.name}")
                objective_value = None
        else:
            # No solution found
            print(f"No solution found for {data_file.name}")
            objective_value = None
            status = "unknown"

        return (objective_value, status, solve_time)

    except Exception as e:
        solve_time = time.time() - start_time
        print(f"Error solving {data_file}: {str(e)}")
        import traceback
        traceback.print_exc()
        return (None, "error", solve_time)


def main():
    # Define directories
    data_dir = Path("Convert/data/j60")
    result_dir = Path("result")
    output_file = result_dir / "j60.csv"

    # Create result directory if it doesn't exist
    os.makedirs(result_dir, exist_ok=True)

    # Find all .data files in the data directory
    data_files = list(data_dir.glob("*.data"))
    if not data_files:
        print(f"Warning: No .data files found in {data_dir}")
        print(f"Current directory: {os.getcwd()}")
        print("Directory contents:")
        for item in os.listdir():
            print(f"  {item}")
        return

    print(f"Found {len(data_files)} .data files to process")
    print(f"Using 20 seconds time limit per instance")

    # Initialize CSV
    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header
        csv_writer.writerow(["file name", "Model constraint", "Status", "Solve time (second)"])

        # Process each file
        for i, data_file in enumerate(data_files, 1):
            # Only use the filename, not the path
            file_name = data_file.name
            print(f"\n[{i}/{len(data_files)}] Processing {file_name}...")

            try:
                # Run RCPSP solver with fixed time limit
                makespan, status, solve_time = solve_rcpsp(data_file)

                # Format the results for CSV
                makespan_str = str(makespan) if makespan is not None else "N/A"

                # Write results to CSV
                csv_writer.writerow([
                    file_name,
                    makespan_str,
                    status,
                    f"{solve_time:.2f}"
                ])

                # Flush to disk so partial results are saved
                csvfile.flush()

                print(f"Results for {file_name}:")
                print(f"  Model constraint: {makespan_str}")
                print(f"  Status: {status}")
                print(f"  Solve time: {solve_time:.2f}s")

            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")
                import traceback
                traceback.print_exc()

                # Write error to CSV
                csv_writer.writerow([
                    file_name,
                    "Error",
                    "error",
                    "0.00"
                ])
                csvfile.flush()

    print(f"\nAll done! Results written to {output_file}")


if __name__ == "__main__":
    main()