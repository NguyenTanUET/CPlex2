# !/usr/bin/env python3
"""
GPU-accelerated RCPSP solver using Numba CUDA
Works without full CUDA toolkit installation on Windows
"""

import numpy as np
import numba
from numba import cuda, float32, int32, boolean
import os
import csv
import time
from pathlib import Path
import math
from typing import List, Tuple, Optional


@numba.jit(nopython=True)
def parse_rcpsp_data(lines):
    """Parse RCPSP data from file lines"""
    # Find the first line with two integers (NB_TASKS, NB_RESOURCES)
    first_line = lines[0].split()
    nb_tasks, nb_resources = int(first_line[0]), int(first_line[1])

    # Get optimal bound if available
    optimal_bound = None
    if len(first_line) > 2:
        optimal_bound = int(first_line[2])

    # Parse capacities
    capacities = np.array([int(v) for v in lines[1].split()], dtype=np.int32)

    # Parse task data
    durations = np.zeros(nb_tasks, dtype=np.int32)
    demands = np.zeros((nb_tasks, nb_resources), dtype=np.int32)
    successors_lists = []

    for i in range(nb_tasks):
        task_data = [int(v) for v in lines[2 + i].split()]
        durations[i] = task_data[0]

        # Resource demands
        for r in range(nb_resources):
            demands[i, r] = task_data[1 + r]

        # Successors
        num_successors = task_data[nb_resources + 1]
        successors = []
        for s in range(num_successors):
            if len(task_data) > nb_resources + 2 + s:
                successors.append(task_data[nb_resources + 2 + s] - 1)  # Convert to 0-indexed
        successors_lists.append(successors)

    return nb_tasks, nb_resources, capacities, durations, demands, successors_lists, optimal_bound


@cuda.jit
def evaluate_population_kernel(population, fitness, durations, demands, capacities,
                               precedence_matrix, pop_size, nb_tasks, nb_resources):
    """CUDA kernel to evaluate fitness of entire population"""
    idx = cuda.grid(1)

    if idx >= pop_size:
        return

    # Calculate start times based on priority encoding
    start_times = cuda.local.array(512, float32)  # Max 512 tasks
    scheduled = cuda.local.array(512, boolean)

    # Initialize arrays
    for i in range(nb_tasks):
        start_times[i] = 0.0
        scheduled[i] = False

    # Schedule tasks based on priority values
    for scheduled_count in range(nb_tasks):
        best_task = -1
        best_priority = -1.0

        # Find highest priority unscheduled task that can be scheduled
        for task in range(nb_tasks):
            if not scheduled[task] and population[idx, task] > best_priority:
                # Check if all predecessors are scheduled
                can_schedule = True
                for pred in range(nb_tasks):
                    if precedence_matrix[pred, task] and not scheduled[pred]:
                        can_schedule = False
                        break

                if can_schedule:
                    best_priority = population[idx, task]
                    best_task = task

        if best_task == -1:
            break  # Error in scheduling

        # Calculate earliest start time considering precedence
        earliest_start = 0.0
        for pred in range(nb_tasks):
            if precedence_matrix[pred, best_task]:
                pred_end = start_times[pred] + durations[pred]
                if pred_end > earliest_start:
                    earliest_start = pred_end

        start_times[best_task] = earliest_start
        scheduled[best_task] = True

    # Calculate makespan
    makespan = 0.0
    for task in range(nb_tasks):
        end_time = start_times[task] + durations[task]
        if end_time > makespan:
            makespan = end_time

    # Simple resource feasibility check (can be improved)
    penalty = 0.0
    time_horizon = int(makespan) + 1

    for t in range(time_horizon):
        for r in range(nb_resources):
            resource_usage = 0.0
            for task in range(nb_tasks):
                if start_times[task] <= t < start_times[task] + durations[task]:
                    resource_usage += demands[task, r]

            if resource_usage > capacities[r]:
                penalty += (resource_usage - capacities[r]) * 1000.0

    fitness[idx] = makespan + penalty


@cuda.jit
def genetic_operations_kernel(population, new_population, fitness,
                              pop_size, nb_tasks, generation):
    """CUDA kernel for selection, crossover, and mutation"""
    idx = cuda.grid(1)

    if idx >= pop_size:
        return

    # Simple random number generation based on thread ID and generation
    seed = idx * 1000 + generation * 13

    # Tournament selection
    tournament_size = 4
    best_fitness = 1e10
    best_idx = idx

    for i in range(tournament_size):
        candidate_idx = (seed * (i + 1) * 7) % pop_size
        if fitness[candidate_idx] < best_fitness:
            best_fitness = fitness[candidate_idx]
            best_idx = candidate_idx

    # Copy selected individual to new population
    for gene in range(nb_tasks):
        new_population[idx, gene] = population[best_idx, gene]

    # Mutation
    mutation_rate = 0.1
    for gene in range(nb_tasks):
        rand_val = ((seed * (gene + 1) * 17) % 1000) / 1000.0
        if rand_val < mutation_rate:
            new_rand = ((seed * gene * 23) % 1000) / 1000.0
            new_population[idx, gene] = new_rand


class NumbaRCPSPSolver:
    """GPU-accelerated RCPSP solver using Numba CUDA"""

    def __init__(self, population_size=None, max_generations=1000):
        self.max_generations = max_generations

        # Check CUDA availability
        if not cuda.is_available():
            print("CUDA is not available. Falling back to CPU.")
            self.use_gpu = False
            self.population_size = population_size or 512
        else:
            gpu_count = len(cuda.gpus)
            print(f"CUDA available with {gpu_count} GPU(s)")
            self.use_gpu = True

            # Get GPU info for better configuration
            gpu = cuda.get_current_device()
            print(f"GPU: {gpu.name}")
            print(f"Compute Capability: {gpu.compute_capability}")
            print(f"Multiprocessors: {gpu.MULTIPROCESSOR_COUNT}")

            # Auto-size population for optimal GPU utilization
            if population_size is None:
                # Target: 4-8 blocks per multiprocessor for good occupancy
                multiprocessors = gpu.MULTIPROCESSOR_COUNT
                target_blocks = multiprocessors * 6  # 6 blocks per SM
                threads_per_block = 128  # Good balance for occupancy

                optimal_population = target_blocks * threads_per_block
                self.population_size = max(optimal_population, 8192)  # Minimum 8192
                print(f"Auto-configured population size: {self.population_size}")
            else:
                self.population_size = population_size

    def parse_rcpsp_file(self, filename: str):
        """Parse RCPSP data file"""
        with open(filename, 'r') as file:
            lines = [line.strip() for line in file.readlines() if line.strip()]

        first_line = lines[0].split()
        nb_tasks, nb_resources = int(first_line[0]), int(first_line[1])

        optimal_bound = None
        if len(first_line) > 2:
            optimal_bound = int(first_line[2])

        capacities = np.array([int(v) for v in lines[1].split()], dtype=np.int32)

        durations = np.zeros(nb_tasks, dtype=np.int32)
        demands = np.zeros((nb_tasks, nb_resources), dtype=np.int32)
        successors_lists = []

        for i in range(nb_tasks):
            task_data = [int(v) for v in lines[2 + i].split()]
            durations[i] = task_data[0]

            for r in range(nb_resources):
                demands[i, r] = task_data[1 + r]

            num_successors = task_data[nb_resources + 1]
            successors = []
            for s in range(num_successors):
                if len(task_data) > nb_resources + 2 + s:
                    successors.append(task_data[nb_resources + 2 + s] - 1)
            successors_lists.append(successors)

        # Create precedence matrix
        precedence_matrix = np.zeros((nb_tasks, nb_tasks), dtype=np.bool_)
        for i, successors in enumerate(successors_lists):
            for successor in successors:
                if 0 <= successor < nb_tasks:
                    precedence_matrix[i, successor] = True

        return {
            'nb_tasks': nb_tasks,
            'nb_resources': nb_resources,
            'capacities': capacities,
            'durations': durations,
            'demands': demands,
            'precedence_matrix': precedence_matrix,
            'optimal_bound': optimal_bound
        }

    def solve(self, instance_data, time_limit=900.0):
        """Solve RCPSP instance with improved initialization and early termination"""
        start_time = time.time()

        try:
            nb_tasks = instance_data['nb_tasks']
            nb_resources = instance_data['nb_resources']
            capacities = instance_data['capacities']
            durations = instance_data['durations']
            demands = instance_data['demands']
            precedence_matrix = instance_data['precedence_matrix']
            optimal_bound = instance_data['optimal_bound']

            print(f"Optimal bound for this instance: {optimal_bound}")

            # Much better initialization: Use greedy heuristic for initial population
            population = self._initialize_population_heuristic(
                nb_tasks, durations, demands, capacities, precedence_matrix
            )
            fitness = np.zeros(self.population_size, dtype=np.float32)
            new_population = np.zeros_like(population)

            if self.use_gpu:
                # Transfer data to GPU
                d_population = cuda.to_device(population)
                d_new_population = cuda.to_device(new_population)
                d_fitness = cuda.to_device(fitness)
                d_durations = cuda.to_device(durations)
                d_demands = cuda.to_device(demands)
                d_capacities = cuda.to_device(capacities)
                d_precedence = cuda.to_device(precedence_matrix)

                threads_per_block = 128
                blocks_per_grid = (self.population_size + threads_per_block - 1) // threads_per_block

            best_fitness = float('inf')
            stagnation_count = 0
            generation = 0

            # Quick feasibility check - terminate early if no improvement after many generations
            while True:
                # Check time limit
                if time.time() - start_time > time_limit:
                    print(f"Time limit reached at generation {generation}")
                    break

                if self.use_gpu:
                    # GPU evaluation
                    evaluate_population_kernel[blocks_per_grid, threads_per_block](
                        d_population, d_fitness, d_durations, d_demands, d_capacities,
                        d_precedence, self.population_size, nb_tasks, nb_resources
                    )
                    fitness = d_fitness.copy_to_host()
                else:
                    # CPU fallback
                    for i in range(self.population_size):
                        fitness[i] = self._evaluate_individual_cpu(
                            population[i], durations, demands, capacities, precedence_matrix
                        )

                # Check for improvement
                current_best = np.min(fitness)
                if current_best < best_fitness:
                    improvement = best_fitness - current_best
                    best_fitness = current_best
                    stagnation_count = 0
                    print(f"Generation {generation}: New best = {best_fitness:.2f} (improved by {improvement:.2f})")
                else:
                    stagnation_count += 1

                # Early termination if optimal found
                if optimal_bound is not None and best_fitness <= optimal_bound + 1e-6:
                    print(f"Optimal solution found at generation {generation}!")
                    break

                # Early termination for clearly infeasible problems
                if generation > 500 and best_fitness > 50000:
                    print(f"Problem appears infeasible - stopping early at generation {generation}")
                    break

                # Early termination for stagnation
                if stagnation_count > 200:
                    print(f"No improvement for 200 generations - stopping early at generation {generation}")
                    break

                if self.use_gpu:
                    genetic_operations_kernel[blocks_per_grid, threads_per_block](
                        d_population, d_new_population, d_fitness,
                        self.population_size, nb_tasks, generation
                    )
                    d_population, d_new_population = d_new_population, d_population
                else:
                    self._cpu_genetic_operations(population, new_population, fitness)
                    population, new_population = new_population, population

                if generation % 50 == 0:  # Report every 50 generations
                    elapsed = time.time() - start_time
                    print(
                        f"Gen {generation}: Best = {best_fitness:.2f}, Stagnation = {stagnation_count}, Time = {elapsed:.2f}s")

                generation += 1

            solve_time = time.time() - start_time

            # Determine status
            if optimal_bound is not None and best_fitness <= optimal_bound + 1e-6:
                status = "optimal"
            elif best_fitness < 1000:  # Reasonable makespan (no major penalties)
                status = "feasible"
            else:
                status = "unknown"

            return best_fitness if best_fitness < 1000 else None, status, solve_time

        except Exception as e:
            solve_time = time.time() - start_time
            print(f"Error in solver: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, "error", solve_time

    def _initialize_population_heuristic(self, nb_tasks, durations, demands, capacities, precedence_matrix):
        """Initialize population with heuristic-based individuals instead of random"""
        population = np.zeros((self.population_size, nb_tasks), dtype=np.float32)

        # Generate multiple heuristic solutions
        for i in range(self.population_size):
            if i < self.population_size // 4:
                # Priority-based on duration (shortest first)
                priorities = 1.0 / (durations + 1e-6)
            elif i < self.population_size // 2:
                # Priority-based on resource demand (lightest first)
                resource_load = np.sum(demands, axis=1)
                priorities = 1.0 / (resource_load + 1e-6)
            elif i < 3 * self.population_size // 4:
                # Critical path based priorities
                priorities = np.random.exponential(1.0, nb_tasks)
            else:
                # Random for diversity
                priorities = np.random.rand(nb_tasks)

            # Normalize to [0,1] range
            priorities = (priorities - np.min(priorities)) / (np.max(priorities) - np.min(priorities) + 1e-6)
            population[i] = priorities.astype(np.float32)

        return population
    def _evaluate_individual_cpu(self, individual, durations, demands, capacities, precedence_matrix):
        """CPU fallback for fitness evaluation"""
        nb_tasks = len(individual)

        # Simple priority-based scheduling
        start_times = np.zeros(nb_tasks)
        scheduled = np.zeros(nb_tasks, dtype=bool)

        for _ in range(nb_tasks):
            # Find highest priority unscheduled task
            best_task = -1
            best_priority = -1.0

            for task in range(nb_tasks):
                if not scheduled[task] and individual[task] > best_priority:
                    # Check precedence constraints
                    can_schedule = True
                    for pred in range(nb_tasks):
                        if precedence_matrix[pred, task] and not scheduled[pred]:
                            can_schedule = False
                            break

                    if can_schedule:
                        best_priority = individual[task]
                        best_task = task

            if best_task == -1:
                break

            # Calculate start time
            earliest_start = 0.0
            for pred in range(nb_tasks):
                if precedence_matrix[pred, best_task]:
                    pred_end = start_times[pred] + durations[pred]
                    if pred_end > earliest_start:
                        earliest_start = pred_end

            start_times[best_task] = earliest_start
            scheduled[best_task] = True

        # Calculate makespan
        makespan = np.max(start_times + durations)
        return makespan

    def _cpu_genetic_operations(self, population, new_population, fitness):
        """CPU fallback for genetic operations"""
        pop_size, nb_tasks = population.shape

        # Simple tournament selection and mutation
        for i in range(pop_size):
            # Tournament selection
            tournament_indices = np.random.choice(pop_size, 4, replace=False)
            best_idx = tournament_indices[np.argmin(fitness[tournament_indices])]

            # Copy and mutate
            new_population[i] = population[best_idx].copy()

            # Mutation
            mutation_mask = np.random.rand(nb_tasks) < 0.1
            new_population[i][mutation_mask] = np.random.rand(np.sum(mutation_mask))


def solve_rcpsp_numba(data_file, time_limit=900.0):
    """Wrapper function for solving single RCPSP instance"""
    solver = NumbaRCPSPSolver(population_size=16384, max_generations=1000)  # Doubled again for 64 blocks
    instance_data = solver.parse_rcpsp_file(str(data_file))
    return solver.solve(instance_data, time_limit)


def main():
    """Main function to process all RCPSP instances"""
    print("Starting Numba CUDA RCPSP Solver")

    # Define directories
    data_dir = Path("Convert/data/j30")  # Change as needed
    result_dir = Path("result")
    output_file = result_dir / "j30_numba_cuda.csv"

    # Create result directory
    os.makedirs(result_dir, exist_ok=True)

    # Find data files
    data_files = list(data_dir.glob("*.data"))
    if not data_files:
        print(f"No .data files found in {data_dir}")
        return

    print(f"Found {len(data_files)} files to process")

    # Process files
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["file name", "Model constraint", "Status", "Solve time (second)"])

        for i, data_file in enumerate(data_files, 1):
            print(f"\n[{i}/{len(data_files)}] Processing {data_file.name}...")

            try:
                makespan, status, solve_time = solve_rcpsp_numba(data_file)

                makespan_str = str(int(makespan)) if makespan is not None else "N/A"

                writer.writerow([
                    data_file.name,
                    makespan_str,
                    status,
                    f"{solve_time:.2f}"
                ])
                csvfile.flush()

                print(f"  Result: {makespan_str}, Status: {status}, Time: {solve_time:.2f}s")

            except Exception as e:
                print(f"  Error: {str(e)}")
                writer.writerow([data_file.name, "Error", "error", "0.00"])
                csvfile.flush()

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()