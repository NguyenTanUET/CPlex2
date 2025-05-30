#!/usr/bin/env python3
"""
rcpsp_sm_processor.py

Script để xử lý file RCPSP single-mode (.sm) và file bound (.csv) cho tập J30,
bao gồm parsing scenario, bound và xây dựng mô hình CP-Solver.

Usage:
    python rcpsp_sm_processor.py <scenario.sm> <bounds.csv>

Ví dụ:
    python rcpsp_sm_processor.py test_2022.sm bound_j30.sm.csv
"""
import sys
import os
import csv
import re
from docplex.cp.model import CpoModel

#-----------------------------------------------------------------------------#
# Parser file .data (PSPLIB format, tương tự .bas)                            #
#-----------------------------------------------------------------------------#
def parse_data(fname):
    """
    Đọc file PSPLIB .data hoặc .bas:
      - Dòng 1: NB_TASKS NB_RESOURCES
      - Dòng 2: capacities
      - Tiếp: mỗi dòng tác vụ gồm [duration, demands..., S, successors...]
    """
    with open(fname, 'r') as f:
        NB_TASKS, NB_RESOURCES = map(int, f.readline().split())
        CAPACITIES = list(map(int, f.readline().split()))
        TASKS = [list(map(int, f.readline().split())) for _ in range(NB_TASKS)]
    DURATIONS  = [t[0] for t in TASKS]
    DEMANDS    = [t[1:1+NB_RESOURCES] for t in TASKS]
    SUCCESSORS = [t[1+NB_RESOURCES+1:] for t in TASKS]
    return NB_TASKS, NB_RESOURCES, CAPACITIES, DURATIONS, DEMANDS, SUCCESSORS

#-----------------------------------------------------------------------------#
# Parser file .sm (scenario single-mode PSPLIB)                               #
#-----------------------------------------------------------------------------#
def parse_sm(sm_path):
    """
    Đọc file .sm và trích xuất:
      base_file: tên file .bas tham chiếu
      nb_tasks: tổng số jobs
      nb_resources: số tài nguyên renewable
      durations, demands: từ section REQUESTS/DURATIONS
      capacities: từ section RESOURCE AVAILABILITIES
    """
    lines = []
    with open(sm_path) as f:
        for ln in f:
            if ln.strip() and not ln.startswith('*'):
                lines.append(ln.strip())
    data = {}
    # base data file (.bas)
    for ln in lines:
        if ln.lower().startswith('file with basedata'):
            parts = ln.split(':', 1)
            data['base_file'] = parts[1].strip() if len(parts) > 1 else None
            break
    # số tác vụ
    for ln in lines:
        m = re.match(r'jobs.*:\s*(\d+)', ln, re.IGNORECASE)
        if m:
            data['nb_tasks'] = int(m.group(1))
            break
    # số tài nguyên renewable
    for ln in lines:
        m = re.match(r'- renewable.*:\s*(\d+)', ln, re.IGNORECASE)
        if m:
            data['nb_resources'] = int(m.group(1))
            break
    # REQUESTS/DURATIONS
    try:
        idx = lines.index('REQUESTS/DURATIONS:')
    except ValueError:
        raise RuntimeError("Không tìm thấy section REQUESTS/DURATIONS trong .sm file")
    start = idx + 3
    end = start + data['nb_tasks']
    dur = []
    dem = []
    for ent in lines[start:end]:
        parts = ent.split()
        dur.append(int(parts[2]))
        dem.append(list(map(int, parts[3:3 + data['nb_resources']])))
    data['durations'] = dur
    data['demands']   = dem
    # RESOURCE AVAILABILITIES
    try:
        idx2 = lines.index('RESOURCE AVAILABILITIES:')
    except ValueError:
        raise RuntimeError("Không tìm thấy section RESOURCE AVAILABILITIES trong .sm file")
    data['capacities'] = list(map(int, lines[idx2 + 2].split()))
    return data

#-----------------------------------------------------------------------------#
# Parser file bounds (.csv)                                                      #
#-----------------------------------------------------------------------------#
def parse_bounds(csv_path):
    """
    Đọc CSV format: instance_name, LB, UB
    Trả về dict: {instance: (LB, UB)}
    """
    bd = {}
    with open(csv_path) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3: continue
            inst = row[0].strip()
            try:
                lb, ub = int(row[1]), int(row[2])
            except ValueError:
                continue
            bd[inst] = (lb, ub)
    return bd

#-----------------------------------------------------------------------------#
# Main: parse, build model và solve                                        #
#-----------------------------------------------------------------------------#
def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    sm_file  = sys.argv[1]
    csv_file = sys.argv[2]
    if not os.path.isfile(sm_file) or not os.path.isfile(csv_file):
        print(f"File không tồn tại: {sm_file} hoặc {csv_file}")
        sys.exit(1)

    # parse .sm và bounds
    sm_info = parse_sm(sm_file)
    bounds  = parse_bounds(csv_file)
    scenario = os.path.basename(sm_file)
    lb_ub    = bounds.get(scenario, (None, None))

    # parse .bas nếu có
    base_file = sm_info.get('base_file')
    SUCCESSORS = []
    if base_file:
        base_path = os.path.join(os.path.dirname(sm_file), base_file)
        if os.path.isfile(base_path):
            try:
                _, _, _, _, _, SUCCESSORS = parse_data(base_path)
            except Exception as e:
                print(f"Warning: lỗi khi đọc base data '{base_file}': {e}")
                SUCCESSORS = [[] for _ in range(sm_info['nb_tasks'])]
        else:
            print(f"Warning: không tìm thấy base data file '{base_file}', precedence sẽ bị bỏ qua.")
            SUCCESSORS = [[] for _ in range(sm_info['nb_tasks'])]
    else:
        SUCCESSORS = [[] for _ in range(sm_info['nb_tasks'])]

    # gán dữ liệu cho model
    NB_TASKS     = sm_info['nb_tasks']
    NB_RESOURCES = sm_info['nb_resources']
    CAPACITIES   = sm_info['capacities']
    DURATIONS    = sm_info['durations']
    DEMANDS      = sm_info['demands']

    # Build CP-Solver model
    mdl = CpoModel()
    tasks = [mdl.interval_var(name=f"T{i+1}", size=DURATIONS[i]) for i in range(NB_TASKS)]
    # precedence
    for i in range(NB_TASKS):
        for s in SUCCESSORS[i]:
            mdl.add(mdl.end_before_start(tasks[i], tasks[s-1]))
    # resource capacity
    for r in range(NB_RESOURCES):
        mdl.add(mdl.sum(mdl.pulse(tasks[t], DEMANDS[t][r])
                        for t in range(NB_TASKS)
                        if DEMANDS[t][r] > 0)
                <= CAPACITIES[r])
    # objective: minimize makespan
    makespan = mdl.max(mdl.end_of(t) for t in tasks)
    mdl.add(mdl.minimize(makespan))

    print("Solving scenario", scenario, "with bounds", lb_ub)
    sol = mdl.solve(TimeLimit=30)
    if sol:
        sol.print_solution()
        print(f"Makespan = {sol.get_objective_values()[0]}")
    else:
        print("No solution found")

if __name__ == '__main__':
    main()
