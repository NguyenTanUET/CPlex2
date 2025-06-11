#!/usr/bin/env python3
import re
import csv
import sys
import os
from pathlib import Path


def parse_sm(sm_path: Path):
    """
    Đọc file .sm, trả về:
      - base_id: ví dụ 'j6010_1' (lấy từ tên file thay vì nội dung)
      - n_jobs, m_res
      - caps: list[int] công suất tài nguyên
      - jobs: list of dict { 'dur':int, 'demands':List[int], 'succ':List[int] }
    """
    text = sm_path.read_text().splitlines()

    # 1) Lấy base_id từ tên file thay vì nội dung file
    base_id = sm_path.stem  # j6010_1.sm -> j6010_1

    # 2) Đọc số lượng job (bao gồm 2 dummy) và số resource
    n_jobs = m_res = None
    for line in text:
        if 'jobs (incl. supersource/sink' in line:
            n_jobs = int(re.search(r':\s*(\d+)', line).group(1))
        if 'renewable' in line:
            m_res = int(re.search(r':\s*(\d+)', line).group(1))
        if n_jobs is not None and m_res is not None:
            break

    # 3) Đọc capacities - SỬA LỖI: Đọc đúng phần RESOURCEAVAILABILITIES
    caps = []
    found_resource_section = False
    for line in text:
        if "RESOURCEAVAILABILITIES:" in line:
            found_resource_section = True
            continue

        if found_resource_section:
            toks = line.strip().split()
            # Kiểm tra xem có phải dòng capacities không
            if len(toks) == m_res and all(tok.isdigit() for tok in toks):
                caps = list(map(int, toks))  # ← SỬA: map(int, toks)
                break
            # Nếu gặp dòng không phải số hoặc section mới thì dừng
            if toks and not all(tok.isdigit() for tok in toks):
                break

    if not caps:
        raise RuntimeError(f"Không đọc được capacities (m_res={m_res}) trong {sm_path}")

    # 4) Đọc precedence relations
    succ = {}
    in_prec = False
    for line in text:
        if line.strip().startswith('PRECEDENCE RELATIONS:'):
            in_prec = True
            continue
        if in_prec:
            if not line.strip() or line.startswith('REQUESTS/DURATIONS:'):
                break
            parts = line.split()
            if not parts[0].isdigit():
                continue
            job = int(parts[0])
            cnt = int(parts[2])
            succ[job] = [int(x) for x in parts[3:3 + cnt]]

    # 5) Đọc durations & demands
    demands = {}
    in_req = False
    for line in text:
        if line.strip().startswith('REQUESTS/DURATIONS:'):
            in_req = True
            continue
        if in_req:
            if not line.strip() or line.startswith('*****'):
                break
            parts = line.split()
            if not parts[0].isdigit():
                continue
            job = int(parts[0])
            dur = int(parts[2])
            dem = [int(x) for x in parts[3:3 + m_res]]
            demands[job] = (dur, dem)

    # 6) Build jobs list in order 1..n_jobs
    jobs = []
    for i in range(1, n_jobs + 1):
        if i not in demands:
            raise RuntimeError(f"Không có thông tin job {i} trong {sm_path}")
        d, dem = demands[i]
        s = succ.get(i, [])
        jobs.append({'dur': d, 'demands': dem, 'succ': s})

    print(f"Capacities read: {caps}")
    print(f"Number of resources: {m_res}")
    print(f"Expected capacity length: {m_res}")

    return base_id, n_jobs, m_res, caps, jobs


def read_bounds(csv_path: Path):
    """
    Đọc file CSV dạng:
       filename.sm, lower_bound, upper_bound
    Trả về dict: { 'j6010_1': (lb, ub), ... }
    """
    mapping = {}
    with csv_path.open() as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            fname = row[0].strip()

            # Lấy base_id từ tên file
            base_id = Path(fname).stem  # j6010_1.sm -> j6010_1

            try:
                lb = int(row[1].strip())
                ub = int(row[2].strip())
            except ValueError:
                continue

            # Lưu với key chính xác
            mapping[base_id] = (lb, ub)

            # Lưu thêm với key có đuôi .sm để tăng khả năng match
            mapping[fname] = (lb, ub)

    return mapping


def write_psplib(base_id, n, m, caps, jobs, bound, out_dir=Path('.')):
    """
    Ghi file rcpsp_{base_id}.data với format ĐÚNG
    """
    if bound is not None:
        lb, ub = bound
    else:
        lb, ub = None, None

    fname = f"rcpsp_{base_id}.data"
    out = out_dir / fname

    with out.open('w') as f:
        # DEBUG: In ra để kiểm tra
        print(f"DEBUG: Writing {base_id} - n={n}, m={m}, caps={caps}, bound={bound}")

        # Header - CHÍNH XÁC như file gốc
        if lb is not None and ub is not None:
            f.write(f"{n} {m} {lb} {ub}\n")
        elif lb is not None:
            f.write(f"{n} {m} {lb}\n")  # Chỉ có LB
        else:
            f.write(f"{n} {m}\n")  # Không có bound

        # Capacities - ĐÚNG THỨ TỰ
        f.write(" ".join(map(str, caps)) + "\n")

        # Task data
        for job in jobs:
            d = job['dur']
            dem = job['demands']
            succ = job['succ']

            line = f"{d}"
            for res_demand in dem:
                line += f" {res_demand}"
            line += f" {len(succ)}"
            for s in succ:
                line += f" {s}"
            f.write(line + "\n")

    print(f"-> Written {out} with bounds LB:{lb} UB:{ub}")


def main():
    args = sys.argv[1:]

    # Xác định dataset và thư mục tương ứng
    if len(args) == 1:
        dataset = args[0].lower()
        if dataset not in ['j30', 'j60', 'j90', 'j120']:
            print("Usage: python convert.py [j30|j60|j90|j120]")
            print("Example: python convert.py j60")
            sys.exit(1)

        # Thiết lập đường dẫn dựa trên dataset
        raw_data_dir = Path(f'raw_data/{dataset}')
        bounds_dir = Path('bounds')
        csv_path = bounds_dir / f'bound_{dataset}.sm.csv'
        output_dir = Path(f'data/{dataset}')

    else:
        print("Usage: python convert.py [j30|j60|j90|j120]")
        print("Example: python convert.py j60")
        sys.exit(1)

    # Kiểm tra thư mục raw_data
    if not raw_data_dir.exists():
        print(f"ERROR: Thư mục {raw_data_dir} không tồn tại")
        sys.exit(1)

    # Kiểm tra file bounds
    if not csv_path.exists():
        print(f"ERROR: không tìm thấy bound file {csv_path}")
        sys.exit(1)

    # Tạo thư mục output nếu chưa có
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tìm tất cả file .sm trong thư mục raw_data
    sm_list = list(raw_data_dir.glob('*.sm'))
    if not sm_list:
        print(f"ERROR: Không tìm thấy file .sm nào trong {raw_data_dir}")
        sys.exit(1)

    bounds = read_bounds(csv_path)

    print(f"Dataset: {dataset}")
    print(f"Raw data directory: {raw_data_dir}")
    print(f"Loading bounds from: {csv_path}")
    print(f"Output directory: {output_dir}")
    print(f"Found {len(sm_list)} .sm files to convert")
    print("All files will include both lower and upper bounds")

    # Debug: In danh sách một số key trong bounds để kiểm tra
    print(f"\nSample bound keys (first 10):")
    for i, key in enumerate(sorted(bounds.keys())[:10]):
        print(f"  - {key}: {bounds[key]}")

    converted_count = 0

    for sm_file in sm_list:
        try:
            base_id, n_jobs, m_res, caps, jobs = parse_sm(sm_file)

            # Tìm bounds dựa trên base_id chính xác
            bound = bounds.get(base_id)

            if bound:
                lb, ub = bound
                print(f"Converting {sm_file.name} (base_id = {base_id}) with bounds: LB={lb}, UB={ub}")
            else:
                print(f"WARNING: No bounds found for {sm_file.name} (base_id = {base_id})")
                print("  Using default bounds LB=0, UB=0")
                bound = (0, 0)

            write_psplib(base_id, n_jobs, m_res, caps, jobs, bound, output_dir)
            converted_count += 1

        except Exception as e:
            print(f"ERROR processing {sm_file}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n=== Conversion Complete ===")
    print(f"Successfully converted {converted_count}/{len(sm_list)} files")
    print(f"Output files saved to: {output_dir}")


if __name__ == "__main__":
    main()