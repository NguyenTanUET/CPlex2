#!/usr/bin/env python3
import re
import csv
import sys
from pathlib import Path


def parse_sm(sm_path: Path):
    """
    Đọc file .sm, trả về:
      - base_id: ví dụ 'j301_5'
      - n_jobs, m_res
      - caps: list[int] công suất tài nguyên
      - jobs: list of dict { 'dur':int, 'demands':List[int], 'succ':List[int] }
    """
    text = sm_path.read_text().splitlines()

    # 1) Lấy base_id
    base_id = None
    for line in text:
        if line.strip().lower().startswith('file with basedata'):
            # ví dụ: "file with basedata : j301_5.bas"
            base = line.split(':', 1)[1].strip()
            base_id = Path(base).stem  # -> 'j301_5'
            break

    # Nếu không tìm thấy base_id, sử dụng tên file
    if base_id is None:
        base_id = sm_path.stem

    # Đảm bảo base_id có dạng đúng (j301_5, không phải j30_17)
    if not re.match(r'j30\d+_\d+', base_id):
        # Nếu định dạng là j30_17 thì chuyển thành j3017_1 (giả sử file thứ 1)
        match = re.match(r'j(\d+)_(\d+)', base_id)
        if match:
            project_num = match.group(1)
            instance_num = match.group(2)
            if len(project_num) == 2:  # Nếu là j30_17
                base_id = f"j30{instance_num}_1"  # Chuyển thành j3017_1

    # Dự phòng: lấy trực tiếp từ tên file
    if sm_path.stem.startswith('j30') and '_' in sm_path.stem:
        base_id = sm_path.stem

    # 2) Đọc số lượng job (bao gồm 2 dummy) và số resource
    n_jobs = m_res = None
    for line in text:
        if 'jobs (incl. supersource/sink' in line:
            n_jobs = int(re.search(r':\s*(\d+)', line).group(1))
        if 'renewable' in line:
            m_res = int(re.search(r':\s*(\d+)', line).group(1))
        if n_jobs is not None and m_res is not None:
            break

    # 3) Đọc capacities (bỏ qua mọi dòng không phải chỉ chứa chữ số)
    caps = []
    for line in text:
        if "RESOURCEAVAILABILITIES:" in line:
            # Tìm đến phần resource availabilities
            continue
        toks = line.strip().split()
        # nếu đúng m_res số nguyên thì đây là dòng capacities
        if len(toks) == m_res and all(tok.isdigit() for tok in toks):
            caps = list(map(int, toks))
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

    return base_id, n_jobs, m_res, caps, jobs


def read_bounds(csv_path: Path):
    """
    Đọc file CSV dạng:
       filename.sm, lower_bound, upper_bound
    Trả về dict: { 'j301_5': (lb, ub), ... }
    """
    mapping = {}
    with csv_path.open() as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            fname = row[0].strip()

            # Xử lý nhiều dạng key để tăng khả năng match
            key1 = Path(fname).stem  # 'j301_5.sm' -> 'j301_5'
            key2 = key1.replace('_', '')  # 'j301_5' -> 'j3015'

            # Loại bỏ phần mở rộng .sm nếu có
            if key1.endswith('.sm'):
                key1 = key1[:-3]

            try:
                lb = int(row[1].strip())
                ub = int(row[2].strip())
            except ValueError:
                continue

            # Thêm tất cả các dạng key vào mapping
            mapping[key1] = (lb, ub)
            mapping[key2] = (lb, ub)

            # Thêm dạng không có phần mở rộng
            pure_name = fname.replace('.sm', '')
            mapping[pure_name] = (lb, ub)

    return mapping


def write_psplib(base_id, n, m, caps, jobs, bound, out_dir=Path('.')):
    """
    Ghi file rcpsp_{base_id}_{LB}_{UB}.data
    Đầu file có header: n m LB UB
    """
    lb, ub = bound if bound is not None else (0, 0)
    # Tạo tên file với thông tin bounds
    fname = f"rcpsp_{base_id}.data"
    out = out_dir / fname
    with out.open('w') as f:
        # Format header có thêm LB và UB: n_jobs m_res lb ub
        f.write(f"{n} {m} {lb} {ub}\n")

        # Format line 2 giống rcpsp_default.data: capacities
        f.write(" ".join(map(str, caps)) + "\n")

        # Thông tin từng job (giống định dạng rcpsp_default.data)
        for job in jobs:
            d = job['dur']
            dem = job['demands']
            succ = job['succ']

            # Format mỗi dòng: duration demands successors
            line = f"{d}"
            for res_demand in dem:
                line += f" {res_demand}"
            line += f" {len(succ)}"
            for s in succ:
                line += f" {s}"
            f.write(line + "\n")

    print(f"-> Written {out} with bounds {lb} {ub}")

    # Thông tin từng job (giống định dạng rcpsp_default.data)
    for job in jobs:
        d = job['dur']
        dem = job['demands']
        succ = job['succ']

        # Format mỗi dòng: duration demands successors
        line = f"{d}"
        for res_demand in dem:
            line += f" {res_demand}"
        line += f" {len(succ)}"
        for s in succ:
            line += f" {s}"
        f.write(line + "\n")

def main():
    args = sys.argv[1:]
    if len(args) == 0:
        # batch mode
        sm_list = list(Path('.').glob('*.sm'))
        csv_path = Path('bounds/bound_j30.sm.csv')
    elif len(args) == 2:
        sm_list = [Path(args[0])]
        csv_path = Path(args[1])
    else:
        print("Usage:")
        print(" # batch convert all *.sm:")
        print("    python convert.py")
        print("  # convert single file:")
        print("    python convert.py <file.sm> <bounds.csv>")
        sys.exit(1)

    if not csv_path.exists():
        print(f"ERROR: không tìm thấy bound file {csv_path}")
        sys.exit(1)

    bounds = read_bounds(csv_path)

    # Debug: In danh sách tất cả các key trong bounds để kiểm tra
    print("Available bound keys:")
    for key in sorted(bounds.keys()):
        print(f"  - {key}: {bounds[key]}")

    for sm_file in sm_list:
        if not sm_file.exists():
            print(f"SKIP (not found): {sm_file}")
            continue

        try:
            base_id, n_jobs, m_res, caps, jobs = parse_sm(sm_file)

            # Thử các biến thể khác nhau của base_id để tìm trong bounds
            bound = None
            variants = [
                base_id,
                base_id.replace('_', ''),
                f"{base_id}.sm",
            ]

            for variant in variants:
                if variant in bounds:
                    bound = bounds[variant]
                    break

            # Tìm kiếm theo pattern
            if bound is None:
                # Xử lý trường hợp đặc biệt cho j301_5
                if base_id == 'j301_5':
                    for key in bounds:
                        if '301_5' in key or '3015' in key:
                            bound = bounds[key]
                            break

            if bound:
                print(f"Converting {sm_file.name} (base_id = {base_id}) with bounds: {bound}")
            else:
                print(f"WARNING: No bounds found for {sm_file.name} (base_id = {base_id})")
                # Tìm trong csv file trực tiếp
                with csv_path.open() as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) < 3:
                            continue
                        fname = row[0].strip()
                        if base_id in fname or fname in base_id:
                            try:
                                lb = int(row[1].strip())
                                ub = int(row[2].strip())
                                bound = (lb, ub)
                                print(f"  Found matching bounds in CSV: {bound}")
                                break
                            except ValueError:
                                continue

                if not bound:
                    print("  Using default bounds (0, 0)")
                    bound = (0, 0)

            write_psplib(base_id, n_jobs, m_res, caps, jobs, bound)
        except Exception as e:
            print(f"ERROR processing {sm_file}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()