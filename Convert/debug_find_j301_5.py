#!/usr/bin/env python3
# Đây là script đơn giản để kiểm tra file CSV và tìm bound cho j301_5
import csv
from pathlib import Path


def debug_find_j301_5():
    csv_path = Path('bounds/bound_j30.sm.csv')

    if not csv_path.exists():
        print(f"ERROR: không tìm thấy file {csv_path}")
        return

    print("Searching for j301_5 in bound_j30.sm.csv...")

    with csv_path.open() as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue

            fname = row[0].strip()

            # Kiểm tra tên file có chứa 301_5 hoặc tương tự không
            if '301_5' in fname or '3015' in fname:
                print(f"Found potential match: {fname}")
                try:
                    lb = int(row[1].strip())
                    ub = int(row[2].strip())
                    print(f"  Bounds: LB={lb}, UB={ub}")
                except ValueError:
                    print(f"  Invalid bounds: {row[1:]}")


if __name__ == "__main__":
    debug_find_j301_5()