#!/usr/bin/env python3
"""
Script để thêm upper bounds vào các file .data hiện có
Format hiện tại: tasks renewable lower_bound
Format mới: tasks renewable lower_bound upper_bound
"""

import csv
import sys
from pathlib import Path


def read_bounds_csv(csv_path):
    """
    Đọc file CSV với format: filename.sm, lower_bound, upper_bound
    Trả về dict: {base_name: (lb, ub)}
    """
    bounds = {}

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue

            filename = row[0].strip()
            try:
                lb = int(row[1].strip())
                ub = int(row[2].strip())
            except ValueError:
                continue

            # Lấy base name từ filename
            # j601_1.sm -> j601_1
            base_name = Path(filename).stem
            if base_name.endswith('.sm'):
                base_name = base_name[:-3]

            bounds[base_name] = (lb, ub)

    return bounds


def process_data_file(data_file_path, bounds_dict):
    """
    Xử lý một file .data: đọc, thêm upper bound, và ghi lại
    """
    try:
        # Đọc file hiện tại
        with open(data_file_path, 'r') as f:
            lines = f.readlines()

        if not lines:
            print(f"Warning: File {data_file_path} trống")
            return False

        # Parse dòng đầu tiên
        first_line = lines[0].strip().split()
        if len(first_line) != 3:
            print(f"Warning: Dòng đầu của {data_file_path} không có đúng 3 số: {first_line}")
            return False

        tasks = int(first_line[0])
        renewable = int(first_line[1])
        lower_bound = int(first_line[2])

        # Tìm base name từ file path
        # rcpsp_j601_1.data -> j601_1
        file_stem = data_file_path.stem
        if file_stem.startswith('rcpsp_'):
            base_name = file_stem[6:]  # Bỏ "rcpsp_"
        else:
            base_name = file_stem

        # Tìm upper bound
        if base_name in bounds_dict:
            lb_from_csv, ub_from_csv = bounds_dict[base_name]

            # Kiểm tra xem lower bound có khớp không
            if lb_from_csv != lower_bound:
                print(f"Warning: Lower bound không khớp cho {base_name}: file={lower_bound}, csv={lb_from_csv}")

            upper_bound = ub_from_csv
        else:
            print(f"Warning: Không tìm thấy bounds cho {base_name} trong CSV")
            return False

        # Tạo dòng đầu mới
        new_first_line = f"{tasks} {renewable} {lower_bound} {upper_bound}\n"

        # Ghi lại file với upper bound
        with open(data_file_path, 'w') as f:
            f.write(new_first_line)
            f.writelines(lines[1:])  # Ghi lại các dòng còn lại

        print(
            f"Updated {data_file_path.name}: {tasks} {renewable} {lower_bound} -> {tasks} {renewable} {lower_bound} {upper_bound}")
        return True

    except Exception as e:
        print(f"Error processing {data_file_path}: {e}")
        return False


def main():
    # Parse arguments
    if len(sys.argv) == 2:
        dataset = sys.argv[1].lower()
        if dataset not in ['j30', 'j60', 'j90', 'j120']:
            print("Usage: python add_upper_bounds.py [j30|j60|j90|j120]")
            print("Example: python add_upper_bounds.py j60")
            sys.exit(1)
    else:
        print("Usage: python add_upper_bounds.py [j30|j60|j90|j120]")
        print("Example: python add_upper_bounds.py j60")
        sys.exit(1)

    # Đường dẫn - SỬA: bỏ "Convert/" vì đã đang trong thư mục Convert
    data_dir = Path(f'data/{dataset}')
    bounds_csv = Path(f'bounds/bound_{dataset}.sm.csv')

    # Kiểm tra thư mục và file
    if not data_dir.exists():
        print(f"Error: Thư mục {data_dir} không tồn tại")
        print(f"Current directory: {Path.cwd()}")
        print("Available directories:")
        for item in Path('.').iterdir():
            if item.is_dir():
                print(f"  {item}")
        sys.exit(1)

    if not bounds_csv.exists():
        print(f"Error: File bounds {bounds_csv} không tồn tại")
        sys.exit(1)

    # Đọc bounds từ CSV
    print(f"Đọc bounds từ {bounds_csv}...")
    bounds_dict = read_bounds_csv(bounds_csv)
    print(f"Đã đọc {len(bounds_dict)} bounds")

    # In một vài bounds để debug
    print("\nMột vài bounds từ CSV:")
    for i, (key, value) in enumerate(list(bounds_dict.items())[:5]):
        print(f"  {key}: {value}")

    # Tìm tất cả file .data
    data_files = list(data_dir.glob('*.data'))
    if not data_files:
        print(f"Không tìm thấy file .data nào trong {data_dir}")
        sys.exit(1)

    print(f"Tìm thấy {len(data_files)} file .data để xử lý")

    # Xử lý từng file
    success_count = 0
    total_count = len(data_files)

    for data_file in data_files:
        if process_data_file(data_file, bounds_dict):
            success_count += 1

    print(f"\n=== Hoàn thành ===")
    print(f"Đã xử lý thành công {success_count}/{total_count} files")

    # Hiển thị một vài file đã được update để kiểm tra
    if success_count > 0:
        print(f"\nKiểm tra vài file đầu tiên:")
        for i, data_file in enumerate(data_files[:3]):
            try:
                with open(data_file, 'r') as f:
                    first_line = f.readline().strip()
                print(f"  {data_file.name}: {first_line}")
            except:
                pass


if __name__ == "__main__":
    main()