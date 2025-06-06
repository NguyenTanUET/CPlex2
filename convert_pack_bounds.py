#!/usr/bin/env python3
import os
import sys
import csv
from pathlib import Path


def read_bounds_from_csv(csv_path):
    """
    Đọc file CSV chứa thông tin về lower bounds và upper bounds
    Trả về dictionary với key là tên file và value là tuple (lb, ub)
    """
    print(f"Đọc file bounds: {csv_path}")
    bounds_dict = {}

    try:
        # Đọc file CSV với dấu phân cách là dấu chấm phẩy
        with open(csv_path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            rows = list(reader)

        print(f"Đã đọc {len(rows)} dòng từ file bounds")

        # In ra 5 dòng đầu để debug
        print("5 dòng đầu của file CSV:")
        for i in range(min(5, len(rows))):
            print(f"  Dòng {i + 1}: {rows[i]}")

        # Bỏ qua 2 dòng đầu (header) và sử dụng dòng thứ 3 làm header
        header_row = 2  # Dòng thứ 3 (index 2)
        data_start_row = 3  # Dòng thứ 4 (index 3)

        if len(rows) <= header_row:
            print("File CSV không đủ dòng")
            return bounds_dict

        # Tìm các cột chứa thông tin LB và UB từ header
        header_line = rows[header_row][0]  # Lấy phần tử đầu tiên của dòng header

        # Phân tách header thành các cột riêng biệt
        header_columns = header_line.split(',')

        print(f"Header được phân tách thành {len(header_columns)} cột:")
        for i, col in enumerate(header_columns[:5]):  # Chỉ in 5 cột đầu
            print(f"  Cột {i}: '{col.strip()}'")

        # Tìm vị trí của các cột cần thiết
        name_col = 0  # Cột đầu tiên là tên file
        lb_col = 1  # Cột thứ 2 là lower bound
        ub_col = 2  # Cột thứ 3 là upper bound

        # Kiểm tra xem có đủ cột không
        if len(header_columns) < 3:
            print(f"Header không đủ cột (chỉ có {len(header_columns)} cột)")
            return bounds_dict

        # In thông tin các cột
        print(f"Dùng cột {name_col} ('{header_columns[name_col].strip()}') làm tên file")
        print(f"Dùng cột {lb_col} ('{header_columns[lb_col].strip()}') làm Lower Bound")
        print(f"Dùng cột {ub_col} ('{header_columns[ub_col].strip()}') làm Upper Bound")

        # Xử lý từng dòng dữ liệu
        processed_count = 0
        for row_index, row in enumerate(rows[data_start_row:], start=data_start_row):
            if not row or not row[0]:  # Bỏ qua dòng trống
                continue

            # Phân tách dòng dữ liệu thành các cột
            data_columns = row[0].split(',')

            if len(data_columns) <= max(name_col, lb_col, ub_col):
                print(f"Dòng {row_index + 1} không đủ cột: {data_columns}")
                continue

            # Lấy tên file từ đường dẫn đầy đủ
            full_path = data_columns[name_col].strip()
            if not full_path or 'Pack_d' not in full_path:
                continue

            # Trích xuất tên file từ đường dẫn đầy đủ
            filename = Path(full_path).name

            # Trích xuất số Pack từ tên file
            pack_num = ''.join(filter(str.isdigit, filename))

            # Chỉ lấy 3 số đầu tiên (định dạng chuẩn là PackXXX)
            if len(pack_num) > 3:
                pack_num = pack_num[:3]

            # Bỏ các số 0 ở đầu (Pack001 -> Pack1)
            if pack_num:
                pack_num = str(int(pack_num))

                # Tạo key cho dictionary
                key = f"Pack_d{pack_num}"

                # Lấy giá trị lower bound và upper bound
                try:
                    # Đảm bảo bounds > 0, sử dụng None nếu giá trị <= 0 hoặc không hợp lệ
                    lb_str = data_columns[lb_col].strip()
                    ub_str = data_columns[ub_col].strip()

                    # Chuyển đổi định dạng số từ dấu phẩy sang dấu chấm nếu cần
                    lb_str = lb_str.replace(',', '.')
                    ub_str = ub_str.replace(',', '.')

                    lb_value = float(lb_str) if lb_str else 0
                    ub_value = float(ub_str) if ub_str else 0

                    # Chỉ sử dụng bounds nếu chúng > 0
                    lb = int(lb_value) if lb_value > 0 else None
                    ub = int(ub_value) if ub_value > 0 else None

                    # Lưu bounds vào dictionary
                    bounds_dict[key] = (lb, ub)
                    processed_count += 1

                    print(f"Đọc được: {key} -> LB={lb}, UB={ub}")

                    # Thêm các biến thể khác của key để tăng khả năng match
                    if len(pack_num) == 1:
                        bounds_dict[f"Pack_d00{pack_num}"] = (lb, ub)
                        bounds_dict[f"Pack_d0{pack_num}"] = (lb, ub)
                    elif len(pack_num) == 2:
                        bounds_dict[f"Pack_d0{pack_num}"] = (lb, ub)
                except (ValueError, IndexError) as e:
                    print(f"Không thể đọc bounds cho {filename}: {e}")

        # Hiển thị bounds đã đọc
        print(f"\nĐã đọc bounds cho {len(bounds_dict)} files từ {processed_count} dòng dữ liệu:")
        for key, (lb, ub) in list(bounds_dict.items())[:5]:
            print(f"  {key}: LB={lb}, UB={ub}")
        if len(bounds_dict) > 5:
            print("  ...")

        return bounds_dict

    except Exception as e:
        print(f"Lỗi khi đọc file bounds: {e}")
        import traceback
        traceback.print_exc()
        return {}


def read_rcp_file(file_path):
    """
    Đọc file .rcp và trả về thông tin cần thiết
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Dòng đầu tiên: số lượng task và số lượng tài nguyên
    header = lines[0].strip().split()
    n_tasks = int(header[0])
    n_resources = int(header[1])

    # Dòng thứ hai: capacity của các tài nguyên
    capacities = list(map(int, lines[1].strip().split()))

    # Các dòng còn lại: thông tin của các task
    tasks = []
    for i in range(2, len(lines)):
        line = lines[i].strip()
        if line:
            tasks.append(line)

    return n_tasks, n_resources, capacities, tasks


def create_data_file(output_path, n_tasks, n_resources, capacities, tasks, lb, ub):
    """
    Tạo file .data với thông tin bounds
    """
    with open(output_path, 'w') as f:
        # Dòng đầu: số lượng task, số lượng tài nguyên, và bounds nếu có
        if lb is not None and ub is not None:
            f.write(f"{n_tasks} {n_resources} {lb} {ub}\n")
        elif lb is not None:
            f.write(f"{n_tasks} {n_resources} {lb}\n")
        elif ub is not None:
            # Nếu chỉ có UB, thêm một giá trị 0 làm LB placeholder
            f.write(f"{n_tasks} {n_resources} 0 {ub}\n")
        else:
            # Nếu không có bounds, chỉ ghi số task và resource
            f.write(f"{n_tasks} {n_resources}\n")

        # Dòng thứ hai: capacity của các tài nguyên
        f.write(" ".join(map(str, capacities)) + "\n")

        # Các dòng còn lại: thông tin của các task
        for task in tasks:
            f.write(task + "\n")


def convert_excel_to_csv(excel_path, csv_path):
    """
    Thông báo cho người dùng cách chuyển đổi Excel sang CSV
    """
    print("\nLỗi: Không thể đọc file Excel do thiếu thư viện openpyxl")
    print("Bạn cần thực hiện một trong hai cách sau:")
    print("1. Cài đặt openpyxl bằng lệnh: pip install openpyxl")
    print("   Sau đó chạy lại script này")
    print("2. Chuyển đổi file Excel sang CSV bằng cách:")
    print("   - Mở file Excel")
    print("   - Chọn File > Save As")
    print("   - Chọn định dạng CSV (Comma delimited)")
    print("   - Lưu file và sử dụng đường dẫn của file CSV trong script này")
    print("\nBạn cũng có thể sửa script này để sử dụng file CSV thay vì Excel")

    # Trả về False để biết là chưa thành công
    return False


def main():
    # Kiểm tra thư mục và file
    EXCEL_PATH = Path("Convert/bounds/pack_d_FORWARD_STAIRCASE_2025-05-17-07-50-05.xlsx")
    RCP_DIR = Path("Convert/raw_data/pack_d")
    OUTPUT_DIR = Path("Convert/data/pack_d")
    CSV_PATH = Path("Convert/bounds/pack_d_FORWARD_STAIRCASE_2025-05-17-07-50-05.csv")

    if CSV_PATH.exists():
        print(f"Tìm thấy file CSV: {CSV_PATH}")
        bounds_dict = read_bounds_from_csv(CSV_PATH)
    elif EXCEL_PATH.exists():
        # Thông báo cách chuyển đổi Excel sang CSV
        convert_excel_to_csv(EXCEL_PATH, CSV_PATH)

        # Hỏi người dùng có muốn tiếp tục mà không có bounds
        response = input("\nBạn có muốn tiếp tục mà không có thông tin bounds? (y/n): ")
        if response.lower() != 'y':
            print("Đã hủy xử lý")
            return

        # Tiếp tục với bounds rỗng
        bounds_dict = {}
    else:
        print(f"Lỗi: Không tìm thấy file bounds tại {EXCEL_PATH} hoặc {CSV_PATH}")
        return

    if not RCP_DIR.exists():
        print(f"Lỗi: Không tìm thấy thư mục dữ liệu đầu vào tại {RCP_DIR}")
        return

    # Tạo thư mục output nếu chưa tồn tại
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Xử lý từng file .rcp
    processed_count = 0
    error_count = 0

    rcp_files = list(RCP_DIR.glob("*.rcp"))
    print(f"\nTìm thấy {len(rcp_files)} file .rcp để xử lý")

    for rcp_file in rcp_files:
        try:
            # Lấy tên file không có phần mở rộng
            base_name = rcp_file.stem

            # Tìm bounds cho file này
            lb, ub = None, None  # Giá trị mặc định là None, không phải 0
            for key in [base_name,
                        base_name.replace('Pack_d0', 'Pack_d').replace('Pack_d00', 'Pack_d'),
                        # Xử lý trường hợp Pack032b -> Pack32 (bỏ 'b')
                        base_name.replace('b', '')]:
                if key in bounds_dict:
                    lb, ub = bounds_dict[key]
                    break

            # Hiển thị bounds với giá trị rõ ràng
            lb_str = str(lb) if lb is not None else "None"
            ub_str = str(ub) if ub is not None else "None"
            print(f"Xử lý {rcp_file.name}: LB={lb_str}, UB={ub_str}")

            # Đọc file .rcp
            n_tasks, n_resources, capacities, tasks = read_rcp_file(rcp_file)

            # Tạo đường dẫn output
            output_path = OUTPUT_DIR / f"{base_name}.data"

            # Tạo file .data
            create_data_file(output_path, n_tasks, n_resources, capacities, tasks, lb, ub)

            processed_count += 1

        except Exception as e:
            print(f"Lỗi khi xử lý {rcp_file.name}: {str(e)}")
            error_count += 1

    print(f"\nĐã xử lý thành công {processed_count} files, có {error_count} lỗi")
    print(f"Các file .data đã được lưu vào {OUTPUT_DIR}")


if __name__ == "__main__":
    main()