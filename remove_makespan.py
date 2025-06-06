import os

# Đường dẫn của folder nguồn và folder đích
source_folder = "Convert/data/j90"
destination_folder = "Convert/data/j90_no_bound"

# Tạo folder đích nếu chưa tồn tại
os.makedirs(destination_folder, exist_ok=True)

# Duyệt qua tất cả các file trong folder nguồn
for file_name in os.listdir(source_folder):
    source_file_path = os.path.join(source_folder, file_name)

    # Bỏ qua nếu không phải là file .data
    if not file_name.endswith(".data"):
        continue

    destination_file_path = os.path.join(destination_folder, file_name)

    # Đọc nội dung file nguồn, chỉnh sửa và ghi lại vào file đích
    with open(source_file_path, "r") as source_file:
        lines = source_file.readlines()

        # Sửa dòng đầu tiên bằng cách xóa số thứ 3
        if lines:
            first_line = lines[0].split()
            # Chỉ xóa nếu có ít nhất 3 số trong dòng
            if len(first_line) >= 3:
                # Xóa số thứ 3
                del first_line[2]
                # Ghép lại các phần tử vào dòng đầu tiên
                lines[0] = " ".join(first_line) + "\n"

    # Lưu dữ liệu mới vào file đích
    with open(destination_file_path, "w") as destination_file:
        destination_file.writelines(lines)

print("Hoàn thành việc chỉnh sửa và sao chép các file!")
