import csv

# 输入文件
input_file = "train_images_shuffle.txt"
# 输出文件
output_file = "label2class.csv"

label_class_dict = {}

with open(input_file, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # 分割文件路径和数字标签
        path, label = line.split()
        label = int(label)
        # 提取类名
        # 先获取目录名，再去掉数字前缀
        dirname = path.split('/')[0]      # '031.Black_billed_Cuckoo'
        class_name = dirname.split('.')[1]  # 'Black_billed_Cuckoo'
        label_class_dict[label] = class_name

# 写入 CSV
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['label', 'class_name'])
    for label in sorted(label_class_dict.keys()):
        writer.writerow([label, label_class_dict[label]])

print(f"Done! Mapping saved to {output_file}")
