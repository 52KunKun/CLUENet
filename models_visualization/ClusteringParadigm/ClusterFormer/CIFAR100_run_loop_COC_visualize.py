import subprocess
import sys
import os
# 自动获取当前解释器路径
python_exe = sys.executable

# CPVT_mini
blocks = [2,2,2,2]
heads = [1, 2, 5, 8]
image_path_list = [1]
for image in image_path_list:
    image_path = str(image)
    # 遍历 stage 和 head
    for stage in range(len(heads)):  # stage: 0~3
        for block in range(blocks[stage]):
            for head in range(heads[stage]):  # head: 0~7
                print(f"正在执行：stage={stage},block={block}, head={head}")
                # 构造命令行参数
                cmd = [
                    python_exe, "CIFAR100_cluster_visualize_COC.py",
                    "--image", image_path,
                    "--model", "coc_tiny",
                    "--num-classes", "100",
                    "--block",f'{block}',
                    "--stage", f'{stage}',
                    "--head", f'{head}',
                    "--block",f'{1}',
                    "--checkpoint", "./coc_mini_cifar_CIFAR100_epoch105.pth",
                    "--alpha", "1.0",
                ]
                # 执行命令
                subprocess.run(cmd)

