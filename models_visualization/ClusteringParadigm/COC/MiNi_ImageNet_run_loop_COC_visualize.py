import subprocess
import sys
import os
# 自动获取当前解释器路径
python_exe = sys.executable

# CPVT_mini
blocks = [3, 4, 5, 2]
heads = [4, 4, 8, 8]
image_path_list = ["n0153282900000016.jpg"]
for image in image_path_list:
    image_path = str(image)
    # 遍历 stage 和 head
    for stage in range(len(heads)):  # stage: 0~3
        for block in range(blocks[stage]):
            for head in range(heads[stage]):  # head: 0~7
                print(f"正在执行：stage={stage},block={block}, head={head}")
                # 构造命令行参数
                cmd = [
                    python_exe, "MiNiImageNet_cluster_visualize_ClusterFormer.py",
                    "--image", image_path,
                    "--data_path",r'C:\Users\13779\Desktop\dataset\MINI_ImageNet\mini-imagenet\images',
                    "--model", "coc_tiny",
                    "--num-classes", "100",
                    "--block",f'{block}',
                    "--stage", f'{stage}',
                    "--head", f'{head}',
                    "--checkpoint", "./COC_mini_epoch105.pth",
                    "--alpha", "1.0",
                ]
                # 执行命令
                subprocess.run(cmd)

