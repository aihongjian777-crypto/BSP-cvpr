import os
file_path = "/root/autodl-tmp/BSP/CoW-main/checkpoints/deeplabv3_resnet101_coco-586e9e4e.pth"
print(f"文件大小: {os.path.getsize(file_path) / 1024 / 1024:.2f} MB")