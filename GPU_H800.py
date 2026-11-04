import os
import subprocess

print("=== 系统底层硬件检测 ===")

# 1. 检查 /dev 目录下的硬件映射文件
print("\n[1] 检查设备映射文件 (/dev/nvidia*):")
if os.path.exists('/dev/nvidia0') or os.path.exists('/dev/nvidiactl'):
    print("✅ 找到底层的 NVIDIA 设备文件，容器已分配 GPU。")
else:
    print("❌ 未找到 /dev/nvidia 映射文件。容器底层没有分配到显卡设备。")

# 2. 检查主板 PCIe 插槽上的硬件 (相当于终端的 lspci)
print("\n[2] 检查主板 PCIe 插槽硬件:")
try:
    # 调用系统 lspci 命令
    result = subprocess.run(['lspci'], stdout=subprocess.PIPE, text=True)
    # 过滤包含 nvidia 或 vga 的行
    gpu_hardware = [line for line in result.stdout.split('\n') if 'nvidia' in line.lower() or 'vga' in line.lower()]
    
    if gpu_hardware:
        print("✅ 主板/虚拟机底层发现以下显卡硬件:")
        for dev in gpu_hardware:
            print(f"  -> {dev}")
    else:
        print("❌ 主板上未检测到任何显卡硬件。")
        print("⚠️ 结论：系统当前大概率处于【无卡模式开机】状态，或者云平台的虚拟机分配出错。")
except FileNotFoundError:
    print("⚠️ 系统未安装 lspci 工具，跳过此项检查。")