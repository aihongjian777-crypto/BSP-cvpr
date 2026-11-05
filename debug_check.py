import nibabel as nib
import numpy as np
import os


path_pred = '/mnt/e/CoW-Linest/CoW-main/runs/CoW_train_CHAOST2_cv0/26/interm_preds/prediction_8_LK.nii.gz'


print(f"🔍 正在检查文件: {path_pred}")

if not os.path.exists(path_pred):
    print("❌ 错误：找不到文件！请确认文件夹里是否有 prediction_8_LK.nii.gz")
else:
    try:

        nii = nib.load(path_pred)
        data = nii.get_fdata()

        pixel_count = np.sum(data > 0.1)

        print("-" * 40)
        print(f"📏 图像尺寸 (Shape): {data.shape}")
        print(f"🟥 红色像素数量 (Count): {pixel_count}")
        print("-" * 40)

        if pixel_count == 0:
            print("💀 【诊断结果】：空文件 (全黑)！")
            print("   原因：之前的参数 (bg_factor=0.6) 太狠了，把左肾彻底抑制没了。")
            print("   👉 对策：我们需要回调参数，或者换用 Hard Prototypes。")
        else:
            print(f"👻 【诊断结果】：有内容！(发现了 {pixel_count} 个像素)")
            print("   原因：模型预测成功了，只是 Slicer 显示位置没对上。")
            print("   👉 对策：可以直接进行下一步的可视化分析。")

            z_sum = np.sum(data, axis=(0, 1))
            slices = np.where(z_sum > 0)[0]
            if len(slices) > 0:
                print(f"📍 它们藏在第 {slices} 层 (去Slicer看这些层!)")

    except Exception as e:
        print(f"❌ 读取出错: {e}")
