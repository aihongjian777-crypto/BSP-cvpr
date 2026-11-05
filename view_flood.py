import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np


path_pred = '/mnt/e/CoW-Linest/CoW-main/runs/CoW_train_CHAOST2_cv0/26/interm_preds/prediction_8_LK.nii.gz'

path_img = '/mnt/e/CoW-Linest/CoW-main/data/CHAOST2/image_8.nii.gz'

try:
    pred = nib.load(path_pred).get_fdata()
    img = nib.load(path_img).get_fdata()

    slice_idx = 8

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img[:, :, slice_idx], cmap='gray')
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(img[:, :, slice_idx], cmap='gray')
    plt.imshow(pred[:, :, slice_idx], cmap='Reds', alpha=0.6)
    plt.title('Your Prediction (The Flood)')

    plt.savefig('flood_check.png')
    print("📸 图片已保存为 flood_check.png，快打开看看！")

except Exception as e:
    print(f"出错: {e}")
