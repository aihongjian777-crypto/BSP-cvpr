import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

# 你的Linux路径
path_pred = '/mnt/e/CoW-Linest/CoW-main/runs/CoW_train_CHAOST2_cv0/26/interm_preds/prediction_8_LK.nii.gz'
# 对应的原图路径 (请确认这个路径在你的Linux下是对的，如果不对请修改)
path_img = '/mnt/e/CoW-Linest/CoW-main/data/CHAOST2/image_8.nii.gz' 

try:
    pred = nib.load(path_pred).get_fdata()
    img = nib.load(path_img).get_fdata()
    
    # 也就是中间那一层
    slice_idx = 8 
    
    plt.figure(figsize=(10, 5))
    
    # 原图
    plt.subplot(1, 2, 1)
    plt.imshow(img[:, :, slice_idx], cmap='gray')
    plt.title('Original Image')
    
    # 预测图 (看看是不是红成一片)
    plt.subplot(1, 2, 2)
    plt.imshow(img[:, :, slice_idx], cmap='gray')
    plt.imshow(pred[:, :, slice_idx], cmap='Reds', alpha=0.6) # 红色半透明叠加
    plt.title('Your Prediction (The Flood)')
    
    plt.savefig('flood_check.png')
    print("📸 图片已保存为 flood_check.png，快打开看看！")

except Exception as e:
    print(f"出错: {e}")