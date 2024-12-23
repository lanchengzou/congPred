import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import normalized_root_mse as nrms

def call_metrics(y, y_bar, axis=0):
    ssim_res = ssim(y, y_bar, data_range=y.max()-y.min(),channel_axis=axis)
    nrms_res = nrms(y, y_bar,normalization='min-max')

    threshold = np.percentile(y, 98)
    indices = np.where(y >= threshold)

    y_gt_top = y[indices]
    y_top = y_bar[indices]

    # 计算MSE
    mse2 = np.mean((y_gt_top - y_top) ** 2)

    threshold = np.percentile(y, 95)
    indices = np.where(y >= threshold)

    y_gt_top = y[indices]
    y_top = y_bar[indices]

    # 计算MSE
    mse5 = np.mean((y_gt_top - y_top) ** 2)

    threshold = np.percentile(y, 90)
    indices = np.where(y >= threshold)

    y_gt_top = y[indices]
    y_top = y_bar[indices]

    # 计算MSE
    mse10 = np.mean((y_gt_top - y_top) ** 2)

    if np.isnan(ssim_res): 
        ssim_res = None
    elif np.isinf(ssim_res): 
        ssim_res = None
    if np.isnan(nrms_res): 
        nrms_res = None
    elif np.isinf(nrms_res): 
        nrms_res = None
    return ssim_res, nrms_res, mse2, mse5, mse10
