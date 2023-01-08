import numpy as np
import jittor as jt

from utils import get_shift

def conv(x, w):
    """卷积函数
    """
    H,W,C = x.shape
    Kh, Kw, _C, Kc = w.shape
    assert C==_C
    xx = x.reindex([H-Kh+1,W-Kw+1,Kh,Kw,C,Kc], [
        'i0+i2', # Hid+Khid
        'i1+i3', # Wid+KWid
        'i4', # Cid|
    ])
    ww = w.broadcast_var(xx)
    yy = xx*ww
    y = yy.sum([2,3,4]) # Kh, Kw, c
    return y

def l2_conv(img, result_img, dilated_mask):
    x, y = img.shape[:2]
    x_0, x_1, y_0, y_1 = get_shift(dilated_mask)
    
    img_piece = img[x_0:x_1, y_0:y_1]
    mask_piece = dilated_mask[x_0:x_1, y_0:y_1]
    
    # 转化为jittor
    img_jt = jt.array(img_piece / 255.)
    mask_jt = jt.array(mask_piece)
    result_img_jt = jt.array(result_img / 255.)
    
    with jt.no_grad():
        masked_img = (img_jt * mask_jt[..., None])[..., None]
        
        # \sum X_pZ_p
        conv_1 = conv(result_img_jt, masked_img).fetch_sync()
        
        # \sum X_p^2
        conv_2 = conv(result_img_jt.pow(2).sum(-1)[..., None], mask_jt[..., None, None]).fetch_sync()
        
        # \sum Z_p^2
        img_pow_sum = masked_img.pow(2).sum()
        img_pow_sum_b = img_pow_sum[None, None, ...].broadcast_var(conv_1)
        
        # \sum (X_p-Z_p)^2
        output = conv_2 + img_pow_sum_b - 2 * conv_1
    output = np.squeeze(output.numpy())
    return output
    