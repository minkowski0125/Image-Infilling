import numpy as np
from PIL import Image
from tqdm import tqdm

from conv import l2_conv
from graph_cut import graph_cut
from poisson_blending import poisson_blending
from utils import get_shift

def dilate_one_step(src, visit_map):
    output = src.copy()
    
    # sign = np.logical_and(src, visit_map)
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            if src[i][j] == 0 or visit_map[i][j] == 0:
                continue
            visit_map[i][j] = 0
            if i > 0:
                output[i-1][j] = 1
            if i < src.shape[0]-1:
                output[i+1][j] = 1
            if j > 0:
                output[i][j-1] = 1
            if j < src.shape[1]-1:
                output[i][j+1] = 1
    return output, visit_map
            
def dilate(src, k=1):
    output = src.copy()
    visit_map = np.ones_like(src)
    for i in tqdm(range(k), desc='dilating'):
        output, visit_map = dilate_one_step(output, visit_map)
    return output

# def dilate(src, k=1):
#     output = src.copy()
#     x, y = src.shape
#     for i in range(src.shape[0]):
#         for j in range(src.shape[1]):
#             print('dilating...', i * y + j, end='\r')
#             if src[i][j] == 1:
#                 continue
#             p_x = np.abs(np.arange(x) - i)
#             p_y = np.abs(np.arange(y) - j)
#             p = (p_x[:, None].repeat(y, axis=1) + p_y[None, :].repeat(x, axis=0)) <= k
#             output[i][j] = np.logical_and(p, src).any() > 0
#     return output

def conv(x, w):
    N,H,W,C = x.shape
    Kh, Kw, _C, Kc = w.shape
    assert C==_C
    xx = x.reindex([N,H-Kh+1,W-Kw+1,Kh,Kw,C,Kc], [
        'i0', # Nid
        'i1+i3', # Hid+Khid
        'i2+i4', # Wid+KWid
        'i5', # Cid|
    ])
    ww = w.broadcast_var(xx)
    yy = xx*ww
    y = yy.sum([3,4,5]) # Kh, Kw, c
    return y

def main(input_id=1, result_id=1, case=False):
    if case:
        img_path = f"data/inputcase.jpg"
        mask_path = f"data/inputcase_mask.jpg"
        result_path = f"data/input{input_id}/result_img{str(result_id).zfill(3)}.jpg"
        input_id = 'case'
    else:
        img_path = f"data/input{input_id}.jpg"
        mask_path = f"data/input{input_id}_mask.jpg"
        result_path = f"data/input{input_id}/result_img{str(result_id).zfill(3)}.jpg"
    
    mask = Image.open(mask_path)
    mask = (np.asarray(mask).sum(axis=-1) == 0).astype(np.float64)
    
    k = 80
    dilated_mask = dilate(mask, k=k) - mask
    
    img = np.asarray(Image.open(img_path)).astype(np.float64)
    result_img = np.asarray(Image.open(result_path)).astype(np.float64)
    
    conv_output, (x_0, x_1, y_0, y_1) = l2_conv(img, result_img, dilated_mask, dilated_mask + mask)
    x, y = conv_output.shape
    idx = conv_output.argmin()
    shift_x, shift_y = idx // y, idx % y
    img_piece = img[x_0:x_1, y_0:y_1]
    result_img_piece = result_img[shift_x:shift_x+x_1-x_0, shift_y:shift_y+y_1-y_0]
    
    graphcut_mask = graph_cut(img_piece, result_img_piece, mask, dilated_mask, x_0, x_1, y_0, y_1)
    
    blending_result = poisson_blending(img_piece, result_img_piece, (graphcut_mask+mask)[x_0:x_1, y_0:y_1])
    output_img = img.copy()
    output_img[x_0:x_1, y_0:y_1] = blending_result
    output_img = Image.fromarray(output_img.astype(np.uint8))
    output_img.save(f"outputs/input{input_id}_result{result_id}.jpg")
    
if __name__ == '__main__':
    for i in range(1, 5):
        for j in range(1, 21):
            main(input_id=i, result_id=j)
        
    for j in tqdm(range(1, 21)):
        main(input_id=3, result_id=j, case=True)