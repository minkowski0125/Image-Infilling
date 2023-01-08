import os
import numpy as np
from PIL import Image
from tqdm import tqdm

from conv import l2_conv
from graph_cut import graph_cut
from poisson_blending import poisson_blending
from utils import get_shift

def dilate_one_step(src, visit_map):
    """单步dilate
    """
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
    """BFS进行dilate
    """
    output = src.copy()
    visit_map = np.ones_like(src)
    for i in tqdm(range(k), desc='dilating'):
        output, visit_map = dilate_one_step(output, visit_map)
    return output

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
    
    # 读取mask
    mask = Image.open(mask_path)
    mask = (np.asarray(mask).sum(axis=-1) == 0).astype(np.float64)
    
    # 使用bfs，dilate扩大mask
    k = 5
    dilated_mask = dilate(mask, k=k) - mask
    
    # 读取待补全图片与补全目标的图片
    img = np.asarray(Image.open(img_path)).astype(np.float64)
    result_img = Image.open(result_path)
    
    # 获得待补全区域大小（子矩形）
    x_0, x_1, y_0, y_1 = get_shift(dilated_mask + mask)
    mask_x, mask_y = x_1 - x_0, y_1 - y_0
    
    # 如果补全目标图片不够大，进行resize
    result_y, result_x = result_img.size
    if mask_x >= result_x:
        result_img = result_img.resize((int(result_y / result_x * mask_x)+1, mask_x+1))
    elif mask_y >= result_y:
        result_img = result_img.resize((mask_y+1, int(result_x / result_y * mask_y)+1))
    result_img = np.asarray(result_img).astype(np.float64)
    
    # 精细匹配卷积
    conv_output = l2_conv(img, result_img, dilated_mask + mask)
    x, y = conv_output.shape
    idx = conv_output.argmin()
    shift_x, shift_y = idx // y, idx % y
    img_piece = img[x_0:x_1, y_0:y_1]
    result_img_piece = result_img[shift_x:shift_x+x_1-x_0, shift_y:shift_y+y_1-y_0]
    
    # 计算融合边界
    graphcut_mask = graph_cut(img_piece, result_img_piece, mask, dilated_mask, x_0, x_1, y_0, y_1)
    
    # 自然融合
    blending_result = poisson_blending(img_piece, result_img_piece, (graphcut_mask+mask)[x_0:x_1, y_0:y_1])
    output_img = img.copy()
    output_img[x_0:x_1, y_0:y_1] = blending_result
    
    # 存储结果图像
    output_img = Image.fromarray(output_img.astype(np.uint8))
    # output_img.save(f"outputs/input{input_id}_result{result_id}.jpg")
    
if __name__ == '__main__':
    for i in range(1, 5):
        for j in range(1, 21):
            if os.path.exists(f"outputs/input{i}_result{j}.jpg"):
                continue
            try:
                main(input_id=i, result_id=j)
            except:
                continue
    for j in tqdm(range(1, 21)):
        if os.path.exists(f"outputs/inputcase_result{j}.jpg"):
            continue
        try:
        main(input_id=3, result_id=j, case=True)
        except:
            continue
    # main(input_id=3, result_id=1, case=True)