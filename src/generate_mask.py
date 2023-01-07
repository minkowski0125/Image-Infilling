from tqdm import tqdm
import numpy as np
from PIL import Image

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

img = Image.open("data/inputcase.jpg")
a = np.asarray(img).sum(axis=-1)
b = (a < 80).astype(np.uint8)
b[:100] = 0
c = dilate(b, k=15)
c = 1 - c

c = c[..., None].repeat(3, axis=-1) * 255
out_img = Image.fromarray(c)
out_img.save('data/inputcase_mask.jpg')