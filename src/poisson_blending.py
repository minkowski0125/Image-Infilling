import numpy as np
import jittor as jt
import jsparse.nn.functional as F
jt.flags.nvcc_flags += '-lcusparse'
jt.flags.use_cuda = 1

def make_matrix(img, result_img, mask):
    d, d_rev = [], {}
    x, y = mask.shape[:2]
    for i in range(x):
        for j in range(y):
            if mask[i][j] == 1:
                d_rev[i * y + j] = len(d)
                d.append(i * y + j)
    params, nps, vals, origin_img_vals = [], [], [], []
    for k in range(len(d)):
        idx = d[k]
        i, j = idx // y, idx % y
        n_p, val = 0, 0
        param = []
        for (m, n) in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]:
            if m < 0 or m >= x or n < 0 or n >= y:
                continue
            n_p += 1
            if mask[m][n] == 1:
                param.append(d_rev[m * y + n])
            else:
                val += img[m][n]
            val -= result_img[m][n]
        val += n_p * result_img[i][j]
        params.append(param)
        nps.append(n_p)
        vals.append(val)
        origin_img_vals.append(result_img[i][j])
    return d, d_rev, params, nps, vals, origin_img_vals

def solve_matrix(params, nps, vals, origin_img_vals=None, params_val=None):
    size = len(nps)
    rows = []
    cols = []
    m_vals = []
    for i in range(size):
        rows.extend([i]*len(params[i]))
        cols.extend(params[i])
        if params_val is None:
            m_vals.extend([1 / nps[i] for id in params[i]])
        else:
            m_vals.extend([params_val[i][j] / nps[i] for j, id in enumerate(params[i])])
    
    vals_3d = [val / nps[i]  for i, val in enumerate(vals)]
    outputs = []
    with jt.no_grad():
        for i in range(3):
            if origin_img_vals is None:
                vec = jt.array(np.ones((size, 1)))
            else: 
                vec = jt.array(np.array([item[i] for item in origin_img_vals])[:, None])
            vals_1d = jt.array(np.array([item[i] for item in vals_3d])[:, None])
            while True:
                output = F.spmm(
                    rows=jt.array(rows),
                    cols=jt.array(cols),
                    vals=jt.array(m_vals),
                    size=(size, size),
                    mat=vec,
                    cuda_spmm_alg=1
                )
                output = output + vals_1d
                dif = (output - vec).abs().max()
                print(dif, end='\r')
                if dif <= 0.1:
                    break
                vec = output
            outputs.append(output)
    return np.concatenate([output.numpy() for output in outputs], axis=-1)

def test_solve():
    params = [[1,2], [0,2], [0,1]]
    params_val = [[3,-2], [-4,1], [-6,-3]]
    nps = [8, 11, 12]
    vals = [np.ones(3, dtype=np.float64) * 20, np.ones(3, dtype=np.float64) * 33, np.ones(3, dtype=np.float64) * 36]
    x = solve_matrix(params, nps, vals, params_val=params_val)
    print(x)

def poisson_blending(img, result_img, mask):
    d, d_rev, params, nps, vals, origin_img_vals = make_matrix(img, result_img, mask)
    x = solve_matrix(params, nps, vals, origin_img_vals=origin_img_vals)
    
    y = img.shape[1]
    for k, idx in enumerate(d):
        i, j = idx // y, idx % y
        img[i, j] = x[k]
    return img
    
if __name__ == '__main__':
    test_solve()