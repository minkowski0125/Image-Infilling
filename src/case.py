import numpy as np
import jittor as jt
import jsparse.nn.functional as F
jt.flags.nvcc_flags += '-lcusparse'
jt.flags.use_cuda = 1

size = 1024

### construct sparse matrix
mat = np.random.rand(size,size).astype(np.float32)
mat = np.where(mat > 0.995, mat, 0)
indices = np.nonzero(mat)
values = mat[indices]

### construct vector
vec = np.random.rand(size,1).astype(np.float32)

## execute sparse matrix multiply a vector
output = F.spmm(
        rows=jt.array(indices[0]), 
        cols=jt.array(indices[1]), 
        vals=jt.array(values), 
        size=(size,size), 
        mat=jt.array(vec), 
        cuda_spmm_alg=1)

print(output)