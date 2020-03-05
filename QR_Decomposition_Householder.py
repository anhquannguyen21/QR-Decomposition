import numpy as np



#Reference https://gist.github.com/Hsankesara/cd35edb30825df19f182a6ecf96e126e
def column_convertor(x):
    x.shape = (1, x.shape[0])
    return x

def get_norm(x):
    return np.sqrt(np.sum(np.square(x)))

def householder_transformation(v):
    size_of_v = v.shape[1]
    e1 = np.zeros_like(v)
    e1[0, 0] = 1
    vector = get_norm(v) * e1
    if v[0, 0] < 0.0:
        vector = - vector
    u = (v + vector).astype(np.float32)
    H = np.identity(size_of_v) - ((2 * np.matmul(np.transpose(u), u)) / np.matmul(u, np.transpose(u)))
    return H

def QR_Step_Decomposition(q, r, iter, n):
    v = column_convertor(r[iter:, iter])
    Hbar = householder_transformation(v)
    H = np.identity(n)
    H[iter:, iter:] = Hbar
    r = np.matmul(H, r)
    q = np.matmul(q, H)
    return q, r

def QR_Decomposition(A):
    n,m=A.shape
    Q = np.identity(n)
    R = A.astype(np.float32)
    for i in range(min(n, m)):
        Q, R = QR_Step_Decomposition(Q, R, i, n)
    min_dim = min(m, n)
    R = np.around(R, decimals=6)
    R = R[:min_dim, :min_dim]
    Q = np.around(Q, decimals=6)
    print('A after QR factorization')
    print('R matrix')
    print(R, '\n')
    print('Q matrix')
    print(Q)


A=np.array([[1,1,1],[0,1,1],[0,0,1]])
if __name__ == "__main__":
    QR_Decomposition(np.transpose(A))
