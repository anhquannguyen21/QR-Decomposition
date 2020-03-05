import numpy as np


def normalize(vector):
    norm_vector=np.linalg.norm(vector)
    return vector/norm_vector


def GramSchmidtProcess(A):
    A=np.transpose(A)
    n=A.shape[1]
    B=np.zeros((n,n), dtype=np.float32)
    e_1=normalize(A[:,0])
    B[:,0]=e_1
    for i in range(1,n):
        temp=0
        for j in range(0,i):
            temp=temp+(np.inner(A[:,i],B[:,j]))*B[:,j]
        temp2=normalize(A[:,i]-temp)
        B[:,i]=temp2
    return B

#Alternative 1:
#A=QR => R=Q^-1.dot(A)
def QR_Decomposition_GramSchimidt_1(A):
    Q=GramSchmidtProcess(A)
    R=(Q.T).dot(A.T)
    return Q,R

A=np.array([[1,1,1],[0,1,1],[0,0,1]])
Q,R=QR_Decomposition_GramSchimidt_1(A)
print(Q)
print(R)

#Alternative 2

def QR_Decomposition_GramSchmidt_2(A):
    Q=GramSchmidtProcess(A)
    A=np.transpose(A)
    m=A.shape[0]
    R=np.zeros((m,m))
    for i in range(0, m):
        for j in range(0, i+1):
            R[j][i]=np.inner(A[:,i], Q[:,j])
    return Q,R
Q,R=QR_Decomposition_GramSchmidt_2(A)
print(Q)
print(R)






