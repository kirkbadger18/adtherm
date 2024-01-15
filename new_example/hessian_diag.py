import numpy.linalg as LA
import numpy as np

H = np.loadtxt('rigid_body_hessian.dat')

eig, vec = LA.eigh(H)

print(eig)
