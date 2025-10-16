import numpy as np

A = np.array([
    [0,0,1,1,0,0],
    [1,0,0,0,0,1],
    [0,0,0,0,1,1],
    [0,0,1,0,1,0],
    [0,0,1,0,0,0],
    [0,0,0,0,1,1]
], dtype=float)
n = A.shape[0]
#Sum of rows 
out_deg = np.sum(A,axis=1)

#Transition Matrix creation
M = np.zeros((n,n))
for i in range(n):
    if out_deg[i] > 0:
        M[:,i] = A[i,:]/out_deg[i]
    else:
        M[i,i] = 1.0


#r vector initialization
r_vector = np.ones((n, 1)) / n
tol = 1e-6
max_iters = 1000
beta = 0.85
c = (1-beta)/n * np.ones((n,1))

for i in range(max_iters):
    r_new = beta*(M @ r_vector) + c

    if np.linalg.norm(r_new-r_vector,1) < tol:
        print(f"Converged in {i+1} iterations")
        r_vector = r_new
        break
    print(f"r vector after iteration {i+1}:\n{r_new}")
    r_vector = r_new
else:
    print('Max iteration reached')

vertices = ['A','B','C','D','E','F']
for i in range(n):
    print(f"Vertex/website {vertices[i]} captures {r_vector[i][0]*100:.2f}% of all network traffic")

