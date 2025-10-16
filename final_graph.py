import numpy as np


graph_adjacency = {
    'v1':['v2','v4','u7'],
    'v2':['v1', 'v2','v10'],
    'v3':['v4','v5','u2', 'u5'],
    'v4':['v6', 's2', 'w'],
    'v5':['s4'],
    'v6':['v2','s1','s2'],
    'v7':['v9'],
    'v8':['v1','v4','v9'],
    'v9':['v7'],
    'v10':['v8','v9','v10','s3'],
    'u1':['u1','u10','v5','s8'],
    'u2':['u1','v3'],
    'u3':['u2','u3','u4'],
    'u4':['s2','u5','u10'],
    'u5':['u3','u5','u9'],
    'u6':['u4','u8','u10'],
    'u7':['s5','w','u5'],
    'u8':['u7','u9','s5','s6'],
    'u9':['u6'],
    'u10':[],
    's1':['u2','s7'],
    's2':[],
    's3':['s2','v5','v6'],
    's4':['s1','s3','s7','u6'],
    's5':['s6','s10','u1'],
    's6':['u3','u10'],
    's7':['s7','s9','s8'],
    's8':['s9','s10'],
    's9':['s8'],
    's10':['s7'],
    'w':['w']
}

vertices = list(graph_adjacency.keys())
n = len(vertices)

vertex_idx = {vertex: idx for idx, vertex in enumerate(vertices)}

A = np.zeros((n,n), dtype=float)

for vertex, edges in graph_adjacency.items():
    i = vertex_idx[vertex]
    for edge in edges:
        if edge in vertex_idx:
            j = vertex_idx[edge]
            A[i,j] = 1

print("Row sums (out-degrees):", np.sum(A, axis=1))
print("Adjacency matrix shape:", A.shape)
print("Sample row (first node):", A[0])


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
        print(f"Converged to r vector:\n{r_new}")
        r_vector = r_new
        break
    r_vector = r_new
else:
    print('Max iteration reached')

print("\nVertex/website Probabilities")
for idx,vertex in enumerate(vertices):
    print(f"{vertex}: {r_vector[idx][0]*100: .4f}%")


r_probs = r_vector.flatten()
top_vertices = np.argsort(r_probs)[::-1][:10]
final_vertex_probs = [(vertices[idx], (r_probs[idx]*100).item()) for idx in top_vertices]
print("Top ten vertex probabilities:")
for vertex, probability in final_vertex_probs:
    print(f"{vertex} ={probability: .4f}%")
