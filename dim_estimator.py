import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.linalg import matrix_rank

from attn import *

t = 3
num_layers = 2
normalize = True

dim_vec = []
d_list = list(range(3, 11))
d_vec = np.array(d_list)
l_vec = np.array([num_layers] * len(d_list))
a_vec = np.array([2] * len(d_list))

sample_sizes =  [25, 40, 65, 85, 110, 140, 175, 250]
n_iter_array = [5, 5, 5, 5, 5, 5, 5, 8]

print(sample_sizes)

for l, a, d, n_samples, n_iters in zip(l_vec, a_vec, d_vec, sample_sizes, n_iter_array):
    print('Hidden dimension: ', a)
    print('Number of layers: ', l)
    print('Embedding dimension: ', d)
    print('Number of samples', n_samples)
    ranks = []
    for i in range(n_iters):
        grads = []
        model = deep_attn(l, d, a, normalize=normalize)
        x =  torch.randn(int(n_samples), t, d)
        y = model(x).flatten()
        for j in range(len(y)):
            model.zero_grad()
            y[j].backward(retain_graph=True)
            gr = torch.cat([p.grad.flatten() for p in model.parameters()])
            grads.append(gr.unsqueeze(0))
        jacob = torch.cat(grads, dim=0)
        ranks.append(matrix_rank(jacob))
    dim_vec.append(max(ranks))
 

plt.figure()
plt.rcParams.update({'font.size': 12})
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

true_dim_vec = l_vec * (a_vec * (2 * d_vec - a_vec) ) + (d_vec * d_vec) - l_vec * (1 - int(normalize))

plt.scatter(d_vec, true_dim_vec, label='Expected', c='red', marker='x', s=110)
plt.scatter(d_vec, dim_vec, label='Estimated', c='royalblue', marker='o', s=25)

plt.xticks(d_vec)

plt.gca().set_ylim(true_dim_vec[0] - 5 , true_dim_vec[-1] + 5)
plt.xlabel('Embedding Dimension (δ)')
plt.ylabel('Neuromanifold Dimension')
plt.legend()

plt.savefig('./dimexp_delta.pdf')


