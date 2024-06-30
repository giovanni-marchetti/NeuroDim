# import numpy as np 
# from numpy.random import randn
# from numpy import reshape
# import skdim
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.linalg import matrix_rank

from attn import *

d = 4
t = 3

num_samples = 10
num_iters = 5
normalize = True

dim_vec = []
a_vec = np.arange(1, d + 1)
l_vec = np.array([3] * len(a_vec))

for l, a in zip(l_vec, a_vec):
    print('Hidden dimension: ', a)
    print('Number of layers: ', l)
    ranks = []
    for i in range(num_iters):
        grads = []
        model = deep_attn(l, d, a, normalize=normalize)  
        x =  torch.randn(num_samples, t, d) 
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
# plt.gca().spines['bottom'].set_visible(False)
# plt.gca().spines['left'].set_visible(False)

    # plt.gca().get_xaxis().set_ticks([])
    # plt.gca().get_yaxis().set_ticks([])


# plt.plot(a_vec, dim_vec, label='Estimated', linewidth=5.)

true_dim_vec = l_vec * (a_vec * (2 * d - a_vec) + (d * d)) - (l_vec - 1) * (d * d) - l_vec * (1 - int(normalize))
plt.plot(a_vec, true_dim_vec, label='Estimated', linewidth=6., c='royalblue')
plt.plot(a_vec, true_dim_vec, label='Expected', linewidth=7., linestyle='dotted', c='red')

plt.gca().set_ylim(true_dim_vec[0] - 5 , true_dim_vec[-1] + 5)
plt.xlabel('Hidden dim. (a)')
plt.ylabel('Neuromanifold dim.')
plt.legend()
# plt.show()
plt.savefig('./dimexp.pdf')



#     out = reshape(attn_layer(Q, K, V, x, normalize=normalize), [b1,-1])

#     estimator = skdim.id.KNN(20)
#     # estimator = skdim.id.DANCo(k=10)
#     # estimator = skdim.id.MLE('dnoiseGaussH')
#     # estimator = skdim.id.TwoNN()
#     dim = estimator.fit_transform(out).mean()
#     dim_vec.append(dim)