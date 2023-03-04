from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

import lmfit

#%%

def combine_data(x1, y1, x2, y2, path = None):
    mask = np.isin(x2, x1)
    np.append(x1, x2[~mask])  
    np.append(y1, y2[~mask])
    
    for i in range(x2[mask].size):
        idx = np.where(x1==x2[mask][i])[0][0]
        if y1[idx] > y2[mask][i]:
            y1[idx] = y2[mask][i]
            
    if path is not None:
        x1 = np.save(path+'sweep.npy')
        y1 = np.save(path+'cost.npy')
        
    return x1, y1

#%%
path = '2p6m_ADAMWR/'
x = np.load(path+'sweep.npy')
y = np.load(path+'cost.npy')

x, y = combine_data(x, y, np.load('2p6m_ADAMWR_r/sweep.npy'), np.load('2p6m_ADAMWR_r/cost.npy'))

params = lmfit.Parameters()
params.add_many(('a', 1/9), ('b', 1), ('c', 1/9))


def exp3p2(x, a, b, c):
    return np.exp(a + b * x + c * (x**2))

def exp3p1(x, a, b, c):
    return a * np.exp(b/(x+c))

def explinear(x, a, b, c, d):
    return a * np.exp(-x/b) + c + d * x

def relu(x, a, b, c):
    return np.maximum(a, b * x) - c 

model = lmfit.Model(relu, independent_vars=['x'])
fit_result = model.fit(y, x=x, a=1/9, b=1, c=1/9)

fit_result.plot_fit()
plt.show()

print(fit_result.fit_report())

#%%

best_vals = defaultdict(lambda: np.zeros(x.size))
stderrs = defaultdict(lambda: np.zeros(x.size))
chi_sq = np.zeros_like(x)
for i in range(x.size):
    idx2 = np.arange(0, x.size)
    idx2 = np.delete(idx2, i)
    tmp_x = x[idx2]
    tmp = model.fit(y[idx2], x=tmp_x, a=fit_result.params['a'],
                    b=fit_result.params['b'],
                    c=fit_result.params['c'])
    chi_sq[i] = tmp.chisqr
    for p in tmp.params:
        tpar = tmp.params[p]
        best_vals[p][i] = tpar.value
        stderrs[p][i] = (tpar.stderr / fit_result.params[p].stderr)
        
fig, ax = plt.subplots()
ax.plot(x, (fit_result.chisqr - chi_sq) / chi_sq)
ax.set_ylabel(r'Relative red. $\chi^2$ change')
ax.set_xlabel('x')

repeat_sweep = []
repeat = []
for i in range(x.size):
    if ((fit_result.chisqr - chi_sq) / chi_sq)[i] > 0.05:
        repeat_sweep.append(x[i])
        repeat.append(i)
        
print(repeat_sweep)
print(repeat)

#%%
fig, axs = plt.subplots(6, figsize=(4, 7), sharex='col')
axs[0].plot(x, best_vals['a'])
axs[0].set_ylabel('best a')

axs[1].plot(x, best_vals['b'])
axs[1].set_ylabel('best b')

axs[2].plot(x, best_vals['c'])
axs[2].set_ylabel('best c')

axs[3].plot(x, stderrs['a'])
axs[3].set_ylabel('err a change')

axs[4].plot(x, stderrs['b'])
axs[4].set_ylabel('err b change')

axs[5].plot(x, stderrs['c'])
axs[5].set_ylabel('err c change')

axs[5].set_xlabel('x')