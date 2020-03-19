# -*- coding: utf-8 -*-
"""
Created on Mon May 16 17:11:29 2016

@author: deanpospisil
"""
import numpy as np
#def forward_selection(X, y, n):
X = np.random.randn(370, 100)
y = np.tile(np.random.randn(370, 1), (1,10))
n = 20
n = min([y.shape[0], n])
best_model_inds  = np.ones((y.shape[1], n)).astype(int)
best_model_inds[:,0] = np.argmin(
                    np.array([np.linalg.lstsq(np.expand_dims(pred, 1), y)[0]
                    for pred in list(X.T)]), 0).T[:,0].astype(int)


temp_best_mod_ind = -np.ones((y.shape[1],1)).astype(int)

for n_pred in range(1,n):
    for ti, target in enumerate(y.T):
        best_residual = np.inf
        best_ind = -1
        for i, pred in enumerate(X.T):
            if i not in best_model_inds[ti,:]:
#                temp_pred = np.hstack((np.expand_dims(pred, 1), X[:, best_model_inds[ti, :n_pred]]))
                temp_pred = X[:, np.hstack((i, best_model_inds[ti, :n_pred]))]

                candidate_residual = np.linalg.lstsq(temp_pred, target)[1]
                
                if candidate_residual<best_residual:
                    best_residual = candidate_residual
                    best_ind = i
                    
        best_model_inds[ti,n_pred] = best_ind

