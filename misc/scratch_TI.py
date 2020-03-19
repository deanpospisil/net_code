# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 21:44:16 2016

@author: deanpospisil
"""
import matplotlib.pyplot as plt
import numpy as np

def measure_TIA_array(unit, m=1,n=1):
    error_space = m*n - m 
    effect_space = m*n - error_space
    s = np.linalg.svd(unit, compute_uv=False)
    numerator = (s[0]**2)/effect_space
    denominator = sum(s[1:]**2)/(error_space)
    return numerator/denominator

n_s = np.arange(5,100)
m=30
iterations = range(100)
for pc in [0,1]:
    F_mean = [];
    for m in n_s:
        Fs=[]
        for ind in iterations:
            mat = np.random.normal(size=(m,n))
            mat -= np.mean(mat, 0)
    
            constant = 0
            df_reg = n
            error_space = m*n - df_reg - constant
            effect_space = m*n - error_space       
            
            if pc==1:            
                i_variable = np.random.normal(size=(m,1))
            elif pc==0:
                i_variable = np.linalg.svd(mat, compute_uv=True)[0][:,0].reshape(30,1)
            else:
                i_variable = np.random.normal(size=(m,1))
                error_space = 1
                effect_space = 1 
            i_variable -= np.mean(i_variable)
            
            x = np.linalg.lstsq(i_variable, mat)[0]
            b_hat = np.dot(i_variable, x)
            m_y = np.sum(b_hat**2) / effect_space
            m_e = np.sum((b_hat - mat)**2) / error_space
            Fs.append(m_y / m_e)
        F_mean.append(np.mean(Fs))
    
    plt.plot(n_s, F_mean)
    
plt.legend(['PC','IID normal', 'PC unscaled'], title='Independent Variable')
plt.xlabel('n rows')
plt.ylabel('F-value')
'''

n_s = np.arange(5,20)
m=30
iterations = range(100)
for pc in [0,1]:
    F_mean = [];
    for n in n_s:
        Fs=[]
        print(n)
        for ind in iterations:
            mat = np.random.normal(size=(m,n))
            mat -= np.mean(mat, 0)
    
            constant = 0
            df_reg = n
            error_space = m*n - df_reg - constant
            effect_space = m*n - error_space       
            
            if pc:            
                i_variable = np.random.normal(size=(m,1))
            else:
                i_variable = np.linalg.svd(mat, compute_uv=True)[0][:,0].reshape(30,1)
            i_variable -= np.mean(i_variable)
            
            x = np.linalg.lstsq(i_variable, mat)[0]
            b_hat = np.dot(i_variable, x)
            m_y = np.sum(b_hat**2) / effect_space
            m_e = np.sum((b_hat - mat)**2) / error_space
            Fs.append(m_y / m_e)
        F_mean.append(np.mean(Fs))
    
    plt.plot(n_s, F_mean)
plt.legend(['PC','IID normal'], title='Dependent Variable')
plt.xlabel('n rows')
plt.ylabel('F-value')


n_s = np.arange(3,20)
m=30
iterations = range(2000)
tins_s = [];
for n in n_s:
    temp=[]
    print(n)
    for ind in iterations:
        mat = np.random.normal(size=(m,n))
        mat -= np.mean(mat, 0)
        
        constant = 0
        df_reg = n
        error_space = m*n - df_reg - constant
        effect_space = m*n - error_space       
        
        i_variable = np.random.normal(size=(m,1))
        i_variable -= np.mean(i_variable)
        
        x = np.linalg.lstsq(i_variable, mat)[0]
        b_hat = np.dot(i_variable, x)
        m_y = np.sum(b_hat**2)/effect_space
        m_e = np.sum((b_hat-mat)**2)/error_space
        temp.append(m_y/m_e)
    tins_s.append(np.mean(temp))

plt.plot(n_s, tins_s)

m=25.
iterations = range(5000)
temp = []
e_s = []
b_hats_s = []
for ind in iterations:
    d_variable = np.random.normal(size=(m,1))
    d_variable -= np.mean(d_variable)
    
    constant = 0
    df_reg = 1
    error_space = m - df_reg - constant
    effect_space = m - error_space       
    
    i_variable = np.random.normal(size=(m,1))
    i_variable -= np.mean(i_variable)
    #i_variable = np.concatenate((i_variable, np.ones((m,1))),1)
    
    x = np.linalg.lstsq(i_variable, d_variable)[0]
    b_hat = np.dot(i_variable, x)
    m_y = np.sum(b_hat**2)/effect_space
    m_e = np.sum((b_hat-d_variable)**2)/error_space
    temp.append(m_y/m_e)
    b_hats_s.append(np.sum(b_hat**2))
    e_s.append(np.sum((b_hat-d_variable)**2))

plt.plot(iterations, temp)
print(np.mean(temp))
print(np.mean(e_s))
print(np.mean(b_hats_s))


'''