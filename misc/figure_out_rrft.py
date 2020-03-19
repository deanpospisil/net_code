# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 11:04:54 2016

@author: deanpospisil
"""
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# define a function for calculating the t-statistic
def t_stat(x1,x2):
    n=len(x1)
    mean_dif = (np.mean(x1)-np.mean(x2))
    pooled_var = (((np.var(x1, ddof=1) + np.var(x2, ddof=1))/2)**0.5)
    se = pooled_var*(2/n)**0.5
    return np.abs(mean_dif/se)
#set my parameters of the null distribution
n = 10
mu = 1
sd = 1
iters = 10
p_list = []
for i in range(iters):
    #get my two samples from the null dist
    sample1 = np.random.normal(loc=mu, scale=sd, size=n)
    sample2 = np.random.normal(loc=mu, scale=sd, size=n)
    #calc t_stat
    t = t_stat(sample1, sample2)
    #look up its p-value
    p = 2*(1-(stats.t.cdf(t, df=2*(n-1))))
    p_list.append(p)

p_vals = np.array(p_list)    
_ = plt.hist(p_vals, bins=100)
plt.xlabel('p-value')
plt.ylabel('count')
plt.title('Distribution of p-values')

print('at p<.05 I reject the null hypothesis '+ 
      str(sum(p_vals<.05)/iters) + ' percent of the time')
print('at p<0.1 I reject the null hypothesis '+
      str(sum(p_vals<.1)/iters) + ' percent of the time')

def t_test_iters(n, mu, sd, iters=1000):
    p_list = []
    for i in range(iters):
        #get my two samples from the null dist
        sample1 = np.random.normal(loc=mu, scale=sd, size=n)
        sample2 = np.random.normal(loc=mu, scale=sd, size=n)
        #calc t_stat
        t = t_stat(sample1, sample2)
        #look up its p-value
        p = 2*(1-(stats.t.cdf(t, df=2*(n-1))))
        p_list.append(p)
    return np.array(p_list)
    
    
p_thresh = .05
iters = 10
mu = 0
sd = 1
#vary over n while keeping m
frac_sig = [sum(t_test_iters(n=n, mu=mu, sd=sd, iters=iters)<.05)
            for n in range(10,20)]

#%%
#set my parameters of the null distribution
n = 1000
mu = 1

iters = 10000
p_list = []
t_list = []
for i in range(iters):
    #get my two samples from the null dist
    sample1 = np.random.poisson(lam=mu, size=n)
    sample2 = np.random.poisson(lam=mu, size=n)
    #calc t_stat
    t = t_stat(sample1, sample2)
    t_list.append(t)
    #look up its p-value
    p = 2*(1-(stats.t.cdf(t, df=2*(n-1))))
    p_list.append(p)

p_vals = np.array(p_list)
plt.figure(figsize=(20,20))    
#_ = plt.hist(p_vals, bins=1000, cumulative=True)
_ = plt.scatter(range(len(p_vals)), p_vals, s=0.5,c='k',edgecolors='none')

plt.xlabel('p-value')
plt.ylabel('count')
plt.title('Distribution of p-values')

print('at p<.05 I reject the null hypothesis '+ 
      str(sum(p_vals<.05)/iters) + ' percent of the time')
print('at p<0.1 I reject the null hypothesis '+
      str(sum(p_vals<.1)/iters) + ' percent of the time')

#%%
a = np.random.poisson(lam=10,size=(5,1000))
b = np.random.poisson(lam=1,size=(50,1000))
plt.subplot(211)
plt.hist(np.sum(a,0))
plt.subplot(212)
plt.hist(np.sum(b,0))
#%%
# now lets figure out the cut off of.
def test_stat1(x,lam0,lam1):
    return (np.mean(x)-lam0)/(lam1-lam0)
def test_stat2(x,lam0,lam1):
    log_alt_prob = np.sum(np.log(stats.poisson.pmf(x, mu=lam1)))
    log_null_prob = np.sum(np.log(stats.poisson.pmf(x, mu=lam0)))
    return log_alt_prob-log_null_prob
def test_stat3(x,lam0,lam1):
    log_alt_prob = stats.poisson.pmf(stats.mode(x), mu=lam1)
    log_null_prob = stats.poisson.pmf(stats.mode(x), mu=lam0)
    return log_alt_prob-log_null_prob
lam1 = 6
lam0 = 5
n = 5

x=np.random.poisson(lam=lam0, size=(10000,n))
test_stat1_dist = np.array([test_stat1(samp, lam0, lam1) for samp in x])
stat1_crit = np.percentile(test_stat1_dist,95)

test_stat2_dist = np.array([test_stat2(samp, lam0, lam1) for samp in x])
stat2_crit = np.percentile(test_stat2_dist,95)

test_stat3_dist = np.array([test_stat3(samp, lam0, lam1) for samp in x])
stat3_crit = np.percentile(test_stat3_dist,95)

#%%
#tstat pars
def mean_dif(x1, x2):
    return np.mean(x1)-np.mean(x2)
def pool_var(x1, x2):
    n=len(x1)
    pooled_var = (((np.var(x1, ddof=1) + np.var(x2, ddof=1))/2)**0.5)
    se = pooled_var*((2/n)**0.5)
    return se
def p_val_t(t, n):
    p = 2*(1-(stats.t.cdf(t, df=2*(n-1))))
    return p

#numerator
n=5
mu=10
sample1 = np.random.poisson(lam=mu, size=n)
sample2 = np.random.poisson(lam=mu, size=n)


num = mean_dif(sample1, sample2)
den = pool_var(sample1, sample2)
t = num/den
p = p_val_t(t, n)

t_true, p_true = stats.ttest_ind(sample1, sample2, equal_var=True)

#denominator
