# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 15:02:28 2020

@author: pselt
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from scipy.optimize import curve_fit
import scipy
import scipy.stats
import statistics


font = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 10,
        }

font2 = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 14,
        }


######################################################
######################################################
#
#
# VISBY TEST
#
#


my_header = ('place', 'battles','win rate','PR','avg frags','avg damage')
name='desmoines'
df = pd.read_csv(r'C:\Users\pselt\OneDrive\projects\wows_MM_simulations\ships_data\\'+name+'.csv',dtype=float,header=0,names=my_header)
# print(df)

dfs=df.sort_values(['battles'],ascending=True)
print('####################################')
print('##################')

print(sum(dfs['battles']))

print('##################')
print('Median win rate is:')
print(statistics.median(dfs['win rate']))
med_wr = statistics.median(dfs['win rate'])
print('mean win rate is:')
print('')
print(statistics.mean(dfs['win rate']))
avg_wr = statistics.mean(dfs['win rate'])
print('########')
print('mean weighted win rate is:')
print(np.average(dfs['win rate'],weights=dfs['battles']))
print('')
print('########')
print('total number of battles is')
print(sum(dfs['battles']))
print('Median battles is:')
print(statistics.median(dfs['battles']))
# med_wr = statistics.median(dfs['battles'])
print('mean battles is:')
print(statistics.mean(dfs['battles']))
# avg_wr = statistics.mean(dfs['battles'])
print('')
print('########')
print('mean avg damage is:')
print(statistics.mean(dfs['avg damage']))
avg_dam = statistics.mean(dfs['avg damage'])
print('')
print('mean weighted average damage is:')
print(np.average(dfs['avg damage'],weights=dfs['battles']))
print('##################')
print('')
####################################################################


fig = plt.figure(figsize=(14,6),dpi=300)

grid = plt.GridSpec(21, 40, wspace=10, hspace=10)

plt.subplot(grid[0:, 0:20])
colors = np.array(dfs['battles'])
plt.scatter(dfs['win rate'],dfs['avg damage'],cmap='jet',c=colors,alpha=1,s=8,norm=matplotlib.colors.LogNorm(),vmax=1000)
plt.hlines(avg_dam,xmin=min(dfs['win rate']),xmax=max(dfs['win rate']))
plt.vlines(avg_wr,ymin=min(dfs['avg damage']),ymax=max(dfs['avg damage']))
plt.title(name,fontdict=font2)
# plt.xscale('log')
# plt.yscale('log')
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)

plt.xticks(np.arange(10, 100, 10))
ax = plt.gca()
ax.set_facecolor('grey')
ax.set_ylabel('Avg. damage', fontdict=font)
ax.set_xlabel('Player winrate in percent', fontdict=font)
# plt.tight_layout()
# plt.show()
# plt.show()
# fig.savefig(name+'_damage_vs_WR.png',dpi=100, format='png',bbox_inches='tight', pad_inches=0.1) 
# plt.close()



# fig = plt.figure(figsize=(5,3),dpi=200)
# colors = np.array(dfs['battles'])

plt.subplot(grid[0:9, 21:30])
plt.scatter(dfs['win rate'],dfs['battles'],alpha=0.5,s=2,color='navy')
# plt.xscale('log')
plt.vlines(avg_wr,ymin=min(dfs['battles']),ymax=max(dfs['battles']),color='red')
plt.yscale('log')
# plt.colorbar()
# plt.ylim(0,5000)
ax = plt.gca()
ax.set_facecolor('grey')
ax.set_ylabel('Battles played', fontdict=font)
ax.set_xlabel('Player winrate in percent', fontdict=font)
# plt.tight_layout()
# plt.show()
# plt.show()
# fig.savefig(name+'_battles_vs_WR.png',dpi=100, format='png',bbox_inches='tight', pad_inches=0.1) 
# plt.close()
# 



def g_peak(x,x0,sigma,I):
    g=(x-x0)**2/(2*sigma**2)
    return I*np.exp(-1*g)

def g2_peak(x,x0,sigma,I,x1,sigma1,I1):
    g=(x-x0)**2/(2*sigma**2)
    g2=(x-x1)**2/(2*sigma1**2)
    return I*np.exp(-1*g)+I1*np.exp(-1*g2)

# fig = plt.figure(figsize=(5,3),dpi=200)

plt.subplot(grid[0:9, 32:])

levels = np.arange(20,80,1)
data, bins = np.histogram(dfs['win rate'],levels)
binscenters = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
popt, pcov = curve_fit(g_peak, xdata=binscenters, ydata=data, p0=[50, 5.0, 15000])
xspace = np.linspace(20, 80, 10000)
plt.bar(binscenters, data, width=bins[1] - bins[0], color='navy', label=r'Histogram entries')
plt.plot(xspace, g_peak(xspace, *popt), color='darkorange', linewidth=2.5, label=r'Fitted function')

print('###########')
print('fitted center is:')
print(str(popt[0])+u" \u00B1 "+str(pcov[0,0]))
print('fitted sigma is:')
print(str(popt[1])+u" \u00B1 "+str(pcov[1,1]))

ax=plt.gca()
# plt.yscale('log')
# plt.xlim(45,55)
plt.title('Player win rate distribution \n' + '$\mu_0$ = '+str(round(popt[0],1)) + '%   $\sigma$ = '+str(round(popt[1],1))+'%', fontdict=font)
ax.set_xlabel('Win rate', fontdict=font)
ax.set_ylabel('Number of Players', fontdict=font)
# plt.tight_layout()
# plt.show()
# fig.savefig(name+'_WR_hist.png',dpi=100, format='png',bbox_inches='tight', pad_inches=0.1) 
# plt.close()

# fig = plt.figure(figsize=(5,3),dpi=200)

plt.subplot(grid[13:, 32:])

levels = np.arange(20,80,1)
data, bins = np.histogram(dfs['win rate'],levels,weights=dfs['battles'])
binscenters = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
popt, pcov = curve_fit(g_peak, xdata=binscenters, ydata=data, p0=[52, 5.0, 50000])
xspace = np.linspace(20, 80, 10000)
plt.bar(binscenters, data, width=bins[1] - bins[0], color='navy', label=r'Histogram entries')
plt.plot(xspace, g_peak(xspace, *popt), color='darkorange', linewidth=2.5, label=r'Fitted function')

print('#############')
print('fitted center is:')
print(str(popt[0])+u" \u00B1 "+str(pcov[0,0]))
print('fitted sigma is:')
print(str(popt[1])+u" \u00B1 "+str(pcov[1,1]))
ax=plt.gca()
# plt.yscale('log')
# plt.xlim(45,55)
# plt.title('Player win rate distribution \n weighted by battles', fontdict=font)
plt.title('Player win rate distribution \n weighted by battles \n' + '$\mu_0$ = '+str(round(popt[0],1)) + '%   $\sigma$ = '+str(round(popt[1],1))+'%', fontdict=font)
ax.set_xlabel('Win rate', fontdict=font)
ax.set_ylabel('Number of Battles', fontdict=font)
# plt.tight_layout()
# plt.show()
# fig.savefig(name+'_damage_vs_WR_weighted.png',dpi=100, format='png',bbox_inches='tight', pad_inches=0.1) 
# plt.close()


# fig = plt.figure(figsize=(4,4),dpi=200)

plt.subplot(grid[13:, 21:30])

levels = np.arange(80,2000,20)
data, bins = np.histogram(dfs['battles'],levels)
binscenters = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
# popt, pcov = curve_fit(g_peak, xdata=binscenters, ydata=data, p0=[50, 5.0, 15000])
# xspace = np.linspace(30, 70, 10000)
plt.bar(binscenters, data, width=bins[1] - bins[0], color='navy', label=r'Histogram entries')
# plt.plot(xspace, g_peak(xspace, *popt), color='darkorange', linewidth=2.5, label=r'Fitted function')
# print(*popt)
# plt.xscale('log')
plt.yscale('log')
ax=plt.gca()
ax.set_xlabel('Number of games', fontdict=font)
ax.set_ylabel('Number of Players', fontdict=font)
plt.title('Games per player', fontdict=font)
# plt.tight_layout()
plt.show()
fig.savefig(name+'_total.png',dpi=200, format='png',bbox_inches='tight', pad_inches=0.1) 
plt.close()
####################################################################

