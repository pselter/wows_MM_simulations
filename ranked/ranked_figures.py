# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 00:28:42 2020

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

dtypes = {'Player': 'str', 'Battles_R': 'int', 'Rank': 'int', 'Winrate_R': 'float', 'Avg.frags_R':'float','Avg.damage_R':'float'}
dtypes2 = {'Player': 'str', 'Battles': 'int',  'Winrate': 'float','PR': 'float', 'Avg.frags':'float','Avg.damage':'float'}

df_random = pd.read_csv(r'C:\Users\pselt\OneDrive\projects\wows_MM_simulations\ranked\randoms.csv',header=0,dtype=dtypes2)
df_ranked= pd.read_csv(r'C:\Users\pselt\OneDrive\projects\wows_MM_simulations\ranked\ranked.csv',header=0,dtype=dtypes)

# print(df)

dfs_ranked=df_ranked.sort_values(['Player'],ascending=True)
dfs_random=df_random.sort_values(['Player'],ascending=True)


# test = df_random.loc[df_random['Player'] == 'DerKleineNA', 'Winrate'].iloc[0]
# print(test)

 
df_combine = dfs_ranked.merge(dfs_random,how='left',on='Player')
# print(testdf)
df_combine.to_csv('test_test.csv')
# N_players = 10

test = df_ranked.loc[df_ranked['Player'] == '007_MD', 'Winrate_R'].iloc[0]
# print(test)
test2 = df_combine.loc[df_combine['Player'] == '007_MD', 'Winrate'].iloc[0]
# print(test2)
# print(df_random.iloc[0]['Player'])

dfs_combine = df_combine.sort_values(['Battles_R'],ascending=True)


print('##################')
print('Median win rate is:')
print(statistics.median(dfs_combine['Winrate']))

print('mean win rate is:')
print(np.nanmean(dfs_combine['Winrate']))



print('########')
print('total number of battles is')
print(sum(dfs_combine['Battles_R']))
print('Median battles is:')
print(statistics.median(dfs_combine['Battles_R']))
# med_wr = statistics.median(dfs['battles'])
print('mean battles is:')
print(np.nanmean(dfs_combine['Battles_R']))
# avg_wr = statistics.mean(dfs['battles'])

print('##################')

###############################################################################
dfs_combine = df_combine.sort_values(['Battles_R'],ascending=True)
fig = plt.figure(figsize=(6,4),dpi=300)

levels = np.linspace(1,16,16)
print(levels)
new_1= np.array(dfs_combine['Rank'])
data, bins = np.histogram(new_1,levels)
binscenters = np.array([levels[i] for i in range(len(bins)-1)])
plt.bar(binscenters, data, width=bins[1] - bins[0], color='navy', label=r'Histogram entries',alpha=1)
plt.xticks(np.linspace(1,16,16))
plt.title('Rank Distribution',fontdict=font)
ax = plt.gca()
ax.set_ylabel('Players', fontdict=font)
ax.set_xlabel('Current Rank', fontdict=font)
plt.show()
fig.savefig('Rank_distribution.png',dpi=100, format='png',bbox_inches='tight', pad_inches=0.1) 
plt.close()
###############################################################################

###############################################################################
dfs_combine = df_combine.sort_values(['Battles_R'],ascending=True)
fig = plt.figure(figsize=(6,4),dpi=300)

def g_peak(x,x0,sigma,I):
    g=(x-x0)**2/(2*sigma**2)
    return I*np.exp(-1*g)

levels = np.arange(30,80,1)
print(levels)
new_1= np.array(dfs_combine['Winrate'])
data, bins = np.histogram(new_1,levels)
binscenters = np.array([levels[i] for i in range(len(bins)-1)])
plt.bar(binscenters, data, width=bins[1] - bins[0], color='navy', label=r'Histogram entries',alpha=1)

popt, pcov = curve_fit(g_peak, xdata=binscenters, ydata=data, p0=[50, 5.0, 15000])
xspace = np.linspace(20, 80, 10000)

plt.plot(xspace, g_peak(xspace, *popt), color='darkorange', linewidth=2.5, label=r'Fitted function')
print('fitted center is:')
print(str(popt[0])+u" \u00B1 "+str(pcov[0,0]))
print('fitted sigma is:')
print(str(popt[1])+u" \u00B1 "+str(pcov[1,1]))

plt.xticks(np.arange(30,80,10))
plt.title('Account Winrate Distribution',fontdict=font)
ax = plt.gca()
ax.set_ylabel('Players', fontdict=font)
ax.set_xlabel('Account Winrate', fontdict=font)
plt.show()
fig.savefig('WR_distribution_of_players.png',dpi=100, format='png',bbox_inches='tight', pad_inches=0.1) 
plt.close()
###############################################################################


###############################################################################
dfs_combine = df_combine.sort_values(['Battles_R'],ascending=True)

fig = plt.figure(figsize=(6,4),dpi=300)


colors = np.array(dfs_combine['Battles_R'])
plt.scatter(dfs_combine['Winrate'],dfs_combine['Winrate_R'],cmap='jet',c=colors,alpha=1,s=2,norm=matplotlib.colors.LogNorm(),vmax=500)
# plt.hlines(avg_dam,xmin=min(dfs['win rate']),xmax=max(dfs['win rate']))
plt.vlines(50,0,100,linestyle='dashed')
plt.hlines(50,0,100,linestyle='dashed')
plt.xlim(30,80)
# plt.xscale('log')
# plt.yscale('log')
plt.colorbar(label='ranked battles')
plt.xticks(np.arange(30, 90, 10))
xdim = np.linspace(0,100,100)
plt.plot(xdim,xdim,color='k',linestyle='--')
ax = plt.gca()
ax.set_facecolor('grey')
ax.set_ylabel('Winrate in Ranked', fontdict=font)
ax.set_xlabel('Winrate Account', fontdict=font)
plt.annotate("Briney \nTriangle", (75,55),backgroundcolor='w',horizontalalignment='center')
plt.annotate("Salt quadrant", (65,25),backgroundcolor='w',horizontalalignment='center')
# plt.tight_layout()
# plt.show()
plt.show()
fig.savefig('Ranked_WR_vs_WR.png',dpi=100, format='png',bbox_inches='tight', pad_inches=0.1) 
plt.close()
###############################################################################


###############################################################################
dfs_combine = df_combine.sort_values(['Battles_R'],ascending=True)

fig = plt.figure(figsize=(6,4),dpi=300)


colors = np.array(dfs_combine['Battles_R'])
plt.scatter(dfs_combine['Winrate_R'],dfs_combine['Battles'],cmap='jet',c=colors,alpha=1,s=1,norm=matplotlib.colors.LogNorm(),vmax=500)
# plt.hlines(avg_dam,xmin=min(dfs['win rate']),xmax=max(dfs['win rate']))
# plt.vlines(50,0,100,linestyle='dashed')
plt.xlim(30,80)
# plt.xscale('log')
plt.yscale('log')
plt.colorbar(label='ranked battles')
plt.xticks(np.arange(30, 90, 10))
xdim = np.linspace(0,100,100)
# plt.plot(xdim,xdim,color='k',linestyle='--')
ax = plt.gca()
ax.set_facecolor('grey')
ax.set_ylabel('Battles in random', fontdict=font)
ax.set_xlabel('Winrate in Ranked', fontdict=font)
# plt.tight_layout()
# plt.show()
plt.show()
fig.savefig('Ranked_WR_vs_acc_battles.png',dpi=100, format='png',bbox_inches='tight', pad_inches=0.1) 
plt.close()
###############################################################################



## ###############################################################################
## dfs_combine = df_combine.sort_values(['Battles_R'],ascending=True)
##
## fig = plt.figure(figsize=(6,4),dpi=300)

## colors = np.array(dfs_combine.loc[dfs_combine['Rank'] == 10,'Battles_R'].iloc[:])
## new_1= np.array(dfs_combine.loc[dfs_combine['Rank'] == 10,'Winrate'].iloc[:])
## new_2= np.array(dfs_combine.loc[dfs_combine['Rank'] == 10,'Winrate_R'].iloc[:])
               

## plt.scatter(new_1,new_2,cmap='jet',alpha=1,s=2,c=colors,norm=matplotlib.colors.LogNorm(),vmax=max(dfs_combine['Battles_R']))
## # plt.hlines(avg_dam,xmin=min(dfs['win rate']),xmax=max(dfs['win rate']))
## plt.vlines(50,0,100,linestyle='dashed')
## plt.xlim(30,80)
## # plt.xscale('log')
## # plt.yscale('log')
## plt.colorbar(label='ranked battles')
## plt.xticks(np.arange(30, 90, 10))
## xdim = np.linspace(0,100,100)
## plt.plot(xdim,xdim,color='k',linestyle='--')
## ax = plt.gca()
## ax.set_facecolor('grey')
## ax.set_ylabel('Winrate in Ranked', fontdict=font)
## ax.set_xlabel('Winrate Account', fontdict=font)
## # plt.tight_layout()
## # plt.show()
## plt.show()
## # fig.savefig(name+'_damage_vs_WR.png',dpi=100, format='png',bbox_inches='tight', pad_inches=0.1) 
## plt.close()
## ###############################################################################


###############################################################################
dfs_combine = df_combine.sort_values(['Battles_R'],ascending=True)

fig = plt.figure(figsize=(6,4),dpi=300)

colors = np.array(dfs_combine.loc[dfs_combine['Battles'].isnull(),'Battles_R'].iloc[:])
new_1= np.array(dfs_combine.loc[dfs_combine['Battles'].isnull(),'Winrate_R'].iloc[:])
new_2= np.array(dfs_combine.loc[dfs_combine['Battles'].isnull(),'Rank'].iloc[:])
print(len(new_2))

plt.scatter(new_2,new_1,cmap='jet',alpha=1,s=10,c=colors,norm=matplotlib.colors.LogNorm(),vmax=max(dfs_combine['Battles_R']))
# plt.hlines(avg_dam,xmin=min(dfs['win rate']),xmax=max(dfs['win rate']))
plt.hlines(50,0,17,linestyle='dashed')
plt.xlim(17,0)
# plt.xscale('log')
# plt.yscale('log')
plt.title('Ranked Players with less than 100 random solo games',fontdict=font)
plt.colorbar(label='ranked battles')
plt.xticks(np.arange(1, 17, 1))
xdim = np.linspace(0,100,100)
# plt.plot(xdim,xdim,color='k',linestyle='--')
ax = plt.gca()
ax.set_facecolor('grey')
ax.set_ylabel('Winrate in Ranked', fontdict=font)
ax.set_xlabel('Rank', fontdict=font)
# plt.tight_layout()
# plt.show()
plt.show()
fig.savefig('ranked_non_randoms.png',dpi=100, format='png',bbox_inches='tight', pad_inches=0.1) 
plt.close()
###############################################################################

###############################################################################
dfs_combine = df_combine.sort_values(['Battles_R'],ascending=True)

fig = plt.figure(figsize=(8,3),dpi=200)
colors = np.array(dfs_combine['Battles_R'])

plt.scatter(dfs_combine['Rank'],dfs_combine['Winrate'],alpha=0.5,s=25,marker='_',c=colors,cmap='jet',norm=matplotlib.colors.LogNorm(),vmax=max(dfs_combine['Battles_R']))
# plt.xscale('log')
plt.colorbar(label='Battles played')
# plt.yscale('log')
# plt.colorbar()
plt.xlim(17,0)
plt.xticks(np.arange(0, 17, 1))
ax = plt.gca()
ax.set_facecolor('grey')
ax.set_ylabel('Account WR', fontdict=font)
ax.set_xlabel('Rank', fontdict=font)
# plt.tight_layout()
# plt.show()
plt.show()
fig.savefig('Rank_vs_WR.png',dpi=100, format='png',bbox_inches='tight', pad_inches=0.1) 
plt.close()
###############################################################################



###############################################################################
dfs_combine = df_combine.sort_values(['Winrate'],ascending=True)

fig = plt.figure(figsize=(8,6),dpi=200)

grid = plt.GridSpec(6, 8, wspace=1, hspace=2)

colors = np.array(dfs_combine['Winrate'])

plt.subplot(grid[0:3, 0:])
plt.scatter(dfs_combine['Rank'],dfs_combine['Battles_R'],alpha=1,s=25,c=colors,cmap='jet',marker='_')
# plt.xscale('log')
plt.colorbar(label='Account winrate')
# plt.yscale('log')

plt.xlim(17,0)
plt.xticks(np.arange(0, 17, 1))
ax = plt.gca()
ax.set_facecolor('grey')
ax.set_ylabel('Battles in Ranked', fontdict=font)
ax.set_xlabel('Rank', fontdict=font)
# plt.tight_layout()
# plt.show()

plt.subplot(grid[3:, 0:])
plt.scatter(dfs_combine['Rank'],dfs_combine['Battles_R'],alpha=1,s=25,c=colors,cmap='jet',marker='_')
# plt.xscale('log')
plt.colorbar(label='Account winrate')
plt.yscale('log')

plt.xlim(17,0)
plt.xticks(np.arange(0, 17, 1))
ax = plt.gca()
ax.set_facecolor('grey')
ax.set_ylabel('Battles in Ranked', fontdict=font)
ax.set_xlabel('Rank', fontdict=font)



plt.show()
fig.savefig('Rank_vs_battles_played_in_ranked.png',dpi=100, format='png',bbox_inches='tight', pad_inches=0.1) 
plt.close()
###############################################################################


###############################################################################


fig = plt.figure(figsize=(32,8),dpi=200)
grid = plt.GridSpec(4, 8, wspace=0.3, hspace=0.6)

levels = np.arange(30,80,1)

for n in range(0,4,1):
    plt.subplot(grid[0, n:n+1])
    new_1= np.array(dfs_combine.loc[dfs_combine['Rank'] == 16-n,'Winrate'].iloc[:])
    data, bins = np.histogram(new_1,levels)
    binscenters = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
    plt.bar(binscenters, data, width=bins[1] - bins[0], color='navy', label=r'Histogram entries',alpha=1)
    plt.xticks(np.arange(20, 90, 10))
    plt.title('Rank '+str(16-n),fontdict=font)
    ax = plt.gca()
    ax.set_ylabel('Players', fontdict=font)
    # ax.set_xlabel('Account WR', fontdict=font)
    
for n in range(4,8,1):
    plt.subplot(grid[1, n-4:n-4+1])
    new_1= np.array(dfs_combine.loc[dfs_combine['Rank'] == 16-n,'Winrate'].iloc[:])
    data, bins = np.histogram(new_1,levels)
    binscenters = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
    plt.bar(binscenters, data, width=bins[1] - bins[0], color='navy', label=r'Histogram entries',alpha=1)
    plt.xticks(np.arange(20, 90, 10))
    plt.title('Rank '+str(16-n),fontdict=font)
    ax = plt.gca()
    ax.set_ylabel('Players', fontdict=font)
    # ax.set_xlabel('Account WR', fontdict=font)

for n in range(8,12,1):
    plt.subplot(grid[2, n-8:n-8+1])
    new_1= np.array(dfs_combine.loc[dfs_combine['Rank'] == 16-n,'Winrate'].iloc[:])
    data, bins = np.histogram(new_1,levels)
    binscenters = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
    plt.bar(binscenters, data, width=bins[1] - bins[0], color='navy', label=r'Histogram entries',alpha=1)
    plt.xticks(np.arange(20, 90, 10))
    plt.title('Rank '+str(16-n),fontdict=font)
    ax = plt.gca()
    ax.set_ylabel('Players', fontdict=font)
    # ax.set_xlabel('Account WR', fontdict=font)

for n in range(12,16,1):
    plt.subplot(grid[3, n-12:n-12+1])
    new_1= np.array(dfs_combine.loc[dfs_combine['Rank'] == 16-n,'Winrate'].iloc[:])
    data, bins = np.histogram(new_1,levels)
    binscenters = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
    plt.bar(binscenters, data, width=bins[1] - bins[0], color='navy', label=r'Histogram entries',alpha=1)
    plt.xticks(np.arange(20, 90, 10))
    plt.title('Rank '+str(16-n),fontdict=font)
    ax = plt.gca()
    ax.set_ylabel('Players', fontdict=font)
    ax.set_xlabel('Account WR', fontdict=font)
    
    
plt.xticks(np.arange(20, 90, 10))

plt.show()
fig.savefig('ranked.png',dpi=100, format='png',bbox_inches='tight', pad_inches=0.1) 
plt.close()
###############################################################################





###############################################################################
dfs_combine = df_combine.sort_values(['Battles_R'],ascending=True)

fig = plt.figure(figsize=(32,12),dpi=200)
grid = plt.GridSpec(4, 8, wspace=0.3, hspace=0.6)

levels = np.arange(30,80,1)

for n in range(0,4,1):
    plt.subplot(grid[0, n:n+1])
    colors = np.array(dfs_combine.loc[dfs_combine['Rank'] == 16-n,'Battles_R'].iloc[:])
    new_1= np.array(dfs_combine.loc[dfs_combine['Rank'] == 16-n,'Winrate'].iloc[:])
    new_2= np.array(dfs_combine.loc[dfs_combine['Rank'] == 16-n,'Winrate_R'].iloc[:])
    
    
    
    plt.scatter(new_1,new_2,cmap='jet',alpha=1,s=2,c=colors,norm=matplotlib.colors.LogNorm(),vmax=max(dfs_combine['Battles_R']))
    plt.vlines(50,0,100,linestyle='dashed')
    plt.xlim(30,80)
    plt.colorbar()
    plt.xticks(np.arange(30, 90, 10))
    xdim = np.linspace(0,100,100)
    plt.plot(xdim,xdim,color='k',linestyle='--')
    plt.hlines(50,0,100,linestyle='dashed')
    ax = plt.gca()
    ax.set_facecolor('grey')
    ax.set_ylabel('Ranked WR', fontdict=font)
    # ax.set_xlabel('Winrate Account', fontdict=font) 
        
    plt.title('Rank '+str(16-n),fontdict=font)
    
    
    ax = plt.gca()
    # ax.set_ylabel('Players', fontdict=font)
    # ax.set_xlabel('Account WR', fontdict=font)
    
for n in range(4,8,1):
    plt.subplot(grid[1, n-4:n-4+1])
    colors = np.array(dfs_combine.loc[dfs_combine['Rank'] == 16-n,'Battles_R'].iloc[:])
    new_1= np.array(dfs_combine.loc[dfs_combine['Rank'] == 16-n,'Winrate'].iloc[:])
    new_2= np.array(dfs_combine.loc[dfs_combine['Rank'] == 16-n,'Winrate_R'].iloc[:])
    
    
    
    plt.scatter(new_1,new_2,cmap='jet',alpha=1,s=2,c=colors,norm=matplotlib.colors.LogNorm(),vmax=max(dfs_combine['Battles_R']))
    plt.vlines(50,0,100,linestyle='dashed')
    plt.xlim(30,80)
    plt.colorbar()
    plt.xticks(np.arange(30, 90, 10))
    xdim = np.linspace(0,100,100)
    plt.plot(xdim,xdim,color='k',linestyle='--')
    plt.hlines(50,0,100,linestyle='dashed')
    ax = plt.gca()
    ax.set_facecolor('grey')
    ax.set_ylabel('Ranked WR', fontdict=font)
    # ax.set_xlabel('Winrate Account', fontdict=font) 
        
    plt.title('Rank '+str(16-n),fontdict=font)
    
    
    ax = plt.gca()
    # ax.set_ylabel('Players', fontdict=font)
    # ax.set_xlabel('Account WR', fontdict=font)

for n in range(8,12,1):
    plt.subplot(grid[2, n-8:n-8+1])
    colors = np.array(dfs_combine.loc[dfs_combine['Rank'] == 16-n,'Battles_R'].iloc[:])
    new_1= np.array(dfs_combine.loc[dfs_combine['Rank'] == 16-n,'Winrate'].iloc[:])
    new_2= np.array(dfs_combine.loc[dfs_combine['Rank'] == 16-n,'Winrate_R'].iloc[:])
    
    
    
    plt.scatter(new_1,new_2,cmap='jet',alpha=1,s=2,c=colors,norm=matplotlib.colors.LogNorm(),vmax=max(dfs_combine['Battles_R']))
    plt.vlines(50,0,100,linestyle='dashed')
    plt.xlim(30,80)
    plt.colorbar()
    plt.xticks(np.arange(30, 90, 10))
    xdim = np.linspace(0,100,100)
    plt.plot(xdim,xdim,color='k',linestyle='--')
    plt.hlines(50,0,100,linestyle='dashed')
    ax = plt.gca()
    ax.set_facecolor('grey')
    ax.set_ylabel('Ranked WR', fontdict=font)
    # ax.set_xlabel('Winrate Account', fontdict=font) 
        
    plt.title('Rank '+str(16-n),fontdict=font)
    
    
    ax = plt.gca()
    # ax.set_ylabel('Players', fontdict=font)
    # ax.set_xlabel('Account WR', fontdict=font)

for n in range(12,16,1):
    plt.subplot(grid[3, n-12:n-12+1])
    colors = np.array(dfs_combine.loc[dfs_combine['Rank'] == 16-n,'Battles_R'].iloc[:])
    new_1= np.array(dfs_combine.loc[dfs_combine['Rank'] == 16-n,'Winrate'].iloc[:])
    new_2= np.array(dfs_combine.loc[dfs_combine['Rank'] == 16-n,'Winrate_R'].iloc[:])
    
    
    
    plt.scatter(new_1,new_2,cmap='jet',alpha=1,s=2,c=colors,norm=matplotlib.colors.LogNorm(),vmax=max(dfs_combine['Battles_R']))
    plt.vlines(50,0,100,linestyle='dashed')
    plt.xlim(30,80)
    plt.colorbar()
    plt.xticks(np.arange(30, 90, 10))
    xdim = np.linspace(0,100,100)
    plt.plot(xdim,xdim,color='k',linestyle='--')
    plt.hlines(50,0,100,linestyle='dashed')
    ax = plt.gca()
    ax.set_facecolor('grey')
    ax.set_ylabel('Ranked WR', fontdict=font)
    # ax.set_xlabel('Winrate Account', fontdict=font) 
        
    plt.title('Rank '+str(16-n),fontdict=font)
    
    
    ax = plt.gca()
    # ax.set_ylabel('Players', fontdict=font)
    ax.set_xlabel('Account WR', fontdict=font)
    
    
plt.xticks(np.arange(20, 90, 10))

plt.show()
fig.savefig('ranked2.png',dpi=100, format='png',bbox_inches='tight', pad_inches=0.1) 
plt.close()
###############################################################################

