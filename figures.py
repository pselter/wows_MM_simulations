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
        'size': 12,
        }



# df2 = pd.read_csv(r'C:\Users\pselt\OneDrive\projects\wows_MM_simulations\Weeklylystats_ships.csv',header=0,index_col=0)
# print(df2.dtypes)
# dfs2 = df2.sort_values(['tier'])
# print(dfs2)

# # ####################################################################
# # # Histograms for ships and tiers

# battles_per_tier = np.zeros(10)
# binscenters = np.arange(1,11,1)
# for n in range(1,11,1):
#     battles_per_tier[n-1]=dfs2['total battles'][dfs2['tier']==n].sum()

# print(battles_per_tier)
# total_battles = battles_per_tier.sum()
# battles_per_tier = battles_per_tier/total_battles*100


# fig = plt.figure(figsize=(4,6),dpi=200)

# plt.bar(binscenters, battles_per_tier, width=0.8, color='navy', label=r'Histogram entries')

# plt.title('Battles played per tier in percent', fontdict=font)
# plt.show()
# plt.close()


# n_CV = dfs2['total battles'][dfs2['class']=='CV'].sum()
# n_DD = dfs2['total battles'][dfs2['class']=='DD'].sum()
# n_BB = dfs2['total battles'][dfs2['class']=='BB'].sum()
# n_CA = dfs2['total battles'][dfs2['class']=='CA'].sum()


# n_ALL = n_CV+n_DD+n_BB+n_CA

# n_CV = n_CV/n_ALL
# n_DD = n_DD/n_ALL
# n_BB = n_BB/n_ALL
# n_CA = n_CA/n_ALL


# # Pie chart, where the slices will be ordered and plotted counter-clockwise:
# labels = 'CV', 'DD', 'BB', 'CA/CL'
# sizes = [n_CV, n_DD, n_BB, n_CA]
# explode = (0.1, 0.1, 0.1, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

# fig1, ax1 = plt.subplots()
# ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
#         shadow=False, startangle=90)
# ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# plt.show()
# fig.savefig('weekly_class_dist.png',dpi=300, format='png',bbox_inches='tight', pad_inches=0.1) 
# ###################################################################

# # ####################################################################
# # # Histograms for CVs per tier

# n_CV_4 = dfs2['total battles'][dfs2['class']=='CV'][dfs2['tier']==4].sum()
# n_CV_6 = dfs2['total battles'][dfs2['class']=='CV'][dfs2['tier']==6].sum()
# n_CV_8 = dfs2['total battles'][dfs2['class']=='CV'][dfs2['tier']==8].sum()
# n_CV_10= dfs2['total battles'][dfs2['class']=='CV'][dfs2['tier']==10].sum()

# print(n_CV_10)
# # Pie chart, where the slices will be ordered and plotted counter-clockwise:

# binscenters = np.arange(4,12,2)


# fig = plt.figure(figsize=(4,6),dpi=200)

# plt.bar(binscenters, (n_CV_4,n_CV_6,n_CV_8,n_CV_10), width=0.8, color='navy', label=r'Histogram entries')

# plt.title('absolute number of CV battles played in tier', fontdict=font)
# plt.show()
# fig.savefig('weekly_class_d.png',dpi=300,bbox_inches='tight', format='png', pad_inches=0.1) 
# plt.close()

# ###################################################################

# # ####################################################################
# # # Histograms for ships and tiers

# battles_per_tier = np.zeros(10)
# binscenters = np.arange(1,11,1)
# for n in range(1,11,1):
  





#     n_CV = dfs2['total battles'][dfs2['class']=='CV'][dfs2['tier']==n].sum()
#     n_DD = dfs2['total battles'][dfs2['class']=='DD'][dfs2['tier']==n].sum()
#     n_BB = dfs2['total battles'][dfs2['class']=='BB'][dfs2['tier']==n].sum()
#     n_CA = dfs2['total battles'][dfs2['class']=='CA'][dfs2['tier']==n].sum()
    
    
#     n_ALL = n_CV+n_DD+n_BB+n_CA
    
#     n_CV = n_CV/n_ALL
#     n_DD = n_DD/n_ALL
#     n_BB = n_BB/n_ALL
#     n_CA = n_CA/n_ALL
    
    
#     # Pie chart, where the slices will be ordered and plotted counter-clockwise:
#     labels = 'CV', 'DD', 'BB', 'CA/CL'
#     sizes = [n_CV, n_DD, n_BB, n_CA]
#     explode = (0.1, 0.1, 0.1, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
    
#     fig1, ax1 = plt.subplots()
#     plt.title('Tier '+str(n),fontdict=font)
#     ax1.pie(sizes, labels=labels, autopct='%1.1f%%',textprops=font,
#             shadow=False, startangle=90)
#     ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    
#     plt.show()
#     fig1.savefig('weekly_class_distribution_tier_'+str(n)+'.png',dpi=300, format='png', pad_inches=0.1) 
# ###################################################################




# my_header = ('place', 'battles','win rate','PR','avg damage')
# df = pd.read_csv(r'C:\Users\pselt\OneDrive\projects\wows_MM_simulations\playerbase.csv',dtype=float,header=0,names=my_header)
# # print(df.dtypes)

# dfs=df.sort_values(['battles'],ascending=True)
# print('##################')
# print('Median win rate is:')
# print(statistics.median(dfs['win rate']))
# med_wr = statistics.median(dfs['win rate'])
# print('mean win rate is:')
# print(statistics.mean(dfs['win rate']))
# avg_wr = statistics.mean(dfs['win rate'])
# print('########')
# print('total number of battles is')
# print(sum(dfs['battles']))
# print('Median battles is:')
# print(statistics.median(dfs['battles']))
# # med_wr = statistics.median(dfs['battles'])
# print('mean battles is:')
# print(statistics.mean(dfs['battles']))
# # avg_wr = statistics.mean(dfs['battles'])
# print('########')
# print('mean avg damage is:')
# print(statistics.mean(dfs['avg damage']))
# avg_dam = statistics.mean(dfs['avg damage'])
# print('##################')
####################################################################
# fig = plt.figure(figsize=(5,4),dpi=200)
# colors = np.array(dfs['battles'])
# plt.scatter(dfs['win rate'],dfs['PR'],c=colors,alpha=0.5,s=2,norm=matplotlib.colors.LogNorm())

# # plt.scatter(dfs['win rate'],dfs['PR'],c=colors,cmap='hsv',alpha=0.5,s=2)
# # plt.scatter(playernumbers,skill2,color='tab:red')
# # plt.xscale('log')
# plt.yscale('log')
# plt.colorbar()
# ax = plt.gca()
# ax.set_ylabel('Personal Rating / a.u.', fontdict=font)
# ax.set_xlabel('Player winrate in percent', fontdict=font)
# plt.tight_layout()
# # plt.show()
# plt.show()
# plt.close()

# fig = plt.figure(figsize=(5,4),dpi=200)
# colors = np.array(dfs['battles'])
# plt.scatter(dfs['win rate'],dfs['avg damage'],c=colors,alpha=0.5,s=2,norm=matplotlib.colors.LogNorm())
# # plt.scatter(dfs['win rate'],dfs['avg damage'],c=colors,cmap='hsv',alpha=0.5,s=2)
# # plt.scatter(playernumbers,skill2,color='tab:red')
# # plt.xscale('log')
# plt.hlines(avg_dam,xmin=min(dfs['win rate']),xmax=max(dfs['win rate']))
# plt.vlines(avg_wr,ymin=min(dfs['avg damage']),ymax=max(dfs['avg damage']))
# plt.yscale('log')
# plt.colorbar()
# ax = plt.gca()
# ax.set_ylabel('Avg. damage', fontdict=font)
# ax.set_xlabel('Player winrate in percent', fontdict=font)
# plt.tight_layout()
# # plt.show()
# plt.show()
# fig.savefig('WR_vs_damage.png',dpi=200, format='png',bbox_inches='tight', pad_inches=0.1) 
# plt.close()


# fig = plt.figure(figsize=(5,4),dpi=200)
# # colors = np.array(dfs['battles'])
# plt.scatter(dfs['win rate'],dfs['battles'],alpha=0.5,s=0.2)
# # plt.scatter(dfs['win rate'],dfs['avg damage'],c=colors,cmap='hsv',alpha=0.5,s=2)
# # plt.scatter(playernumbers,skill2,color='tab:red')
# # plt.xscale('log')
# plt.yscale('log')
# # plt.colorbar()
# # plt.ylim(0,5000)
# ax = plt.gca()
# ax.set_ylabel('Battles played', fontdict=font)
# ax.set_xlabel('Player winrate in percent', fontdict=font)
# plt.tight_layout()
# # plt.show()
# plt.show()
# fig.savefig('WR_vs_battles.png',dpi=200, format='png',bbox_inches='tight', pad_inches=0.1) 
# plt.close()
# ####################################################################


# ####################################################################
# # Histograms for playerbase

# def s_expo(x,a,I,b):
#     return I*np.exp(-a*x**b)

# def expo(x,a,I):
#     return I*np.exp(-a*x)

# def g_peak(x,x0,sigma,I):
#     g=(x-x0)**2/(2*sigma**2)
#     return I*np.exp(-1*g)

# def dual_expo(x,a1,I1,a2,I2,b2):
#     return I1*(np.exp(-a1*x))+I2*(np.exp(-a2*x**b2))


# fig = plt.figure(figsize=(3,3),dpi=100)

# levels = np.arange(30,70,1)
# data, bins = np.histogram(dfs['win rate'],levels)
# binscenters = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
# popt, pcov = curve_fit(g_peak, xdata=binscenters, ydata=data, p0=[50, 5.0, 15000])
# xspace = np.linspace(30, 70, 10000)
# plt.bar(binscenters, data, width=bins[1] - bins[0], color='navy', label=r'Histogram entries')
# plt.plot(xspace, g_peak(xspace, *popt), color='darkorange', linewidth=2.5, label=r'Fitted function')
# print('fitted center is:')
# print(str(popt[0])+u" \u00B1 "+str(pcov[0,0]))
# print('fitted sigma is:')
# print(str(popt[1])+u" \u00B1 "+str(pcov[1,1]))

# plt.title('Player win rate distribution', fontdict=font)

# ax=plt.gca()
# ax.set_ylabel('Players', fontdict=font)
# ax.set_xlabel('Win Rate', fontdict=font)
# fig.savefig('WR_distribution.png',dpi=100, format='png',bbox_inches='tight', pad_inches=0.1) 
# plt.show()
# plt.close()

# fig = plt.figure(figsize=(3,3),dpi=200)
# plt.bar(binscenters, data, width=bins[1] - bins[0], color='navy', label=r'Histogram entries')
# plt.plot(xspace, g_peak(xspace, *popt), color='darkorange', linewidth=2.5, label=r'Fitted function')
# plt.yscale('log')
# plt.title('Player win rate distribution \n logarithmic scale', fontdict=font)
# plt.show()
# plt.close()


# fig = plt.figure(figsize=(3,3),dpi=100)
# levels = np.arange(100,30000,100)
# data, bins = np.histogram(dfs['battles'],levels)
# binscenters = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])

# # popt, pcov = curve_fit(dual_expo, xdata=binscenters, ydata=data, p0=[0.008,1500,0.00008,2000,0.5])
# # xspace = np.linspace(100, 30000, 30000)
# plt.bar(binscenters, data, width=bins[1] - bins[0], color='navy', label=r'Histogram entries')
# # plt.plot(xspace, dual_expo(xspace, *popt), color='darkorange', linewidth=1, label=r'Fitted function')
# plt.title('Player battles distribution', fontdict=font)
# # plt.yscale('log')
# plt.xscale('log')
# ax=plt.gca()
# ax.set_ylabel('Players', fontdict=font)
# ax.set_xlabel('Number of battles', fontdict=font)
# plt.show()
# fig.savefig('Battle_log_distr.png',dpi=100, format='png',bbox_inches='tight', pad_inches=0.1) 
# plt.close()

# fig = plt.figure(figsize=(4,6),dpi=200)
# plt.bar(binscenters, data, width=bins[1] - bins[0], color='navy', label=r'Histogram entries')
# # plt.plot(xspace, dual_expo(xspace, *popt), color='darkorange', linewidth=1, label=r'Fitted function')
# plt.title('Player battles distribution', fontdict=font)
# plt.yscale('log')
# # plt.xscale('log')
# plt.show()
####################################################################



######################################################
######################################################
#
#
# VISBY TEST
#
#


my_header = ('place', 'battles','win rate','PR','avg frags','avg damage')
df = pd.read_csv(r'C:\Users\pselt\OneDrive\projects\wows_MM_simulations\midway_test.csv',dtype=float,header=0,names=my_header)
print(df)

dfs=df.sort_values(['battles'],ascending=True)

print(statistics.median(dfs['win rate']))
print(statistics.mean(dfs['win rate']))
print(sum(dfs['battles']))

print('##################')
print('Median win rate is:')
print(statistics.median(dfs['win rate']))
med_wr = statistics.median(dfs['win rate'])
print('mean win rate is:')
print(statistics.mean(dfs['win rate']))
avg_wr = statistics.mean(dfs['win rate'])
print('########')
print('total number of battles is')
print(sum(dfs['battles']))
print('Median battles is:')
print(statistics.median(dfs['battles']))
# med_wr = statistics.median(dfs['battles'])
print('mean battles is:')
print(statistics.mean(dfs['battles']))
# avg_wr = statistics.mean(dfs['battles'])
print('########')
print('mean avg damage is:')
print(statistics.mean(dfs['avg damage']))
avg_dam = statistics.mean(dfs['avg damage'])
print('##################')

####################################################################
fig = plt.figure(figsize=(5,4),dpi=200)
colors = np.array(dfs['battles'])
plt.scatter(dfs['win rate'],dfs['PR'],cmap='jet',c=colors,alpha=0.5,s=4,norm=matplotlib.colors.LogNorm())
# plt.scatter(dfs['win rate'],dfs['PR'],c=colors,cmap='hsv',alpha=0.5,s=2)
# plt.scatter(playernumbers,skill2,color='tab:red')
# plt.xscale('log')
# plt.yscale('log')
plt.colorbar()
ax = plt.gca()
ax.set_ylabel('Personal Rating / a.u.', fontdict=font)
ax.set_xlabel('Player winrate in percent', fontdict=font)
plt.tight_layout()
# plt.show()
plt.show()
plt.close()

fig = plt.figure(figsize=(5,4),dpi=200)
colors = np.array(dfs['battles'])
plt.scatter(dfs['win rate'],dfs['avg damage'],cmap='jet',c=colors,alpha=1,s=8,norm=matplotlib.colors.LogNorm(),vmax=1000)
plt.hlines(avg_dam,xmin=min(dfs['win rate']),xmax=max(dfs['win rate']))
plt.vlines(avg_wr,ymin=min(dfs['avg damage']),ymax=max(dfs['avg damage']))
# plt.scatter(dfs['win rate'],dfs['avg damage'],c=colors,cmap='hsv',alpha=0.5,s=2)
# plt.scatter(playernumbers,skill2,color='tab:red')
# plt.xscale('log')
# plt.yscale('log')
plt.colorbar()
ax = plt.gca()
ax.set_facecolor('grey')
ax.set_ylabel('Avg. damage', fontdict=font)
ax.set_xlabel('Player winrate in percent', fontdict=font)
plt.tight_layout()
# plt.show()
plt.show()
plt.close()


fig = plt.figure(figsize=(5,4),dpi=200)
colors = np.array(dfs['battles'])
plt.scatter(dfs['win rate'],dfs['avg frags'],cmap='jet',c=colors,alpha=0.5,s=4,norm=matplotlib.colors.LogNorm())

# plt.scatter(dfs['win rate'],dfs['avg damage'],c=colors,cmap='hsv',alpha=0.5,s=2)
# plt.scatter(playernumbers,skill2,color='tab:red')
# plt.xscale('log')
# plt.yscale('log')
plt.colorbar()
ax = plt.gca()
ax.set_ylabel('Avg. kills', fontdict=font)
ax.set_xlabel('Player winrate in percent', fontdict=font)
plt.tight_layout()
# plt.show()
plt.show()
plt.close()

fig = plt.figure(figsize=(5,3),dpi=200)
# colors = np.array(dfs['battles'])
plt.scatter(dfs['win rate'],dfs['battles'],alpha=0.5,s=4,color='navy')

# plt.scatter(dfs['win rate'],dfs['avg damage'],c=colors,cmap='hsv',alpha=0.5,s=2)
# plt.scatter(playernumbers,skill2,color='tab:red')
# plt.xscale('log')
plt.yscale('log')
# plt.colorbar()
# plt.ylim(0,5000)
ax = plt.gca()
ax.set_facecolor('grey')
ax.set_ylabel('Battles played', fontdict=font)
ax.set_xlabel('Player winrate in percent', fontdict=font)
plt.tight_layout()
# plt.show()
plt.show()
plt.close()

# fig = plt.figure(figsize=(5,3),dpi=200)
# # colors = np.array(dfs['battles'])
# plt.hist2d(dfs['win rate'],dfs['battles'],bins=50)
# # plt.scatter(dfs['win rate'],dfs['avg damage'],c=colors,cmap='hsv',alpha=0.5,s=2)
# # plt.scatter(playernumbers,skill2,color='tab:red')
# # plt.xscale('log')
# plt.yscale('log')
# # plt.colorbar()
# # plt.ylim(0,5000)
# ax = plt.gca()
# ax.set_ylabel('Battles played', fontdict=font)
# ax.set_xlabel('Player winrate in percent', fontdict=font)
# plt.tight_layout()
# # plt.show()
# plt.show()
# plt.close()



def g_peak(x,x0,sigma,I):
    g=(x-x0)**2/(2*sigma**2)
    return I*np.exp(-1*g)


fig = plt.figure(figsize=(5,3),dpi=200)

levels = np.arange(20,80,0.5)
data, bins = np.histogram(dfs['win rate'],levels)
binscenters = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
popt, pcov = curve_fit(g_peak, xdata=binscenters, ydata=data, p0=[50, 5.0, 15000])
xspace = np.linspace(20, 80, 10000)
plt.bar(binscenters, data, width=bins[1] - bins[0], color='navy', label=r'Histogram entries')
plt.plot(xspace, g_peak(xspace, *popt), color='darkorange', linewidth=2.5, label=r'Fitted function')
print('fitted center is:')
print(str(popt[0])+u" \u00B1 "+str(pcov[0,0]))
print('fitted sigma is:')
print(str(popt[1])+u" \u00B1 "+str(pcov[1,1]))
ax=plt.gca()
# plt.yscale('log')
# plt.xlim(45,55)
plt.title('Player win rate distribution', fontdict=font)
ax.set_xlabel('Win rate', fontdict=font)
ax.set_ylabel('Number of Players', fontdict=font)
plt.show()
plt.close()



fig = plt.figure(figsize=(4,4),dpi=200)

levels = np.arange(80,2000,20)
data, bins = np.histogram(dfs['battles'],levels)
binscenters = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
# popt, pcov = curve_fit(g_peak, xdata=binscenters, ydata=data, p0=[50, 5.0, 15000])
# xspace = np.linspace(30, 70, 10000)
plt.bar(binscenters, data, width=bins[1] - bins[0], color='navy', label=r'Histogram entries')
# plt.plot(xspace, g_peak(xspace, *popt), color='darkorange', linewidth=2.5, label=r'Fitted function')
# print(*popt)
plt.xscale('log')
plt.yscale('log')
ax=plt.gca()
ax.set_xlabel('Number of games', fontdict=font)
ax.set_ylabel('Number of Players', fontdict=font)
plt.title('Games per player', fontdict=font)
plt.show()
plt.close()
####################################################################

