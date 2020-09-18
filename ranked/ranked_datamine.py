# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 21:34:43 2020

@author: pselt
"""


import pandas as pd
import requests
import bs4 as bs
import numpy as np
from datetime import datetime


from http import cookiejar  # Python 2: import cookielib as cookiejar
class BlockAll(cookiejar.CookiePolicy):
    return_ok = set_ok = domain_return_ok = path_return_ok = lambda self, *args, **kwargs: False
    netscape = True
    rfc2965 = hide_cookie2 = False



s = requests.Session()
s.cookies.set_policy(BlockAll())


url = 'https://na.wows-numbers.com/season/18,The-Eighteenth-Season/'


today= datetime.now()
file=r'C:\Users\pselt\OneDrive\projects\wows_MM_simulations\ranked\ranked_players_season_18-'+str(today.strftime("%b-%d-%Y-%H-%M-%S"))+'.csv'


headers = {
  "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36",
  "X-Requested-With": "XMLHttpRequest",
  "Connection": "close"
}

r = s.get(url, headers=headers)


dfs = pd.read_html(r.text)
df = dfs[-1]
# print(df["Player"][0])
relevant_data = df[["Player","Battles","Rank","Win rate","Avg. frags","Avg. damage"]]
relevant_data.to_csv(file)
print(relevant_data)

n_page = 9
for k in range(2,n_page+1,1):
    new_url = url + '/?p=' + str(k)
    r = s.get(new_url, headers=headers)
    dfs = pd.read_html(r.text,index_col=0,skiprows=0)
    df = dfs[-1]
    # print(df)
    relevant_data = df[["Player","Battles","Rank","Win rate","Avg. frags","Avg. damage"]]
    relevant_data.to_csv(file,mode='a',header=False)
    print('we are at page '+str(k)+' / '+str(n_page))
####################################  
###############################################    

