# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 18:30:31 2020

@author: pselt
"""

import pandas as pd
import requests



from http import cookiejar  # Python 2: import cookielib as cookiejar
class BlockAll(cookiejar.CookiePolicy):
    return_ok = set_ok = domain_return_ok = path_return_ok = lambda self, *args, **kwargs: False
    netscape = True
    rfc2965 = hide_cookie2 = False



s = requests.Session()
s.cookies.set_policy(BlockAll())

################################################
##### ##### ##### ##### ##### ##### 
# This gets the current records of solo winrates
# from wows-numbers.com for the NA server
# 
# As it has to go through 1500+ pages of website 
# and the API likely employs anti-bot measures 
# this may take a couple of hours, but you can let this run in the background

# url = 'https://na.wows-numbers.com/ranking/type,solo'


# headers = {
#   "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36",
#   "X-Requested-With": "XMLHttpRequest",
#   "Connection": "close"
# }

# r = s.get(url, headers=headers)
# my_header = ('Player', 'Battles', 'Win rate', 'PR', 'Avg. damage', 'Max. damage', 'Max. experience', 'Max. planes', 'rank')
# dfs = pd.read_html(r.text,skiprows=0)
# df = dfs[0]
# relevant_data = df[["Player","Battles","Win rate","PR","Avg. damage"]]
# relevant_data.to_csv(r'C:\Users\pselt\OneDrive\projects\wows_MM_simulations\playerbase2.csv')


# n_page = 1558

# for k in range(2,n_page+1,1):

#     new_url = url + '/?p=' + str(k)
#     r = s.get(new_url, headers=headers)
#     my_header = ('Player', 'Battles', 'Win rate', 'PR', 'Avg. damage', 'Max. damage', 'Max. experience', 'Max. planes', 'rank')
#     dfs = pd.read_html(r.text,index_col=0,skiprows=0)
#     df = dfs[0]
#     relevant_data= df[["Player","Battles","Win rate","PR","Avg. damage"]]
#     relevant_data.to_csv(r'C:\Users\pselt\OneDrive\projects\wows_MM_simulations\playerbase2.csv',mode='a',header=False)
#     print('we are at page '+str(k)+' / '+str(n_page))
#   ######################################  
# ################################################  

################################################
##### ##### ##### ##### ##### ##### 
# This gets the current records of a particular ship
# from wows-numbers.com for the NA server
# 
# 
# 
# 



url=  'https://na.wows-numbers.com/ship/4277122768,Hakuryu-30-01-2019-/'




file=r'C:\Users\pselt\OneDrive\projects\wows_MM_simulations\ships_data\haku-old.csv'


headers = {
  "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36",
  "X-Requested-With": "XMLHttpRequest",
  "Connection": "close"
}

r = s.get(url, headers=headers)

dfs = pd.read_html(r.text)
df = dfs[-1]

relevant_data = df[["Battles","Win rate","PR","Avg. frags","Avg. damage"]]
relevant_data.to_csv(file)
print(relevant_data)

n_page = 6
for k in range(2,n_page+1,1):
    new_url = url + '/?p=' + str(k)
    r = s.get(new_url, headers=headers)
    dfs = pd.read_html(r.text,index_col=0,skiprows=0)
    df = dfs[-1]
    # print(df)
    relevant_data = df[["Battles","Win rate","PR","Avg. frags","Avg. damage"]]
    relevant_data.to_csv(file,mode='a',header=False)
    print('we are at page '+str(k)+' / '+str(n_page))
#####################################  
################################################    

