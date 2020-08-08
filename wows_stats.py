# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 20:48:41 2020

@author: pselt
"""


import numpy as np
from matplotlib import pyplot as plt
import random



##############################################################################

class wows_model1(object):
    
    def __init__(self,N_players=24*10):
        
        self.N_players = N_players
        self.playernumbers = np.arange(0,self.N_players,1)
        self.players = np.arange(0,self.N_players,1)
        self.winrate_0 = np.zeros(self.N_players)
        self.playerbase = np.zeros((self.N_players,3))
        self.N_battles = int(self.N_players/24)
        
        
        print(' - - Playerbase object created - - ')


    def define_skill(self,average=50,sigma=10):
        self.average = average
        self.sigma = sigma
        self.skill = np.random.normal(scale=self.sigma,size=self.N_players)+self.average
        self.playerbase[:,0]=self.skill[:]
        print(' - - skill levels have been defined - - ')

####################################

####################################

# Below are cases for the case of equal distribution of games
# in other words for every iteration the whole playerbase is playing a game
# with randomly generated teams

    # ####################################
    # This is the prototypical NO RNG case meaning the outcome of a match
    # is entirely deterministic and based on the average skill of a team
    # THIS IS NOT HOW WOWS WORKS
    # JUST FOR COMPARISON 



    def gen_data_mod1_NO_RNG(self,n_iter):
        self.playerbase2 = np.copy(self.playerbase)
        self.WR_differences = np.zeros(n_iter*self.N_battles)
        for m in range(0,n_iter,1):
            # shuffles the playerlist
            np.random.shuffle(self.players)        
           
            # separate the playerlist into pieces of 24
            # this generates the battle lineups
            # ensuring everyone has am equal number of battles at the end
            
            self.battle_lineups = np.split(self.players,self.N_battles)
            
            ##########
            # This is the loop where all the games in one Matchmaking round are 'played'
            # The shuffled player list is evenly split into parts of 24 players
            # the first 12 are greens, the second 12 are reds
            
            
            for k in range(0,self.N_battles,1):
                
                # Determine the average skill of the players in the green portion    
                self.skill_green = 0
                for l in range(0,12,1):
                    self.current_player_skill = self.playerbase2[self.battle_lineups[k][l],0]
                    self.skill_green += self.current_player_skill/12
                    
                # Determine the average skill of the players in the red portion    
                self.skill_red = 0
                for l in range(12,24,1):
                    self.current_player_skill = self.playerbase2[self.battle_lineups[k][l],0]
                    self.skill_red += self.current_player_skill/12
               
                #######
                # The outcome of the battle is determined here and the list is updated
                # In this version it is entirely up to the skill of the players
                # This is the "NO RNG" case
                self.WR_differences[m*self.N_battles+k] = (self.skill_green-self.skill_red)
                
                if self.skill_green > self.skill_red:
                    for l in range(0,12,1):
                        self.playerbase2[self.battle_lineups[k][l],1] += 1
                        
                        
                if self.skill_green < self.skill_red:
                    for l in range(12,24,1):
                        self.playerbase2[self.battle_lineups[k][l],1] += 1
                        
                        
                if self.skill_green == self.skill_red:
                    for l in range(0,24,1):
                        self.playerbase2[self.battle_lineups[k][l],1] += 0.5
            
        return self.playerbase2,self.WR_differences
        
    #####################################
    # This is the second model WITH RNG 
    # Meaning more skill increases the likelihood of winning
    # there is a RNG parameter indicating how strong skill should influence the outcome



    def gen_data_mod1_RNG_1(self,n_iter,RNG):
        self.playerbase2 = np.copy(self.playerbase)
        self.WR_differences = np.zeros(n_iter*self.N_battles)
        for m in range(0,n_iter,1):
            # shuffles the playerlist
            np.random.shuffle(self.players)        
            
            # separate the playerlist into pieces of 24
            # this generates the battle lineups
            # ensuring everyone has am equal number of battles at the end
            
            self.battle_lineups = np.split(self.players,self.N_battles)
            
            ##########
            # This is the loop where all the games in one Matchmaking round are 'played'
            # The shuffled player list is evenly split into parts of 24 players
            # the first 12 are greens, the second 12 are reds
            
            
            for k in range(0,self.N_battles,1):
                
                # Determine the average skill of the players in the green portion    
                self.skill_green = 0
                for l in range(0,12,1):
                    self.current_player_skill = self.playerbase2[self.battle_lineups[k][l],0]
                    self.skill_green += self.current_player_skill/12
                    
                # Determine the average skill of the players in the red portion    
                self.skill_red = 0
                for l in range(12,24,1):
                    self.current_player_skill = self.playerbase2[self.battle_lineups[k][l],0]
                    self.skill_red += self.current_player_skill/12
               

                
                self.random = (np.random.random()*(2*RNG)-RNG)/12
                # self.WR_differences[m*self.N_battles+k] = self.random
                self.WR_differences[m*self.N_battles+k] = (self.skill_green+self.random -self.skill_red)
                
                if self.skill_green+self.random > self.skill_red:
                    for l in range(0,12,1):
                        self.playerbase2[self.battle_lineups[k][l],1] += 1
                        
                        
                if self.skill_green+self.random < self.skill_red:
                    for l in range(12,24,1):
                        self.playerbase2[self.battle_lineups[k][l],1] += 1
                        
                        
                if self.skill_green+self.random == self.skill_red:
                    for l in range(0,24,1):
                        self.playerbase2[self.battle_lineups[k][l],1] += 0.
        return self.playerbase2,self.WR_differences


    def gen_data_mod1_RNG_NORMAL(self,n_iter,RNG):
        self.playerbase2 = np.copy(self.playerbase)
        self.WR_differences = np.zeros(n_iter*self.N_battles)
        for m in range(0,n_iter,1):
            # shuffles the playerlist
            np.random.shuffle(self.players)        
            
            # separate the playerlist into pieces of 24
            # this generates the battle lineups
            # ensuring everyone has am equal number of battles at the end
            
            self.battle_lineups = np.split(self.players,self.N_battles)
            
            ##########
            # This is the loop where all the games in one Matchmaking round are 'played'
            # The shuffled player list is evenly split into parts of 24 players
            # the first 12 are greens, the second 12 are reds
            
            
            for k in range(0,self.N_battles,1):
                
                # Determine the average skill of the players in the green portion    
                self.skill_green = 0
                for l in range(0,12,1):
                    self.current_player_skill = self.playerbase2[self.battle_lineups[k][l],0]
                    self.skill_green += self.current_player_skill/12
                    
                # Determine the average skill of the players in the red portion    
                self.skill_red = 0
                for l in range(12,24,1):
                    self.current_player_skill = self.playerbase2[self.battle_lineups[k][l],0]
                    self.skill_red += self.current_player_skill/12
               
                #######
                # The outcome of the battle is determined here and the list is updated
                # In this version it is entirely up to the skill of the players
                # This is the "NO RNG" case
                
                self.random = (np.random.randn()*RNG)/12
                self.WR_differences[m*self.N_battles+k] = self.random
                # self.WR_differences[m*self.N_battles+k] = (self.skill_green+self.random -self.skill_red)
                
                if self.skill_green+self.random > self.skill_red:
                    for l in range(0,12,1):
                        self.playerbase2[self.battle_lineups[k][l],1] += 1
                        
                        
                if self.skill_green+self.random < self.skill_red:
                    for l in range(12,24,1):
                        self.playerbase2[self.battle_lineups[k][l],1] += 1
                        
                        
                if self.skill_green+self.random == self.skill_red:
                    for l in range(0,24,1):
                        self.playerbase2[self.battle_lineups[k][l],1] += 0.5
                        
        return self.playerbase2,self.WR_differences     
              
########################################################################    
    # SECOND TYPE OF MATCHMAKING
    # Employs non-uniform number of games per player
    #
    # This is somewhat closer to reality than the simple cases 1    
    
       
    # ####################################

    #
    # This is the prototypical NO RNG case meaning the outcome of a match
    # is entirely deterministic and based on the average skill of a team
    # THIS IS NOT HOW WOWS WORKS
    # JUST FOR COMPARISON 



    def gen_data_mod2_NO_RNG(self,n_iter):
        self.playerbase2 = np.copy(self.playerbase)
        self.WR_differences = np.zeros(n_iter)
        
        
        for m in range(0,n_iter,1):
            # shuffles the playerlist
            np.random.shuffle(self.players)        
           
            # Instead of shuffling the playerbase and then dividing them
            # into pieces of 24 we are now going to sample 24 players
            
            
            self.battle_team = np.random.choice(self.players,size=24,replace=False)
            
            ##########
            # The selected 24 player list is evenly split into parts
            # the first 12 are greens, the second 12 are reds
            
            # Determine the average skill of the players in the green portion    
            self.skill_green = 0
            for l in range(0,12,1):
                self.current_player_skill = self.playerbase2[self.battle_team[l],0]
                self.skill_green += self.current_player_skill/12
                
            # Determine the average skill of the players in the red portion    
            self.skill_red = 0
            for l in range(12,24,1):
                self.current_player_skill = self.playerbase2[self.battle_team[l],0]
                self.skill_red += self.current_player_skill/12
           
            #######
            # The outcome of the battle is determined here and the list is updated
            # In this version it is entirely up to the skill of the players
            # This is the "NO RNG" case
            self.WR_differences[m] = (self.skill_green-self.skill_red)
            
            self.playerbase2[self.battle_team[:],2] += 1
            
            if self.skill_green > self.skill_red:
                for l in range(0,12,1):
                    self.playerbase2[self.battle_team[l],1] += 1
                    
                    
            if self.skill_green < self.skill_red:
                for l in range(12,24,1):
                    self.playerbase2[self.battle_team[l],1] += 1
                    
                    
            if self.skill_green == self.skill_red:
                for l in range(0,24,1):
                    self.playerbase2[self.battle_team[l],1] += 0.5
        
        return self.playerbase2,self.WR_differences    
    
    
        # ####################################

    #
    # This is the prototypical NO RNG case meaning the outcome of a match
    # is entirely deterministic and based on the average skill of a team
    # THIS IS NOT HOW WOWS WORKS
    # JUST FOR COMPARISON 



    def gen_data_mod3_NO_RNG(self,n_iter,N_battles_per_iter=1):
        self.playerbase2 = np.copy(self.playerbase)
        self.WR_differences = np.zeros(n_iter)
        self.battle_density = np.zeros(len(self.playerbase))
        self.battle_density = np.exp(-0.002*np.arange(self.N_players,0,-1))
        self.prop_sum = np.sum(self.battle_density)
        self.battle_density = self.battle_density/self.prop_sum
        print(self.battle_density)        
        plt.plot(self.battle_density)
        
        for m in range(0,n_iter,1):
                       
            # Instead of shuffling the playerbase and then dividing them
            # into pieces of 24 we are now going to sample 24 players
                        
            self.battle_team = np.random.choice(self.players,size=24,replace=False,p=self.battle_density)
            
            ##########
            # The selected 24 player list is evenly split into parts
            # the first 12 are greens, the second 12 are reds
            
            # Determine the average skill of the players in the green portion    
            self.skill_green = 0
            for l in range(0,12,1):
                self.current_player_skill = self.playerbase2[self.battle_team[l],0]
                self.skill_green += self.current_player_skill/12
                
            # Determine the average skill of the players in the red portion    
            self.skill_red = 0
            for l in range(12,24,1):
                self.current_player_skill = self.playerbase2[self.battle_team[l],0]
                self.skill_red += self.current_player_skill/12
           
            #######
            # The outcome of the battle is determined here and the list is updated
            # In this version it is entirely up to the skill of the players
            # This is the "NO RNG" case
            self.WR_differences[m] = (self.skill_green-self.skill_red)
            
            self.playerbase2[self.battle_team[:],2] += 1
            
            if self.skill_green > self.skill_red:
                for l in range(0,12,1):
                    self.playerbase2[self.battle_team[l],1] += 1
                    
                    
            if self.skill_green < self.skill_red:
                for l in range(12,24,1):
                    self.playerbase2[self.battle_team[l],1] += 1
                    
                    
            if self.skill_green == self.skill_red:
                for l in range(0,24,1):
                    self.playerbase2[self.battle_team[l],1] += 0.5
        
        return self.playerbase2,self.WR_differences  
    
    
##############################################################################

# ############################################################################
# # VISUALIZATION SECTION
# # 
# #
font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 12,
        }

n_games = 500000
my_players = wows_model1(N_players=24*100)

my_players.define_skill(sigma=10)

playernumbers = my_players.playernumbers
skill = my_players.playerbase[:,0]

playerbase2, WR_diffs2 = my_players.gen_data_mod3_NO_RNG(n_games)

battles = np.arange(0,len(WR_diffs2),1)

# playerbase3 = my_players.gen_data_RNG_NORMAL(5000,RNG=100)
# playerbase4 = my_players.gen_data_RNG_NORMAL(5000,RNG=1000)


fig = plt.figure(figsize=(4,6),dpi=200)

ax = plt.subplot2grid((3,1),(1,0),colspan=1, rowspan=2)
plt.scatter(skill,playernumbers,color='navy',alpha=0.25,s=4)
ax.set_ylabel('player index', fontdict=font)
ax.set_xlabel('player skill level', fontdict=font)
plt.xlim(0, 100)
ax1 = plt.subplot2grid((3,1),(0,0),colspan=1, rowspan=1)
levels = np.arange(0,102.5,2.5)
n, bins, patches = plt.hist(skill,levels, density=False,facecolor='navy',alpha=0.75)
plt.xlim(0, 100)
plt.tick_params(axis='x', which='both', labelbottom=False, bottom=True, labelsize=8)
plt.title('Player skill distribution', fontdict=font)
ax1.set_ylabel('number of players', fontdict=font)
plt.tight_layout()
plt.show()
plt.close()


fig = plt.figure(figsize=(4,6),dpi=200)

ax = plt.subplot2grid((3,1),(1,0),colspan=1, rowspan=2)
plt.scatter(WR_diffs2,battles,color='black',alpha=0.1,s=5)
ax.set_ylabel('player index', fontdict=font)
ax.set_xlabel('player skill level', fontdict=font)
plt.xlim(-50, 50)
ax1 = plt.subplot2grid((3,1),(0,0),colspan=1, rowspan=1)
levels = np.arange(-100,100,1)
n, bins, patches = plt.hist(WR_diffs2,levels, density=False,facecolor='black',alpha=0.75)
plt.xlim(-50, 50)
plt.tick_params(axis='x', which='both', labelbottom=False, bottom=True, labelsize=8)
plt.title('Player skill distribution', fontdict=font)
ax1.set_ylabel('number of players', fontdict=font)
plt.tight_layout()
plt.show()
plt.close()


# # Skill level vs Winrate

fig = plt.figure(figsize=(6,4),dpi=200)
plt.scatter(playerbase2[:,0],100*playerbase2[:,1]/playerbase2[:,2],c=playerbase2[:,2],cmap='hsv',alpha=0.2,s=15)
# plt.scatter(playernumbers,skill2,color='tab:red')
ax = plt.gca()
ax.set_xlabel('player skill level / a.u.', fontdict=font)
ax.set_ylabel('Player winrate in percent', fontdict=font)
plt.tight_layout()
# plt.show()
plt.show()
plt.close()



fig = plt.figure(figsize=(4,6),dpi=200)

levels = np.arange(0,n_games/10,100)
n, bins, patches = plt.hist(playerbase2[:,2],levels, density=False,facecolor='black',alpha=0.75)
# plt.xlim(0, n_games)
# plt.tick_params(axis='x', which='both', labelbottom=False, bottom=True, labelsize=8)
plt.title('Player skill distribution', fontdict=font)
ax1.set_ylabel('number of players', fontdict=font)
# plt.tight_layout()
plt.show()
plt.close()




# fig = plt.figure(figsize=(6,4),dpi=200)
# plt.scatter(playerbase3[:,0],100*playerbase3[:,1]/500000,color='navy')
# # plt.scatter(playernumbers,skill2,color='tab:red')
# ax = plt.gca()
# ax.set_xlabel('player skill level / a.u.', fontdict=font)
# ax.set_ylabel('Player winrate in percent', fontdict=font)
# plt.tight_layout()
# # plt.show()
# plt.show()

# fig = plt.figure(figsize=(6,4),dpi=200)
# plt.scatter(playerbase4[:,0],100*playerbase4[:,1]/500000,color='navy')
# # plt.scatter(playernumbers,skill2,color='tab:red')
# ax = plt.gca()
# ax.set_xlabel('player skill level / a.u.', fontdict=font)
# ax.set_ylabel('Player winrate in percent', fontdict=font)
# plt.tight_layout()
# # plt.show()
# plt.show()

