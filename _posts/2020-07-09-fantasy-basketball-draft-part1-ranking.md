# Yahoo Fantasy Basketball Draft Part 1 - Snake draft value / best available player

Year after year, I have been part of a NBA fantasy league. Year after year, I have been losing. Not just losing, but not even making the playoffs. Imagine someone who's career is in data analytics, completely failing at something where evaluating data is key. Well, I have had enough and there is only so much pounding a person can take. ENOUGH!

It is time to put my superpowers to the test.

This will be a multiple part series of blog posts as I explore what is the best way to win at fantasy sports. The league that I am in is the most difficult because it is an auction draft keeper league. Where it is not just important to have the best players and evaluate based on rank, but how to properly price players will be very important.

Let us start with the basics. Assuming we have perfect data on a players performance for the current year, how do you build the most optimal team. There are many different strategies to doing this. For this example league, assume a 9 category, head-to-head matchup league. There are a couple of approaches people have to this problem, the most common would be to choose 5 or 6 out of the 9 categories and create a linear optimization problem where based on constraints, maximizing over those categories.

My gripe with this strategy is it is not flexible and is not able to adapt based on what other fantasy owners are doing during the draft. What if majority of other fantasy owners are going after the same categories and players are taken off the board?

### Fantasy Player Rankings
Below is a list of top players taken from https://www.fantraxhq.com/2019-fantasy-basketball-rankings/. The analyst here says that turnover category is not considered, but ranking is based on per-game production. I have always been curious if value of a player changes based on the size of the league. For eg, does having a player that over-indexes on shot blocking and provides nothing else more beneficial in a 4 person league than a 12 person league? 
My problem with ranking based on overall production metric is that if there are many players that over index in assists and steals and few that over index in rebounds and blocks, then you may miss the underlying value of players.

![](/images/logo.png "fast.ai's logo")
![](/images/nba_draft_order.PNG "Draft Rankings 2019-2020")

#### An Alternative

Run simulations of different random teams and see which player influences the most toward winning percentage. This would be similar to win-share. As the draft continues and owners draft players, you are able to run random simulations on the remainder of players to draft the next best available player.

### Prepping data

For the data mining, I have used [Yahoo_fantasy_basketball_analyzer](https://github.com/elwan9880/Yahoo_fantasy_basketball_analyzer) as a starting off point. It includes some other tools and has some ML to do some predictions, but I have just used it for data mining of basketballreference. The github instructions are great. The repo also adds z-scores for each players statistical categories so that you are able to see how much a player deviates from the league average. 

What the library does not do is add position of a player, so we will need to add that in.


```python
from IPython.display import clear_output
from yahoo_oauth import OAuth2
import yahoo_fantasy_api as yfa
import pandas as pd
import numpy as np
import timeit
import time
import sys

sc = OAuth2(None, None, from_file = 'oauth2.json')
game = yfa.Game(sc, 'nba')

lg = game.to_league(game.league_ids(year=2019)[0])

# csv file is outputted by Yahoo_fantasy_basketball_analyzer 
players = pd.read_csv('2019-2020_players.csv')
```

    [2020-07-07 16:16:56,900 DEBUG] [yahoo_oauth.oauth.__init__] Checking 
    [2020-07-07 16:16:56,901 DEBUG] [yahoo_oauth.oauth.token_is_valid] ELAPSED TIME : 22.219847440719604
    [2020-07-07 16:16:56,902 DEBUG] [yahoo_oauth.oauth.token_is_valid] TOKEN IS STILL VALID
    


```python
#hide
players = players.drop(columns=['Unnamed: 36'])
# Get names with single quotes
single_quote_df = players[players['Player'].str.contains("'")]
# Remove single quote player names from players dataframe
players = players[~players.Player.isin(single_quote_df.Player)]

players = players.replace({'DJ Augustin':'D.J. Augustin'
               , 'Juan Hernangomez':'Juancho Hernangomez'
               , 'PJ Tucker':'P.J. Tucker'
               , 'Sviatoslav Mykhailiuk':'Svi Mykhailiuk'
               , 'TJ McConnell':'T.J. McConnell'
               , 'TJ Warren':'T.J. Warren'
               , 'Wesley Iwundu':'Wes Iwundu'})

# Change Maurice Harkless to Moe Harkless
players = players.replace({'Maurice Harkless':'Moe Harkless'})
```

Add dummy variables for each player based on whether they are eligible for a position. 

*Things not shown are some of the data cleaning for some of the players names not matching between basketball reference and yahoo sports API*


```python
is_g = []
is_pg = []
is_sg = []
is_f = []
is_sf = []
is_pf = []
is_c = []
is_util = []

for row in players['Player']:
    positions = lg.player_details(row)[0].get('eligible_positions')
    if {'position': 'G'} in positions:
        is_g.append(1)
    else:
        is_g.append(0)
    if {'position': 'PG'} in positions:
        is_pg.append(1)
    else:
        is_pg.append(0)
    if {'position': 'SG'} in positions:
        is_sg.append(1)
    else:
        is_sg.append(0)
    if {'position': 'F'} in positions:
        is_f.append(1)
    else:
        is_f.append(0)
    if {'position': 'SF'} in positions:
        is_sf.append(1)
    else:
        is_sf.append(0)
    if {'position': 'PF'} in positions:
        is_pf.append(1)
    else:
        is_pf.append(0)
    if {'position': 'C'} in positions:
        is_c.append(1)
    else:
        is_c.append(0)
    if {'position': 'Util'} in positions:
        is_util.append(1)
    else:
        is_util.append(0)
        
players['is_g'] = is_g
players['is_pg'] = is_pg
players['is_sg'] = is_sg
players['is_f'] = is_f
players['is_sf'] = is_sf
players['is_pf'] = is_pf
players['is_c'] = is_c
players['is_util'] = is_util

```


```python
#hide
#d'agengelo: PG, SG
#de'aaron: PG
#de'andre: SF
#de'anthony: PG, SG
#deandre': SF
#devonte' PG, SG
#e'twaun: SG, SF
#royce: SF
is_g = [1,1,0,1,0,1,1,0]
is_pg = [1,1,0,1,0,1,0,0]
is_sg = [1,0,0,1,0,1,1,0]
is_f = [0,0,1,0,1,0,1,1]
is_sf = [0,0,1,0,1,0,1,1]
is_pf = [0,0,0,0,0,0,0,0]
is_c = [0,0,0,0,0,0,0,0]
is_util = [1,1,1,1,1,1,1,1]

single_quote_df['is_g'] = is_g
single_quote_df['is_pg'] = is_pg
single_quote_df['is_sg'] = is_sg
single_quote_df['is_f'] = is_f
single_quote_df['is_sf'] = is_sf
single_quote_df['is_pf'] = is_pf
single_quote_df['is_c'] = is_c
single_quote_df['is_util'] = is_util

players = pd.concat([players,single_quote_df])
players.to_csv('2019-2020_players_with_positions.csv')
```

Here we have our player 2019-2020 stat summary that we will be using. It has their 9 categories, in addition to FGM, FGA, FTM, FTA, 3PTM, 3PTA. The volume of shots in these categories influences the team performance for FG%, FT%, and 3PT%. So unless you can translate volume and percentage into a single variable, it would not be possible to do linear optimization. Another reason I found to run simulations.

It also has the Zscores that I will not be using, and dummy variables for eligible positions.


```python
players.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Player</th>
      <th>FGM</th>
      <th>FGA</th>
      <th>FG%</th>
      <th>FTM</th>
      <th>FTA</th>
      <th>FT%</th>
      <th>3PTM</th>
      <th>3PTA</th>
      <th>3PT%</th>
      <th>...</th>
      <th>zTO</th>
      <th>zTotal</th>
      <th>is_g</th>
      <th>is_pg</th>
      <th>is_sg</th>
      <th>is_f</th>
      <th>is_sf</th>
      <th>is_pf</th>
      <th>is_c</th>
      <th>is_util</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Aaron Gordon</td>
      <td>5.414</td>
      <td>12.517</td>
      <td>0.433</td>
      <td>2.362</td>
      <td>3.500</td>
      <td>0.675</td>
      <td>1.172</td>
      <td>3.897</td>
      <td>0.301</td>
      <td>...</td>
      <td>-0.076</td>
      <td>0.136</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Aaron Holiday</td>
      <td>3.483</td>
      <td>8.552</td>
      <td>0.407</td>
      <td>1.069</td>
      <td>1.241</td>
      <td>0.861</td>
      <td>1.379</td>
      <td>3.500</td>
      <td>0.394</td>
      <td>...</td>
      <td>0.319</td>
      <td>-0.348</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Al Horford</td>
      <td>4.817</td>
      <td>10.900</td>
      <td>0.442</td>
      <td>0.917</td>
      <td>1.217</td>
      <td>0.753</td>
      <td>1.483</td>
      <td>4.400</td>
      <td>0.337</td>
      <td>...</td>
      <td>0.572</td>
      <td>0.141</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Al-Farouq Aminu</td>
      <td>1.389</td>
      <td>4.778</td>
      <td>0.291</td>
      <td>1.056</td>
      <td>1.611</td>
      <td>0.655</td>
      <td>0.500</td>
      <td>2.000</td>
      <td>0.250</td>
      <td>...</td>
      <td>0.740</td>
      <td>-0.590</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Alec Burks</td>
      <td>4.797</td>
      <td>11.881</td>
      <td>0.404</td>
      <td>3.814</td>
      <td>4.254</td>
      <td>0.896</td>
      <td>1.695</td>
      <td>4.627</td>
      <td>0.366</td>
      <td>...</td>
      <td>0.141</td>
      <td>0.219</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 44 columns</p>
</div>



### Draft Random Teams
The plan is to
1. Run a draft where a player record has a random team
2. Get the league results based on the draft, each player will be given the result based on random team they were put on 
3. Get a summary of all the drafts and player performance over multiple random teams (get_draft_summary)
4. Run multiple simulations and shrink the player pool to top performers

Let's explain number 4. I have run the simulation with a 4 team league. With 300 players in our dataframe and 4 teams only requiring 4 * 14 = 56 players, you are getting a lot of simulations of teams that players never to be drafted in this system. This could create a bias in the result. So there needs to be a methodology that over time, players are removed from the draft pool.


```python
def create_teams(num_teams, eligible_players):
    # Uses global players dataframe
    # Adds rand_team assignment to dataframe
    # For each team and position, find a random player and assign to team
    # End of player selection process, only keep drafted players
    # ----------------------
    # Get team performance compared to other teams
    # Number of categories won compared to other teams
    # Number of head to head matchups won compared to other teams
    # return summary dataframe (player, rand_team, performance metrics)

    draft_players = players
    draft_players = draft_players[draft_players['Player'].isin(eligible_players)]
    draft_players['rand_team'] = 0

    for i in range(1,num_teams + 1):
        # find a PG
        rand_player = draft_players[(draft_players['is_pg']==1) & (draft_players['rand_team']==0)].sample()
        rand_player = rand_player.Player.to_string(index=False).strip()
        draft_players.loc[draft_players['Player'] == rand_player, ['rand_team']] = i

        # find a SG
        rand_player = draft_players[(draft_players['is_sg']==1) & (draft_players['rand_team']==0)].sample()
        rand_player = rand_player.Player.to_string(index=False).strip()
        draft_players.loc[draft_players['Player'] == rand_player, ['rand_team']] = i

        # find a G
        rand_player = draft_players[(draft_players['is_g']==1) & (draft_players['rand_team']==0)].sample()
        rand_player = rand_player.Player.to_string(index=False).strip()
        draft_players.loc[draft_players['Player'] == rand_player, ['rand_team']] = i

        # find a SF
        rand_player = draft_players[(draft_players['is_sf']==1) & (draft_players['rand_team']==0)].sample()
        rand_player = rand_player.Player.to_string(index=False).strip()
        draft_players.loc[draft_players['Player'] == rand_player, ['rand_team']] = i

        # find a PF
        rand_player = draft_players[(draft_players['is_pf']==1) & (draft_players['rand_team']==0)].sample()
        rand_player = rand_player.Player.to_string(index=False).strip()
        draft_players.loc[draft_players['Player'] == rand_player, ['rand_team']] = i

        # find a F
        rand_player = draft_players[(draft_players['is_f']==1) & (draft_players['rand_team']==0)].sample()
        rand_player = rand_player.Player.to_string(index=False).strip()
        draft_players.loc[draft_players['Player'] == rand_player, ['rand_team']] = i

        # find a C
        rand_player = draft_players[(draft_players['is_c']==1) & (draft_players['rand_team']==0)].sample()
        rand_player = rand_player.Player.to_string(index=False).strip()
        draft_players.loc[draft_players['Player'] == rand_player, ['rand_team']] = i

        # find a C
        rand_player = draft_players[(draft_players['is_c']==1) & (draft_players['rand_team']==0)].sample()
        rand_player = rand_player.Player.to_string(index=False).strip()
        draft_players.loc[draft_players['Player'] == rand_player, ['rand_team']] = i

        # find a Util
        rand_player = draft_players[(draft_players['is_util']==1) & (draft_players['rand_team']==0)].sample()
        rand_player = rand_player.Player.to_string(index=False).strip()
        draft_players.loc[draft_players['Player'] == rand_player, ['rand_team']] = i

        # find a Util
        rand_player = draft_players[(draft_players['is_util']==1) & (draft_players['rand_team']==0)].sample()
        rand_player = rand_player.Player.to_string(index=False).strip()
        draft_players.loc[draft_players['Player'] == rand_player, ['rand_team']] = i

        # find a Util
        rand_player = draft_players[(draft_players['is_util']==1) & (draft_players['rand_team']==0)].sample()
        rand_player = rand_player.Player.to_string(index=False).strip()
        draft_players.loc[draft_players['Player'] == rand_player, ['rand_team']] = i

        # find a Util
        rand_player = draft_players[(draft_players['is_util']==1) & (draft_players['rand_team']==0)].sample()
        rand_player = rand_player.Player.to_string(index=False).strip()
        draft_players.loc[draft_players['Player'] == rand_player, ['rand_team']] = i

        # find a Util
        rand_player = draft_players[(draft_players['is_util']==1) & (draft_players['rand_team']==0)].sample()
        rand_player = rand_player.Player.to_string(index=False).strip()
        draft_players.loc[draft_players['Player'] == rand_player, ['rand_team']] = i

        # find a Util
        rand_player = draft_players[(draft_players['is_util']==1) & (draft_players['rand_team']==0)].sample()
        rand_player = rand_player.Player.to_string(index=False).strip()
        draft_players.loc[draft_players['Player'] == rand_player, ['rand_team']] = i

    # select only players drafted
    draft_players = draft_players[draft_players['rand_team']>0]
    
    # get draft team results
    draft_teams = draft_players.groupby(['rand_team'])[['FGM','FGA','FTM','FTA','3PTM','PTS','REB','AST','ST','BLK','TO']].apply(sum).reset_index()

    draft_teams['FGP'] = draft_teams['FGM'] / draft_teams['FGA']
    draft_teams['FTP'] = draft_teams['FTM'] / draft_teams['FTA']
    
    # get scores
    cat_win = []
    cat_loss = []
    matchup_win = []
    matchup_loss = []
    for curr_team in draft_teams['rand_team']:
        hh_cat_win = 0
        hh_cat_loss = 0
        hh_match_win = 0
        hh_match_loss = 0

        for matchup_team in draft_teams['rand_team']:
            hhwins = 0
            hhlosses = 0
            # loop only for other teams
            if curr_team != matchup_team:
                # FGP
                if (float(draft_teams[draft_teams['rand_team']==curr_team]['FGP']) > float(draft_teams[draft_teams['rand_team']==matchup_team]['FGP'])):
                    hhwins += 1
                else:
                    hhlosses += 1
                # FTP
                if (float(draft_teams[draft_teams['rand_team']==curr_team]['FTP']) > float(draft_teams[draft_teams['rand_team']==matchup_team]['FTP'])):
                    hhwins += 1
                else:
                    hhlosses += 1
                # PTS
                if (float(draft_teams[draft_teams['rand_team']==curr_team]['PTS']) > float(draft_teams[draft_teams['rand_team']==matchup_team]['PTS'])):
                    hhwins += 1
                else:
                    hhlosses += 1
                # REBS
                if (float(draft_teams[draft_teams['rand_team']==curr_team]['REB']) > float(draft_teams[draft_teams['rand_team']==matchup_team]['REB'])):
                    hhwins += 1
                else:
                    hhlosses += 1
                # AST
                if (float(draft_teams[draft_teams['rand_team']==curr_team]['AST']) > float(draft_teams[draft_teams['rand_team']==matchup_team]['AST'])):
                    hhwins += 1
                else:
                    hhlosses += 1
                # ST
                if (float(draft_teams[draft_teams['rand_team']==curr_team]['ST']) > float(draft_teams[draft_teams['rand_team']==matchup_team]['ST'])):
                    hhwins += 1
                else:
                    hhlosses += 1
                # BLK
                if (float(draft_teams[draft_teams['rand_team']==curr_team]['BLK']) > float(draft_teams[draft_teams['rand_team']==matchup_team]['BLK'])):
                    hhwins += 1
                else:
                    hhlosses += 1
                # BLK
                if (float(draft_teams[draft_teams['rand_team']==curr_team]['TO']) < float(draft_teams[draft_teams['rand_team']==matchup_team]['TO'])):
                    hhwins += 1
                else:
                    hhlosses += 1

                # Add heads up result
                if hhwins > hhlosses:
                    hh_match_win += 1
                else:
                    hh_match_loss += 1

            # Add heads up category totals
            hh_cat_win = hh_cat_win + hh_match_win
            hh_cat_loss = hh_cat_loss + hh_match_loss  

        # when current team match over append scores
        cat_win.append(hh_cat_win)
        cat_loss.append(hh_cat_loss)
        matchup_win.append(hh_match_win)
        matchup_loss.append(hh_match_loss)

    draft_teams['cat_win'] = cat_win
    draft_teams['cat_loss'] = cat_loss
    draft_teams['matchup_win'] = matchup_win
    draft_teams['matchup_loss'] = matchup_loss
    
    rand_summary = draft_players[['Player','rand_team']].merge(draft_teams[['rand_team','cat_win','cat_loss','matchup_win','matchup_loss']], on='rand_team')

    return rand_summary
```


```python
def get_draft_summary(draft_players):
    sim_summary = draft_players.groupby(['Player'])[['cat_win','cat_loss','matchup_win','matchup_loss']].apply(sum).reset_index()
    sim_summary['cat_perc'] = sim_summary['cat_win'] / (sim_summary['cat_win'] + sim_summary['cat_loss'])
    sim_summary['matchup_perc'] = sim_summary['matchup_win'] / (sim_summary['matchup_win'] + sim_summary['matchup_loss'])
    return sim_summary
```


```python
def fantasy_player_rank(num_teams, simulations):

    draft_players = players
    eligible_players = players['Player'].tolist()
    start = timeit.default_timer()
    
    staticTrimSize = 10
    staticTrimInterval = 20
    
    num_trim_players = 0
    
    for r in range(1,simulations + 1):
        clear_output(wait=True)
        draft_players = draft_players.append(create_teams(num_teams, eligible_players), ignore_index=True)

        stop = timeit.default_timer()

        if( r/simulations*100) < 5:
            expected_time = "Calculating..."
        else:
            time_perc = timeit.default_timer()
            expected_time = np.round( (time_perc-start)/(r/simulations) / 60,2)

        print("Current progress:", np.round(r/simulations*100,2), "%")
        print("Current run time:", np.round((stop - start)/60,2), "minutes")
        print("Exptected run time:", expected_time, "minutes")
        
        # get the number of times player has been drafted from still eligible Player dataframe
        # remove players from simulation that will not be drafted by teams, and bias simulation
        
        num_player_drafted_df = draft_players[['Player','rand_team']].groupby(['Player']).agg('count').reset_index()
        num_player_drafted_df.columns = ['Player','occurrences']
        num_player_drafted_df = num_player_drafted_df.merge(players['Player'], how='inner', on='Player')
        
        # min_occurrences from eligible players 300 - (num_trim_players * num_trim_per_round)
        num_player_drafted_df = num_player_drafted_df.sort_values(by=['occurrences'], ascending=False)
        num_player_drafted_df = num_player_drafted_df[:(300 - num_trim_players * staticTrimSize)]
        min_occurrences_of_player = num_player_drafted_df['occurrences'].min()
        
        print('Number of teams:', num_teams)
        print('Num Trim Players:', num_trim_players)
        print('Number of rows in eligible players dataframe:', len(eligible_players))
        print('Minimum times player has been drafted in eligible num_player_drafted dataframe:', min_occurrences_of_player)
        print('Number of rows in num_player_drafted:', len(num_player_drafted_df))
        
        # for first trimming of players, must have 300 players drafted and each player at least 5 occurrences
        # subsequent times, must have at least 5 more occurences than the last
        # also need to make sure there are enough players left to draft
        if ((min_occurrences_of_player == staticTrimInterval) and len(num_player_drafted_df) == 300 and (len(players) > num_teams * 16 )) or \
        ((min_occurrences_of_player == staticTrimInterval * (num_trim_players + 1)) and (len(eligible_players) > num_teams * 16 + staticTrimSize)) :
            
            # evaluate players and number of eligible players = 300 - (num_trim_players * num_trim_per_round)
            num_trim_players += 1
            sim_summary = get_draft_summary(draft_players).sort_values(by=['cat_perc'], ascending=False)
            eligible_players = sim_summary['Player'][:(300 - num_trim_players * staticTrimSize)].tolist()

    sim_summary = draft_players.groupby(['Player'])[['cat_win','cat_loss','matchup_win','matchup_loss']].apply(sum).reset_index()
    sim_summary['cat_perc'] = sim_summary['cat_win'] / (sim_summary['cat_win'] + sim_summary['cat_loss'])
    sim_summary['matchup_perc'] = sim_summary['matchup_win'] / (sim_summary['matchup_win'] + sim_summary['matchup_loss'])
    return sim_summary
```


```python
#hide
players = pd.read_csv('2019-2020_players_with_positions.csv')
players = players.drop(columns=['Unnamed: 0'])
```


```python
fantasy_6teams = fantasy_player_rank(6,1000)
```

    Current progress: 100.0 %
    Current run time: 7.76 minutes
    Exptected run time: 7.76 minutes
    Number of teams: 6
    Num Trim Players: 14
    Number of rows in eligible players dataframe: 160
    Minimum times player has been drafted in eligible num_player_drafted dataframe: 297
    Number of rows in num_player_drafted: 160
    


```python
fantasy_6teams.sort_values(by=['cat_perc'], ascending=False).head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Player</th>
      <th>cat_win</th>
      <th>cat_loss</th>
      <th>matchup_win</th>
      <th>matchup_loss</th>
      <th>cat_perc</th>
      <th>matchup_perc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12</th>
      <td>Anthony Davis</td>
      <td>4513.0</td>
      <td>3134.0</td>
      <td>1302.0</td>
      <td>898.0</td>
      <td>0.590166</td>
      <td>0.591818</td>
    </tr>
    <tr>
      <th>132</th>
      <td>James Harden</td>
      <td>3339.0</td>
      <td>2968.0</td>
      <td>949.0</td>
      <td>851.0</td>
      <td>0.529412</td>
      <td>0.527222</td>
    </tr>
    <tr>
      <th>115</th>
      <td>Hassan Whiteside</td>
      <td>3362.0</td>
      <td>3039.0</td>
      <td>953.0</td>
      <td>872.0</td>
      <td>0.525230</td>
      <td>0.522192</td>
    </tr>
    <tr>
      <th>165</th>
      <td>Kawhi Leonard</td>
      <td>3127.0</td>
      <td>2843.0</td>
      <td>876.0</td>
      <td>814.0</td>
      <td>0.523786</td>
      <td>0.518343</td>
    </tr>
    <tr>
      <th>143</th>
      <td>Jimmy Butler</td>
      <td>2990.0</td>
      <td>2799.0</td>
      <td>850.0</td>
      <td>810.0</td>
      <td>0.516497</td>
      <td>0.512048</td>
    </tr>
    <tr>
      <th>164</th>
      <td>Karl-Anthony Towns</td>
      <td>3254.0</td>
      <td>3122.0</td>
      <td>928.0</td>
      <td>887.0</td>
      <td>0.510351</td>
      <td>0.511295</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Andre Drummond</td>
      <td>3679.0</td>
      <td>3581.0</td>
      <td>1053.0</td>
      <td>1032.0</td>
      <td>0.506749</td>
      <td>0.505036</td>
    </tr>
    <tr>
      <th>69</th>
      <td>Deandre Ayton</td>
      <td>3129.0</td>
      <td>3086.0</td>
      <td>887.0</td>
      <td>898.0</td>
      <td>0.503459</td>
      <td>0.496919</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Ben Simmons</td>
      <td>2570.0</td>
      <td>2555.0</td>
      <td>728.0</td>
      <td>717.0</td>
      <td>0.501463</td>
      <td>0.503806</td>
    </tr>
    <tr>
      <th>108</th>
      <td>Giannis Antetokounmpo</td>
      <td>2847.0</td>
      <td>2855.0</td>
      <td>822.0</td>
      <td>803.0</td>
      <td>0.499298</td>
      <td>0.505846</td>
    </tr>
  </tbody>
</table>
</div>




```python
fantasy_6teams_2ndrun = fantasy_player_rank(6,2000)
```

    Current progress: 100.0 %
    Current run time: 16.64 minutes
    Exptected run time: 16.64 minutes
    Number of teams: 6
    Num Trim Players: 18
    Number of rows in eligible players dataframe: 120
    Minimum times player has been drafted in eligible num_player_drafted dataframe: 845
    Number of rows in num_player_drafted: 120
    


```python
fantasy_6teams_2ndrun.sort_values(by=['cat_perc'], ascending=False).head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Player</th>
      <th>cat_win</th>
      <th>cat_loss</th>
      <th>matchup_win</th>
      <th>matchup_loss</th>
      <th>cat_perc</th>
      <th>matchup_perc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12</th>
      <td>Anthony Davis</td>
      <td>11283.0</td>
      <td>9251.0</td>
      <td>3213.0</td>
      <td>2657.0</td>
      <td>0.549479</td>
      <td>0.547359</td>
    </tr>
    <tr>
      <th>132</th>
      <td>James Harden</td>
      <td>8985.0</td>
      <td>9125.0</td>
      <td>2572.0</td>
      <td>2598.0</td>
      <td>0.496135</td>
      <td>0.497485</td>
    </tr>
    <tr>
      <th>165</th>
      <td>Kawhi Leonard</td>
      <td>9147.0</td>
      <td>9536.0</td>
      <td>2628.0</td>
      <td>2712.0</td>
      <td>0.489589</td>
      <td>0.492135</td>
    </tr>
    <tr>
      <th>115</th>
      <td>Hassan Whiteside</td>
      <td>8261.0</td>
      <td>8997.0</td>
      <td>2357.0</td>
      <td>2543.0</td>
      <td>0.478677</td>
      <td>0.481020</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Ben Simmons</td>
      <td>7473.0</td>
      <td>8491.0</td>
      <td>2149.0</td>
      <td>2436.0</td>
      <td>0.468116</td>
      <td>0.468702</td>
    </tr>
    <tr>
      <th>186</th>
      <td>Kyrie Irving</td>
      <td>8454.0</td>
      <td>9699.0</td>
      <td>2408.0</td>
      <td>2752.0</td>
      <td>0.465708</td>
      <td>0.466667</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Andre Drummond</td>
      <td>9026.0</td>
      <td>10748.0</td>
      <td>2589.0</td>
      <td>3076.0</td>
      <td>0.456458</td>
      <td>0.457017</td>
    </tr>
    <tr>
      <th>147</th>
      <td>John Collins</td>
      <td>9572.0</td>
      <td>11463.0</td>
      <td>2757.0</td>
      <td>3258.0</td>
      <td>0.455051</td>
      <td>0.458354</td>
    </tr>
    <tr>
      <th>143</th>
      <td>Jimmy Butler</td>
      <td>8781.0</td>
      <td>10519.0</td>
      <td>2531.0</td>
      <td>2989.0</td>
      <td>0.454974</td>
      <td>0.458514</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Clint Capela</td>
      <td>9491.0</td>
      <td>11413.0</td>
      <td>2714.0</td>
      <td>3266.0</td>
      <td>0.454028</td>
      <td>0.453846</td>
    </tr>
    <tr>
      <th>228</th>
      <td>Nikola Jokic</td>
      <td>9586.0</td>
      <td>11699.0</td>
      <td>2769.0</td>
      <td>3341.0</td>
      <td>0.450364</td>
      <td>0.453191</td>
    </tr>
    <tr>
      <th>108</th>
      <td>Giannis Antetokounmpo</td>
      <td>7949.0</td>
      <td>9991.0</td>
      <td>2295.0</td>
      <td>2845.0</td>
      <td>0.443088</td>
      <td>0.446498</td>
    </tr>
    <tr>
      <th>69</th>
      <td>Deandre Ayton</td>
      <td>7795.0</td>
      <td>9929.0</td>
      <td>2224.0</td>
      <td>2826.0</td>
      <td>0.439799</td>
      <td>0.440396</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Bam Adebayo</td>
      <td>9036.0</td>
      <td>11527.0</td>
      <td>2609.0</td>
      <td>3286.0</td>
      <td>0.439430</td>
      <td>0.442578</td>
    </tr>
    <tr>
      <th>164</th>
      <td>Karl-Anthony Towns</td>
      <td>7776.0</td>
      <td>9930.0</td>
      <td>2231.0</td>
      <td>2809.0</td>
      <td>0.439173</td>
      <td>0.442659</td>
    </tr>
    <tr>
      <th>149</th>
      <td>Jonathan Isaac</td>
      <td>7684.0</td>
      <td>9837.0</td>
      <td>2209.0</td>
      <td>2811.0</td>
      <td>0.438559</td>
      <td>0.440040</td>
    </tr>
    <tr>
      <th>52</th>
      <td>Damian Lillard</td>
      <td>7053.0</td>
      <td>9051.0</td>
      <td>2014.0</td>
      <td>2541.0</td>
      <td>0.437966</td>
      <td>0.442151</td>
    </tr>
    <tr>
      <th>254</th>
      <td>Rudy Gobert</td>
      <td>7480.0</td>
      <td>9990.0</td>
      <td>2107.0</td>
      <td>2868.0</td>
      <td>0.428163</td>
      <td>0.423518</td>
    </tr>
    <tr>
      <th>247</th>
      <td>Richaun Holmes</td>
      <td>8621.0</td>
      <td>11591.0</td>
      <td>2496.0</td>
      <td>3309.0</td>
      <td>0.426529</td>
      <td>0.429974</td>
    </tr>
    <tr>
      <th>146</th>
      <td>Joel Embiid</td>
      <td>8585.0</td>
      <td>11608.0</td>
      <td>2446.0</td>
      <td>3364.0</td>
      <td>0.425147</td>
      <td>0.420998</td>
    </tr>
  </tbody>
</table>
</div>



### Runtime Problem
The results have not settled with even 5000 simulations, and the runtime is ballooning. If we wanted to run simulations during a live draft, we only have so much time to make a decision, so we are going to have to make this go faster.

*Multiprocessing, Windows, Jupyter Python Interactive Shell*
So many problems and so many attempts to make this work. Long story short, in order for parallel processing to work, it needs to exist in its own script and __main__.


```python
## main in mock_draft_parallel.py
if __name__ == '__main__':

  # regular run
  # starttime = time.perf_counter()
  # results = create_teams(6)
  # for i in range(5):
  #   results = results.append(create_teams(6))

  # endtime = time.perf_counter()
  # print(f'Finished in {round(endtime-starttime,2)} seconds(s)')

  # parallel run
  starttime = time.perf_counter()

  draft_players = players
  eligible_players = players['Player'].tolist()
  num_trim_players = 0
  num_teams = int(sys.argv[1])
  num_simulations = int(sys.argv[2])
  filename = sys.argv[3]

  staticTrimSize = 10
  staticTrimInterval = 20

  with concurrent.futures.ProcessPoolExecutor() as executor:

    # list comprehension
    results = [executor.submit(create_teams, num_teams, eligible_players) for _ in range(num_simulations)]

    for f in concurrent.futures.as_completed(results):
      draft_players = draft_players.append(f.result(), ignore_index=True)
      
      # get the number of times player has been drafted from still eligible Player dataframe
      # remove players from simulation that will not be drafted by teams, and bias simulation
      
      if(num_teams * 16 < len(eligible_players)):
        num_player_drafted_df = draft_players[['Player','rand_team']].groupby(['Player']).agg('count').reset_index()
        num_player_drafted_df.columns = ['Player','occurrences']
        
        # min_occurrences from eligible players 300 - (num_trim_players * num_trim_per_round)
        num_player_drafted_df = num_player_drafted_df.sort_values(by=['occurrences'], ascending=False)
        num_player_drafted_df = num_player_drafted_df[:(300 - num_trim_players * staticTrimSize)]
        min_occurrences_of_player = num_player_drafted_df['occurrences'].min()
      
      # for first trimming of players, must have 300 players drafted and each player at least 5 occurrences
      # subsequent times, must have at least 5 more occurences than the last
      # also need to make sure there are enough players left to draft
      if ((min_occurrences_of_player == staticTrimInterval) and len(num_player_drafted_df) == 300 and (len(players) > num_teams * 16 )) or \
      ((min_occurrences_of_player == staticTrimInterval * (num_trim_players + 1)) and (len(eligible_players) > num_teams * 16)) :
          
        # evaluate players and number of eligible players = 300 - (num_trim_players * num_trim_per_round)
        num_trim_players += 1
        sim_summary = get_draft_summary(draft_players).sort_values(by=['cat_perc'], ascending=False)
        eligible_players = sim_summary['Player'][:(300 - num_trim_players * staticTrimSize)].tolist()
          

  endtime = time.perf_counter()
  print(f'Finished in {round(endtime-starttime,2)} seconds(s)')

  sim_summary = draft_players.groupby(['Player'])[['cat_win','cat_loss','matchup_win','matchup_loss']].apply(sum).reset_index()
  sim_summary['cat_perc'] = sim_summary['cat_win'] / (sim_summary['cat_win'] + sim_summary['cat_loss'])
  sim_summary['matchup_perc'] = sim_summary['matchup_win'] / (sim_summary['matchup_win'] + sim_summary['matchup_loss'])
  sim_summary = sim_summary.sort_values(by=['cat_perc'], ascending=False)
  sim_summary.to_csv(filename)
```


```python
import mock_draft_parallel
%run mock_draft_parallel 6 2000 parallel_result.csv
```

    Finished in 253.84 seconds(s)
    


```python
pd.read_csv('parallel_result.csv').drop(columns=['Unnamed: 0']).head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Player</th>
      <th>cat_win</th>
      <th>cat_loss</th>
      <th>matchup_win</th>
      <th>matchup_loss</th>
      <th>cat_perc</th>
      <th>matchup_perc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Anthony Davis</td>
      <td>7733.0</td>
      <td>4688.0</td>
      <td>2196.0</td>
      <td>1359.0</td>
      <td>0.622575</td>
      <td>0.617722</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Kawhi Leonard</td>
      <td>5543.0</td>
      <td>4315.0</td>
      <td>1581.0</td>
      <td>1254.0</td>
      <td>0.562284</td>
      <td>0.557672</td>
    </tr>
    <tr>
      <th>2</th>
      <td>James Harden</td>
      <td>5366.0</td>
      <td>4192.0</td>
      <td>1534.0</td>
      <td>1196.0</td>
      <td>0.561415</td>
      <td>0.561905</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Hassan Whiteside</td>
      <td>5564.0</td>
      <td>4428.0</td>
      <td>1569.0</td>
      <td>1276.0</td>
      <td>0.556845</td>
      <td>0.551494</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Jimmy Butler</td>
      <td>5147.0</td>
      <td>4140.0</td>
      <td>1457.0</td>
      <td>1178.0</td>
      <td>0.554216</td>
      <td>0.552941</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Kyrie Irving</td>
      <td>5155.0</td>
      <td>4201.0</td>
      <td>1475.0</td>
      <td>1190.0</td>
      <td>0.550983</td>
      <td>0.553471</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Ben Simmons</td>
      <td>4312.0</td>
      <td>3613.0</td>
      <td>1232.0</td>
      <td>1033.0</td>
      <td>0.544101</td>
      <td>0.543929</td>
    </tr>
    <tr>
      <th>7</th>
      <td>John Collins</td>
      <td>6924.0</td>
      <td>5824.0</td>
      <td>1968.0</td>
      <td>1667.0</td>
      <td>0.543144</td>
      <td>0.541403</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Nikola Jokic</td>
      <td>6680.0</td>
      <td>5669.0</td>
      <td>1912.0</td>
      <td>1608.0</td>
      <td>0.540934</td>
      <td>0.543182</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Deandre Ayton</td>
      <td>5479.0</td>
      <td>4887.0</td>
      <td>1545.0</td>
      <td>1405.0</td>
      <td>0.528555</td>
      <td>0.523729</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Giannis Antetokounmpo</td>
      <td>4672.0</td>
      <td>4192.0</td>
      <td>1349.0</td>
      <td>1206.0</td>
      <td>0.527076</td>
      <td>0.527984</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Clint Capela</td>
      <td>6564.0</td>
      <td>6008.0</td>
      <td>1867.0</td>
      <td>1728.0</td>
      <td>0.522113</td>
      <td>0.519332</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Karl-Anthony Towns</td>
      <td>5331.0</td>
      <td>4980.0</td>
      <td>1526.0</td>
      <td>1394.0</td>
      <td>0.517021</td>
      <td>0.522603</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Damian Lillard</td>
      <td>4020.0</td>
      <td>3860.0</td>
      <td>1148.0</td>
      <td>1102.0</td>
      <td>0.510152</td>
      <td>0.510222</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Andre Drummond</td>
      <td>6508.0</td>
      <td>6261.0</td>
      <td>1851.0</td>
      <td>1769.0</td>
      <td>0.509672</td>
      <td>0.511326</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Rudy Gobert</td>
      <td>5073.0</td>
      <td>4896.0</td>
      <td>1439.0</td>
      <td>1381.0</td>
      <td>0.508878</td>
      <td>0.510284</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Jonathan Isaac</td>
      <td>4628.0</td>
      <td>4500.0</td>
      <td>1308.0</td>
      <td>1287.0</td>
      <td>0.507011</td>
      <td>0.504046</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Joel Embiid</td>
      <td>6155.0</td>
      <td>5996.0</td>
      <td>1780.0</td>
      <td>1710.0</td>
      <td>0.506543</td>
      <td>0.510029</td>
    </tr>
    <tr>
      <th>18</th>
      <td>DeMar DeRozan</td>
      <td>5191.0</td>
      <td>5124.0</td>
      <td>1476.0</td>
      <td>1464.0</td>
      <td>0.503248</td>
      <td>0.502041</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Chris Paul</td>
      <td>3859.0</td>
      <td>3863.0</td>
      <td>1085.0</td>
      <td>1130.0</td>
      <td>0.499741</td>
      <td>0.489842</td>
    </tr>
  </tbody>
</table>
</div>



# Parallel Time Improvement
The parallel run improved the run time for the 2000 simulations from 884 seconds to 217 seconds. That is a more than 4 times improvement. Draft time usually allows for 2 minutes per round, so will need to run slightly less simulations, but as teams become filled and number of permutations decreases, it will need less simulations to become accurate. 

### Top Fantasy Players 2019-2020 Season
I have ran simulations for all players and here are the rankings for 6-team, 8-team, 10-team, 12-team, 14-team leagues.


```python
team6_rank_df = pd.read_csv('teams_6_simulations.csv').drop(columns=['Unnamed: 0'])
team8_rank_df = pd.read_csv('teams_8_simulations.csv').drop(columns=['Unnamed: 0'])
team10_rank_df = pd.read_csv('teams_10_simulations.csv').drop(columns=['Unnamed: 0'])
team12_rank_df = pd.read_csv('teams_12_simulations.csv').drop(columns=['Unnamed: 0'])
team14_rank_df = pd.read_csv('teams_14_simulations.csv').drop(columns=['Unnamed: 0'])
```


```python
team6_rank_df = team6_rank_df.rename(columns={"Player": "6-team"})
team8_rank_df = team8_rank_df.rename(columns={"Player": "8-team"})
team10_rank_df = team10_rank_df.rename(columns={"Player": "10-team"})
team12_rank_df = team12_rank_df.rename(columns={"Player": "12-team"})
team14_rank_df = team14_rank_df.rename(columns={"Player": "14-team"})
```


```python
pd.set_option('display.max_rows', 100)
team6_rank_df[['6-team']].merge(team8_rank_df[['8-team']], left_index=True, right_index=True) \
    .merge(team10_rank_df[['10-team']], left_index=True, right_index=True) \
    .merge(team12_rank_df[['12-team']], left_index=True, right_index=True) \
    .merge(team14_rank_df[['14-team']], left_index=True, right_index=True) \
    .head(100)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>6-team</th>
      <th>8-team</th>
      <th>10-team</th>
      <th>12-team</th>
      <th>14-team</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Anthony Davis</td>
      <td>Anthony Davis</td>
      <td>Anthony Davis</td>
      <td>Anthony Davis</td>
      <td>Anthony Davis</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Kawhi Leonard</td>
      <td>Kawhi Leonard</td>
      <td>Kawhi Leonard</td>
      <td>Kawhi Leonard</td>
      <td>Kawhi Leonard</td>
    </tr>
    <tr>
      <th>2</th>
      <td>James Harden</td>
      <td>James Harden</td>
      <td>James Harden</td>
      <td>James Harden</td>
      <td>James Harden</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Hassan Whiteside</td>
      <td>Hassan Whiteside</td>
      <td>Hassan Whiteside</td>
      <td>Hassan Whiteside</td>
      <td>Kyrie Irving</td>
    </tr>
    <tr>
      <th>4</th>
      <td>John Collins</td>
      <td>Kyrie Irving</td>
      <td>Kyrie Irving</td>
      <td>Kyrie Irving</td>
      <td>Hassan Whiteside</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Kyrie Irving</td>
      <td>Jimmy Butler</td>
      <td>John Collins</td>
      <td>Jimmy Butler</td>
      <td>John Collins</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Giannis Antetokounmpo</td>
      <td>John Collins</td>
      <td>Jimmy Butler</td>
      <td>John Collins</td>
      <td>Giannis Antetokounmpo</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Jimmy Butler</td>
      <td>Giannis Antetokounmpo</td>
      <td>Ben Simmons</td>
      <td>Giannis Antetokounmpo</td>
      <td>Jimmy Butler</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Karl-Anthony Towns</td>
      <td>Karl-Anthony Towns</td>
      <td>Giannis Antetokounmpo</td>
      <td>Andre Drummond</td>
      <td>Ben Simmons</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Nikola Jokic</td>
      <td>Andre Drummond</td>
      <td>Karl-Anthony Towns</td>
      <td>Karl-Anthony Towns</td>
      <td>Andre Drummond</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Ben Simmons</td>
      <td>Nikola Jokic</td>
      <td>Nikola Jokic</td>
      <td>Nikola Jokic</td>
      <td>Nikola Jokic</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Andre Drummond</td>
      <td>Deandre Ayton</td>
      <td>Deandre Ayton</td>
      <td>Ben Simmons</td>
      <td>Deandre Ayton</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Deandre Ayton</td>
      <td>Ben Simmons</td>
      <td>Andre Drummond</td>
      <td>Deandre Ayton</td>
      <td>Clint Capela</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Bam Adebayo</td>
      <td>Jonathan Isaac</td>
      <td>Clint Capela</td>
      <td>Jonathan Isaac</td>
      <td>Karl-Anthony Towns</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Clint Capela</td>
      <td>Bam Adebayo</td>
      <td>Joel Embiid</td>
      <td>Rudy Gobert</td>
      <td>Jonathan Isaac</td>
    </tr>
    <tr>
      <th>15</th>
      <td>DeMar DeRozan</td>
      <td>Clint Capela</td>
      <td>LeBron James</td>
      <td>Clint Capela</td>
      <td>Joel Embiid</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Joel Embiid</td>
      <td>Damian Lillard</td>
      <td>Rudy Gobert</td>
      <td>Bam Adebayo</td>
      <td>LeBron James</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Damian Lillard</td>
      <td>LeBron James</td>
      <td>Jonathan Isaac</td>
      <td>Joel Embiid</td>
      <td>Chris Paul</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Jonathan Isaac</td>
      <td>Joel Embiid</td>
      <td>Chris Paul</td>
      <td>Damian Lillard</td>
      <td>Damian Lillard</td>
    </tr>
    <tr>
      <th>19</th>
      <td>LeBron James</td>
      <td>Russell Westbrook</td>
      <td>Damian Lillard</td>
      <td>LeBron James</td>
      <td>DeMar DeRozan</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Chris Paul</td>
      <td>Richaun Holmes</td>
      <td>Russell Westbrook</td>
      <td>DeMar DeRozan</td>
      <td>Rudy Gobert</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Richaun Holmes</td>
      <td>Nikola Vucevic</td>
      <td>Richaun Holmes</td>
      <td>Chris Paul</td>
      <td>Richaun Holmes</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Rudy Gobert</td>
      <td>DeMar DeRozan</td>
      <td>Bam Adebayo</td>
      <td>Russell Westbrook</td>
      <td>Russell Westbrook</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Russell Westbrook</td>
      <td>Chris Paul</td>
      <td>DeMar DeRozan</td>
      <td>Jayson Tatum</td>
      <td>Bam Adebayo</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Nikola Vucevic</td>
      <td>Rudy Gobert</td>
      <td>Nikola Vucevic</td>
      <td>Richaun Holmes</td>
      <td>Nikola Vucevic</td>
    </tr>
    <tr>
      <th>25</th>
      <td>LaMarcus Aldridge</td>
      <td>LaMarcus Aldridge</td>
      <td>LaMarcus Aldridge</td>
      <td>LaMarcus Aldridge</td>
      <td>Luka Doncic</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Bradley Beal</td>
      <td>Luka Doncic</td>
      <td>Bradley Beal</td>
      <td>Nikola Vucevic</td>
      <td>LaMarcus Aldridge</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Brandon Ingram</td>
      <td>Jayson Tatum</td>
      <td>Luka Doncic</td>
      <td>Bradley Beal</td>
      <td>Jayson Tatum</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Devin Booker</td>
      <td>Khris Middleton</td>
      <td>Jayson Tatum</td>
      <td>Khris Middleton</td>
      <td>Khris Middleton</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Luka Doncic</td>
      <td>Brandon Ingram</td>
      <td>Khris Middleton</td>
      <td>Domantas Sabonis</td>
      <td>Bradley Beal</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Domantas Sabonis</td>
      <td>Bradley Beal</td>
      <td>Jrue Holiday</td>
      <td>Luka Doncic</td>
      <td>Pascal Siakam</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Shai Gilgeous-Alexander</td>
      <td>Domantas Sabonis</td>
      <td>Pascal Siakam</td>
      <td>Brandon Ingram</td>
      <td>Brandon Ingram</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Khris Middleton</td>
      <td>Jrue Holiday</td>
      <td>Domantas Sabonis</td>
      <td>Devin Booker</td>
      <td>Shai Gilgeous-Alexander</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Pascal Siakam</td>
      <td>Pascal Siakam</td>
      <td>Devin Booker</td>
      <td>Pascal Siakam</td>
      <td>Domantas Sabonis</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Kristaps Porzingis</td>
      <td>Devin Booker</td>
      <td>Shai Gilgeous-Alexander</td>
      <td>Kristaps Porzingis</td>
      <td>Devin Booker</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Jrue Holiday</td>
      <td>Shai Gilgeous-Alexander</td>
      <td>Brandon Ingram</td>
      <td>Jrue Holiday</td>
      <td>Kyle Lowry</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Jayson Tatum</td>
      <td>Kristaps Porzingis</td>
      <td>Kristaps Porzingis</td>
      <td>Stephen Curry</td>
      <td>Kristaps Porzingis</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Fred VanVleet</td>
      <td>Mitchell Robinson</td>
      <td>Kyle Lowry</td>
      <td>Kyle Lowry</td>
      <td>Jrue Holiday</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Kyle Lowry</td>
      <td>Kyle Lowry</td>
      <td>Stephen Curry</td>
      <td>Gordon Hayward</td>
      <td>Mitchell Robinson</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Gordon Hayward</td>
      <td>Stephen Curry</td>
      <td>Gordon Hayward</td>
      <td>Shai Gilgeous-Alexander</td>
      <td>Trae Young</td>
    </tr>
    <tr>
      <th>40</th>
      <td>Mitchell Robinson</td>
      <td>Fred VanVleet</td>
      <td>T.J. Warren</td>
      <td>Trae Young</td>
      <td>Stephen Curry</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Stephen Curry</td>
      <td>Jonas Valanciunas</td>
      <td>Mitchell Robinson</td>
      <td>Mitchell Robinson</td>
      <td>Jonas Valanciunas</td>
    </tr>
    <tr>
      <th>42</th>
      <td>Jonas Valanciunas</td>
      <td>Trae Young</td>
      <td>Jonas Valanciunas</td>
      <td>Jonas Valanciunas</td>
      <td>Gordon Hayward</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Dejounte Murray</td>
      <td>Dejounte Murray</td>
      <td>Fred VanVleet</td>
      <td>Paul George</td>
      <td>Paul George</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Paul George</td>
      <td>Gordon Hayward</td>
      <td>Kelly Oubre</td>
      <td>Ricky Rubio</td>
      <td>Fred VanVleet</td>
    </tr>
    <tr>
      <th>45</th>
      <td>T.J. Warren</td>
      <td>Paul George</td>
      <td>Trae Young</td>
      <td>Fred VanVleet</td>
      <td>Zach LaVine</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Trae Young</td>
      <td>Kelly Oubre</td>
      <td>Ricky Rubio</td>
      <td>Dejounte Murray</td>
      <td>Kelly Oubre</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Ricky Rubio</td>
      <td>Zach LaVine</td>
      <td>Tobias Harris</td>
      <td>Tobias Harris</td>
      <td>T.J. Warren</td>
    </tr>
    <tr>
      <th>48</th>
      <td>Tobias Harris</td>
      <td>Tobias Harris</td>
      <td>Dejounte Murray</td>
      <td>Kelly Oubre</td>
      <td>Dejounte Murray</td>
    </tr>
    <tr>
      <th>49</th>
      <td>Robert Covington</td>
      <td>Ricky Rubio</td>
      <td>Zach LaVine</td>
      <td>Zach LaVine</td>
      <td>Jamal Murray</td>
    </tr>
    <tr>
      <th>50</th>
      <td>Zach LaVine</td>
      <td>T.J. Warren</td>
      <td>Robert Covington</td>
      <td>T.J. Warren</td>
      <td>Tobias Harris</td>
    </tr>
    <tr>
      <th>51</th>
      <td>Kelly Oubre</td>
      <td>Robert Covington</td>
      <td>Will Barton</td>
      <td>Robert Covington</td>
      <td>Ricky Rubio</td>
    </tr>
    <tr>
      <th>52</th>
      <td>Donovan Mitchell</td>
      <td>Nerlens Noel</td>
      <td>De'Aaron Fox</td>
      <td>Kemba Walker</td>
      <td>Will Barton</td>
    </tr>
    <tr>
      <th>53</th>
      <td>Jamal Murray</td>
      <td>Jamal Murray</td>
      <td>Paul George</td>
      <td>Myles Turner</td>
      <td>Robert Covington</td>
    </tr>
    <tr>
      <th>54</th>
      <td>Marcus Smart</td>
      <td>Donovan Mitchell</td>
      <td>Nerlens Noel</td>
      <td>Will Barton</td>
      <td>Nerlens Noel</td>
    </tr>
    <tr>
      <th>55</th>
      <td>Steven Adams</td>
      <td>Montrezl Harrell</td>
      <td>Jamal Murray</td>
      <td>Norman Powell</td>
      <td>Kemba Walker</td>
    </tr>
    <tr>
      <th>56</th>
      <td>Mikal Bridges</td>
      <td>Will Barton</td>
      <td>Steven Adams</td>
      <td>De'Aaron Fox</td>
      <td>Jarrett Allen</td>
    </tr>
    <tr>
      <th>57</th>
      <td>Nerlens Noel</td>
      <td>Myles Turner</td>
      <td>Norman Powell</td>
      <td>Jamal Murray</td>
      <td>Steven Adams</td>
    </tr>
    <tr>
      <th>58</th>
      <td>Derrick Favors</td>
      <td>Mikal Bridges</td>
      <td>Myles Turner</td>
      <td>Steven Adams</td>
      <td>Mikal Bridges</td>
    </tr>
    <tr>
      <th>59</th>
      <td>Norman Powell</td>
      <td>CJ McCollum</td>
      <td>Jarrett Allen</td>
      <td>Jarrett Allen</td>
      <td>CJ McCollum</td>
    </tr>
    <tr>
      <th>60</th>
      <td>Montrezl Harrell</td>
      <td>Norman Powell</td>
      <td>Mikal Bridges</td>
      <td>Jaylen Brown</td>
      <td>Norman Powell</td>
    </tr>
    <tr>
      <th>61</th>
      <td>Kemba Walker</td>
      <td>Derrick Favors</td>
      <td>Donovan Mitchell</td>
      <td>Eric Bledsoe</td>
      <td>Jaylen Brown</td>
    </tr>
    <tr>
      <th>62</th>
      <td>Daniel Theis</td>
      <td>OG Anunoby</td>
      <td>OG Anunoby</td>
      <td>Nerlens Noel</td>
      <td>Myles Turner</td>
    </tr>
    <tr>
      <th>63</th>
      <td>Eric Bledsoe</td>
      <td>Kemba Walker</td>
      <td>Brook Lopez</td>
      <td>Donovan Mitchell</td>
      <td>Donovan Mitchell</td>
    </tr>
    <tr>
      <th>64</th>
      <td>Myles Turner</td>
      <td>Jarrett Allen</td>
      <td>Malcolm Brogdon</td>
      <td>OG Anunoby</td>
      <td>OG Anunoby</td>
    </tr>
    <tr>
      <th>65</th>
      <td>Jaylen Brown</td>
      <td>Jaylen Brown</td>
      <td>Jaylen Brown</td>
      <td>Daniel Theis</td>
      <td>Montrezl Harrell</td>
    </tr>
    <tr>
      <th>66</th>
      <td>Will Barton</td>
      <td>De'Aaron Fox</td>
      <td>Derrick Favors</td>
      <td>Derrick Favors</td>
      <td>Jeremy Lamb</td>
    </tr>
    <tr>
      <th>67</th>
      <td>OG Anunoby</td>
      <td>Eric Bledsoe</td>
      <td>Eric Bledsoe</td>
      <td>Montrezl Harrell</td>
      <td>Derrick Favors</td>
    </tr>
    <tr>
      <th>68</th>
      <td>De'Aaron Fox</td>
      <td>Marcus Smart</td>
      <td>Kemba Walker</td>
      <td>Brook Lopez</td>
      <td>Eric Bledsoe</td>
    </tr>
    <tr>
      <th>69</th>
      <td>Jeremy Lamb</td>
      <td>Jabari Parker</td>
      <td>Kris Dunn</td>
      <td>CJ McCollum</td>
      <td>Al Horford</td>
    </tr>
    <tr>
      <th>70</th>
      <td>Jarrett Allen</td>
      <td>Steven Adams</td>
      <td>Elfrid Payton</td>
      <td>Mikal Bridges</td>
      <td>Jabari Parker</td>
    </tr>
    <tr>
      <th>71</th>
      <td>Aaron Gordon</td>
      <td>Daniel Theis</td>
      <td>CJ McCollum</td>
      <td>Jabari Parker</td>
      <td>De'Aaron Fox</td>
    </tr>
    <tr>
      <th>72</th>
      <td>Al Horford</td>
      <td>Derrick Rose</td>
      <td>Jabari Parker</td>
      <td>Kris Dunn</td>
      <td>Daniel Theis</td>
    </tr>
    <tr>
      <th>73</th>
      <td>CJ McCollum</td>
      <td>Andrew Wiggins</td>
      <td>Montrezl Harrell</td>
      <td>Al Horford</td>
      <td>Brandon Clarke</td>
    </tr>
    <tr>
      <th>74</th>
      <td>Kris Dunn</td>
      <td>Kris Dunn</td>
      <td>Al Horford</td>
      <td>Malcolm Brogdon</td>
      <td>Kris Dunn</td>
    </tr>
    <tr>
      <th>75</th>
      <td>Jabari Parker</td>
      <td>Zion Williamson</td>
      <td>Brandon Clarke</td>
      <td>Jeremy Lamb</td>
      <td>Derrick Rose</td>
    </tr>
    <tr>
      <th>76</th>
      <td>Brandon Clarke</td>
      <td>Nemanja Bjelica</td>
      <td>Marcus Smart</td>
      <td>Marcus Smart</td>
      <td>Marcus Smart</td>
    </tr>
    <tr>
      <th>77</th>
      <td>Malcolm Brogdon</td>
      <td>Brook Lopez</td>
      <td>Daniel Theis</td>
      <td>Lonzo Ball</td>
      <td>Brook Lopez</td>
    </tr>
    <tr>
      <th>78</th>
      <td>Rui Hachimura</td>
      <td>Ja Morant</td>
      <td>Nemanja Bjelica</td>
      <td>Danilo Gallinari</td>
      <td>Elfrid Payton</td>
    </tr>
    <tr>
      <th>79</th>
      <td>Donte DiVincenzo</td>
      <td>Brandon Clarke</td>
      <td>Lonzo Ball</td>
      <td>Derrick Jones</td>
      <td>Andrew Wiggins</td>
    </tr>
    <tr>
      <th>80</th>
      <td>Lonzo Ball</td>
      <td>Malcolm Brogdon</td>
      <td>Jeremy Lamb</td>
      <td>Zion Williamson</td>
      <td>Danilo Gallinari</td>
    </tr>
    <tr>
      <th>81</th>
      <td>Brook Lopez</td>
      <td>Al Horford</td>
      <td>Zion Williamson</td>
      <td>Brandon Clarke</td>
      <td>Malcolm Brogdon</td>
    </tr>
    <tr>
      <th>82</th>
      <td>Wendell Carter</td>
      <td>Danilo Gallinari</td>
      <td>Aaron Gordon</td>
      <td>Andrew Wiggins</td>
      <td>Zion Williamson</td>
    </tr>
    <tr>
      <th>83</th>
      <td>Nemanja Bjelica</td>
      <td>D'Angelo Russell</td>
      <td>Andrew Wiggins</td>
      <td>Derrick Rose</td>
      <td>Kevin Love</td>
    </tr>
    <tr>
      <th>84</th>
      <td>Serge Ibaka</td>
      <td>Jeremy Lamb</td>
      <td>Danilo Gallinari</td>
      <td>Willie Cauley-Stein</td>
      <td>Derrick Jones</td>
    </tr>
    <tr>
      <th>85</th>
      <td>Thomas Bryant</td>
      <td>Elfrid Payton</td>
      <td>Donte DiVincenzo</td>
      <td>Serge Ibaka</td>
      <td>Nemanja Bjelica</td>
    </tr>
    <tr>
      <th>86</th>
      <td>Derrick Jones</td>
      <td>Rui Hachimura</td>
      <td>DeAndre Jordan</td>
      <td>Donte DiVincenzo</td>
      <td>Aaron Gordon</td>
    </tr>
    <tr>
      <th>87</th>
      <td>Evan Fournier</td>
      <td>Derrick Jones</td>
      <td>Derrick White</td>
      <td>Alec Burks</td>
      <td>Willie Cauley-Stein</td>
    </tr>
    <tr>
      <th>88</th>
      <td>Elfrid Payton</td>
      <td>Donte DiVincenzo</td>
      <td>Kevin Love</td>
      <td>Elfrid Payton</td>
      <td>Larry Nance</td>
    </tr>
    <tr>
      <th>89</th>
      <td>Andrew Wiggins</td>
      <td>Alec Burks</td>
      <td>D'Angelo Russell</td>
      <td>Larry Nance</td>
      <td>Jaren Jackson</td>
    </tr>
    <tr>
      <th>90</th>
      <td>Derrick Rose</td>
      <td>Evan Fournier</td>
      <td>Alec Burks</td>
      <td>Nemanja Bjelica</td>
      <td>D'Angelo Russell</td>
    </tr>
    <tr>
      <th>91</th>
      <td>Zion Williamson</td>
      <td>Draymond Green</td>
      <td>Derrick Jones</td>
      <td>DeAndre Jordan</td>
      <td>Alec Burks</td>
    </tr>
    <tr>
      <th>92</th>
      <td>Danuel House</td>
      <td>Collin Sexton</td>
      <td>Willie Cauley-Stein</td>
      <td>Kevin Love</td>
      <td>Donte DiVincenzo</td>
    </tr>
    <tr>
      <th>93</th>
      <td>Willie Cauley-Stein</td>
      <td>Lonzo Ball</td>
      <td>Draymond Green</td>
      <td>Wendell Carter</td>
      <td>Rui Hachimura</td>
    </tr>
    <tr>
      <th>94</th>
      <td>Marquese Chriss</td>
      <td>Damion Lee</td>
      <td>Thomas Bryant</td>
      <td>Markelle Fultz</td>
      <td>Lonzo Ball</td>
    </tr>
    <tr>
      <th>95</th>
      <td>Danilo Gallinari</td>
      <td>Larry Nance</td>
      <td>Serge Ibaka</td>
      <td>Ja Morant</td>
      <td>Wendell Carter</td>
    </tr>
    <tr>
      <th>96</th>
      <td>Glenn Robinson</td>
      <td>Willie Cauley-Stein</td>
      <td>Evan Fournier</td>
      <td>Thomas Bryant</td>
      <td>DeAndre Jordan</td>
    </tr>
    <tr>
      <th>97</th>
      <td>Collin Sexton</td>
      <td>Tomas Satoransky</td>
      <td>Rui Hachimura</td>
      <td>D'Angelo Russell</td>
      <td>Glenn Robinson</td>
    </tr>
    <tr>
      <th>98</th>
      <td>Tomas Satoransky</td>
      <td>Wendell Carter</td>
      <td>Derrick Rose</td>
      <td>Draymond Green</td>
      <td>Draymond Green</td>
    </tr>
    <tr>
      <th>99</th>
      <td>Derrick White</td>
      <td>DeAndre Jordan</td>
      <td>Ja Morant</td>
      <td>Rui Hachimura</td>
      <td>Serge Ibaka</td>
    </tr>
  </tbody>
</table>
</div>



# Reflection
There is not that much difference in value across different league size as I was expencting. 

Is this a valid approach with a larger player size? What if there was 10,000 players and 30 teams?
This process would still be possible with methods to filter the population quicker and learn who are the eligible players.

With perfect information of the season ahead, this methodology does a good job of ranking players. Future steps will involve predicting player season. The good thing about what I have created is, you can add in players/rookies and edit stats for the year and run the simulation over again. For example, you will probably want to edit Klay Thompson and Steph Curry who were injured with 2020-2021 predicted stats.

# Next Steps
Part 2: Develop a system to draft in a snake draft in process. Let us take this concept for a test drive in a mock draft.


```python

```
