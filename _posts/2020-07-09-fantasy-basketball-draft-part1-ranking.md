# Yahoo Fantasy Basketball Draft Part 1 - Snake draft value / best available player

Year after year, I have been part of a NBA fantasy league. Year after year, I have been losing. Not just losing, but not even making the playoffs. Imagine someone who's career is in data analytics, completely failing at something where evaluating data is key. Well, I have had enough and there is only so much pounding a person can take. ENOUGH!

It is time to put my superpowers to the test.

This will be a multiple part series of blog posts as I explore what is the best way to win at fantasy sports. The league that I am in is the most difficult because it is an auction draft keeper league. Where it is not just important to have the best players and evaluate based on rank, but how to properly price players will be very important.

Let us start with the basics. Assuming we have perfect data on a players performance for the current year, how do you build the most optimal team. There are many different strategies to doing this. For this example league, assume a 9 category, head-to-head matchup league. There are a couple of approaches people have to this problem, the most common would be to choose 5 or 6 out of the 9 categories and create a linear optimization problem where based on constraints, maximizing over those categories.

My gripe with this strategy is it is not flexible and is not able to adapt based on what other fantasy owners are doing during the draft. What if majority of other fantasy owners are going after the same categories and players are taken off the board?

### Fantasy Player Rankings
Below is a list of top players taken from https://www.fantraxhq.com/2019-fantasy-basketball-rankings/. The analyst here says that turnover category is not considered, but ranking is based on per-game production. I have always been curious if value of a player changes based on the size of the league. For eg, does having a player that over-indexes on shot blocking and provides nothing else more beneficial in a 4 person league than a 12 person league? 
My problem with ranking based on overall production metric is that if there are many players that over index in assists and steals and few that over index in rebounds and blocks, then you may miss the underlying value of players.

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
            hh_cat_win = hh_cat_win + hhwins
            hh_cat_loss = hh_cat_loss + hhlosses

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
      <td>11200.0</td>
      <td>7745.0</td>
      <td>1510.0</td>
      <td>595.0</td>
      <td>0.591185</td>
      <td>0.717340</td>
    </tr>
    <tr>
      <th>132</th>
      <td>James Harden</td>
      <td>9009.0</td>
      <td>6246.0</td>
      <td>1210.0</td>
      <td>485.0</td>
      <td>0.590560</td>
      <td>0.713864</td>
    </tr>
    <tr>
      <th>165</th>
      <td>Kawhi Leonard</td>
      <td>8779.0</td>
      <td>6296.0</td>
      <td>1176.0</td>
      <td>499.0</td>
      <td>0.582355</td>
      <td>0.702090</td>
    </tr>
    <tr>
      <th>164</th>
      <td>Karl-Anthony Towns</td>
      <td>9562.0</td>
      <td>7043.0</td>
      <td>1255.0</td>
      <td>590.0</td>
      <td>0.575851</td>
      <td>0.680217</td>
    </tr>
    <tr>
      <th>186</th>
      <td>Kyrie Irving</td>
      <td>8814.0</td>
      <td>6666.0</td>
      <td>1169.0</td>
      <td>551.0</td>
      <td>0.569380</td>
      <td>0.679651</td>
    </tr>
    <tr>
      <th>52</th>
      <td>Damian Lillard</td>
      <td>7750.0</td>
      <td>5885.0</td>
      <td>1020.0</td>
      <td>495.0</td>
      <td>0.568390</td>
      <td>0.673267</td>
    </tr>
    <tr>
      <th>147</th>
      <td>John Collins</td>
      <td>10536.0</td>
      <td>8274.0</td>
      <td>1341.0</td>
      <td>749.0</td>
      <td>0.560128</td>
      <td>0.641627</td>
    </tr>
    <tr>
      <th>241</th>
      <td>Paul George</td>
      <td>7955.0</td>
      <td>6445.0</td>
      <td>1015.0</td>
      <td>585.0</td>
      <td>0.552431</td>
      <td>0.634375</td>
    </tr>
    <tr>
      <th>283</th>
      <td>Trae Young</td>
      <td>7171.0</td>
      <td>5834.0</td>
      <td>915.0</td>
      <td>530.0</td>
      <td>0.551403</td>
      <td>0.633218</td>
    </tr>
    <tr>
      <th>192</th>
      <td>LeBron James</td>
      <td>11333.0</td>
      <td>9277.0</td>
      <td>1440.0</td>
      <td>850.0</td>
      <td>0.549879</td>
      <td>0.628821</td>
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
      <td>17520.0</td>
      <td>11080.0</td>
      <td>2218.0</td>
      <td>1357.0</td>
      <td>0.612587</td>
      <td>0.620420</td>
    </tr>
    <tr>
      <th>1</th>
      <td>James Harden</td>
      <td>12132.0</td>
      <td>8508.0</td>
      <td>1445.0</td>
      <td>1135.0</td>
      <td>0.587791</td>
      <td>0.560078</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Kawhi Leonard</td>
      <td>12492.0</td>
      <td>8908.0</td>
      <td>1471.0</td>
      <td>1204.0</td>
      <td>0.583738</td>
      <td>0.549907</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Kyrie Irving</td>
      <td>13217.0</td>
      <td>9463.0</td>
      <td>1556.0</td>
      <td>1279.0</td>
      <td>0.582760</td>
      <td>0.548854</td>
    </tr>
    <tr>
      <th>4</th>
      <td>John Collins</td>
      <td>16638.0</td>
      <td>12002.0</td>
      <td>1972.0</td>
      <td>1608.0</td>
      <td>0.580936</td>
      <td>0.550838</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Hassan Whiteside</td>
      <td>12553.0</td>
      <td>9247.0</td>
      <td>1487.0</td>
      <td>1238.0</td>
      <td>0.575826</td>
      <td>0.545688</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Jimmy Butler</td>
      <td>12365.0</td>
      <td>9355.0</td>
      <td>1419.0</td>
      <td>1296.0</td>
      <td>0.569291</td>
      <td>0.522652</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Nikola Jokic</td>
      <td>16066.0</td>
      <td>12254.0</td>
      <td>1865.0</td>
      <td>1675.0</td>
      <td>0.567302</td>
      <td>0.526836</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Chris Paul</td>
      <td>9703.0</td>
      <td>7457.0</td>
      <td>1124.0</td>
      <td>1021.0</td>
      <td>0.565443</td>
      <td>0.524009</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Karl-Anthony Towns</td>
      <td>12902.0</td>
      <td>9938.0</td>
      <td>1496.0</td>
      <td>1359.0</td>
      <td>0.564886</td>
      <td>0.523993</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Bradley Beal</td>
      <td>9490.0</td>
      <td>7390.0</td>
      <td>1073.0</td>
      <td>1037.0</td>
      <td>0.562204</td>
      <td>0.508531</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Damian Lillard</td>
      <td>10021.0</td>
      <td>7819.0</td>
      <td>1133.0</td>
      <td>1097.0</td>
      <td>0.561715</td>
      <td>0.508072</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Deandre Ayton</td>
      <td>12189.0</td>
      <td>9531.0</td>
      <td>1406.0</td>
      <td>1309.0</td>
      <td>0.561188</td>
      <td>0.517864</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Richaun Holmes</td>
      <td>16545.0</td>
      <td>13055.0</td>
      <td>1846.0</td>
      <td>1854.0</td>
      <td>0.558953</td>
      <td>0.498919</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Giannis Antetokounmpo</td>
      <td>11715.0</td>
      <td>9245.0</td>
      <td>1394.0</td>
      <td>1226.0</td>
      <td>0.558922</td>
      <td>0.532061</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Jonathan Isaac</td>
      <td>12010.0</td>
      <td>9510.0</td>
      <td>1386.0</td>
      <td>1304.0</td>
      <td>0.558086</td>
      <td>0.515242</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Andre Drummond</td>
      <td>16652.0</td>
      <td>13188.0</td>
      <td>1962.0</td>
      <td>1768.0</td>
      <td>0.558043</td>
      <td>0.526005</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Clint Capela</td>
      <td>15562.0</td>
      <td>12358.0</td>
      <td>1767.0</td>
      <td>1723.0</td>
      <td>0.557378</td>
      <td>0.506304</td>
    </tr>
    <tr>
      <th>18</th>
      <td>DeMar DeRozan</td>
      <td>12458.0</td>
      <td>9902.0</td>
      <td>1369.0</td>
      <td>1426.0</td>
      <td>0.557156</td>
      <td>0.489803</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Ben Simmons</td>
      <td>9960.0</td>
      <td>7920.0</td>
      <td>1137.0</td>
      <td>1098.0</td>
      <td>0.557047</td>
      <td>0.508725</td>
    </tr>
  </tbody>
</table>
</div>



# Parallel Time Improvement
The parallel run improved the run time for the 2000 simulations from 1000 seconds to 253 seconds. That is a 4x improvement. Draft time usually allows for 2 minutes per round, so will need to run slightly less simulations, but as teams become filled and number of permutations decreases, it will need less simulations to become accurate. 

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
      <td>James Harden</td>
      <td>Kawhi Leonard</td>
      <td>Kawhi Leonard</td>
      <td>Kawhi Leonard</td>
      <td>Kawhi Leonard</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Kawhi Leonard</td>
      <td>James Harden</td>
      <td>James Harden</td>
      <td>Kyrie Irving</td>
      <td>James Harden</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Kyrie Irving</td>
      <td>Kyrie Irving</td>
      <td>Hassan Whiteside</td>
      <td>James Harden</td>
      <td>Kyrie Irving</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Hassan Whiteside</td>
      <td>Hassan Whiteside</td>
      <td>Kyrie Irving</td>
      <td>Hassan Whiteside</td>
      <td>Hassan Whiteside</td>
    </tr>
    <tr>
      <th>5</th>
      <td>John Collins</td>
      <td>John Collins</td>
      <td>John Collins</td>
      <td>Jimmy Butler</td>
      <td>Jimmy Butler</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Jimmy Butler</td>
      <td>Jimmy Butler</td>
      <td>Jimmy Butler</td>
      <td>John Collins</td>
      <td>John Collins</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Nikola Jokic</td>
      <td>Jonathan Isaac</td>
      <td>Nikola Jokic</td>
      <td>Nikola Jokic</td>
      <td>Jonathan Isaac</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Damian Lillard</td>
      <td>Nikola Jokic</td>
      <td>Deandre Ayton</td>
      <td>Karl-Anthony Towns</td>
      <td>Karl-Anthony Towns</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Deandre Ayton</td>
      <td>Damian Lillard</td>
      <td>Karl-Anthony Towns</td>
      <td>Deandre Ayton</td>
      <td>Nikola Jokic</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Karl-Anthony Towns</td>
      <td>Karl-Anthony Towns</td>
      <td>Damian Lillard</td>
      <td>Damian Lillard</td>
      <td>Deandre Ayton</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Jonathan Isaac</td>
      <td>Chris Paul</td>
      <td>Giannis Antetokounmpo</td>
      <td>Jonathan Isaac</td>
      <td>Ben Simmons</td>
    </tr>
    <tr>
      <th>12</th>
      <td>DeMar DeRozan</td>
      <td>Giannis Antetokounmpo</td>
      <td>Ben Simmons</td>
      <td>Ben Simmons</td>
      <td>Clint Capela</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Bam Adebayo</td>
      <td>Richaun Holmes</td>
      <td>Chris Paul</td>
      <td>Chris Paul</td>
      <td>Damian Lillard</td>
    </tr>
    <tr>
      <th>14</th>
      <td>LeBron James</td>
      <td>Deandre Ayton</td>
      <td>DeMar DeRozan</td>
      <td>Richaun Holmes</td>
      <td>Chris Paul</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Giannis Antetokounmpo</td>
      <td>DeMar DeRozan</td>
      <td>Richaun Holmes</td>
      <td>Clint Capela</td>
      <td>Giannis Antetokounmpo</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Chris Paul</td>
      <td>Andre Drummond</td>
      <td>LeBron James</td>
      <td>Giannis Antetokounmpo</td>
      <td>DeMar DeRozan</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Richaun Holmes</td>
      <td>Clint Capela</td>
      <td>Joel Embiid</td>
      <td>Rudy Gobert</td>
      <td>Andre Drummond</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Clint Capela</td>
      <td>Joel Embiid</td>
      <td>Jonathan Isaac</td>
      <td>DeMar DeRozan</td>
      <td>Richaun Holmes</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Rudy Gobert</td>
      <td>Bam Adebayo</td>
      <td>Clint Capela</td>
      <td>Andre Drummond</td>
      <td>Russell Westbrook</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Ben Simmons</td>
      <td>Ben Simmons</td>
      <td>Andre Drummond</td>
      <td>Bam Adebayo</td>
      <td>LeBron James</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Russell Westbrook</td>
      <td>LeBron James</td>
      <td>Bam Adebayo</td>
      <td>LeBron James</td>
      <td>Joel Embiid</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Andre Drummond</td>
      <td>LaMarcus Aldridge</td>
      <td>LaMarcus Aldridge</td>
      <td>Joel Embiid</td>
      <td>Bam Adebayo</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Jayson Tatum</td>
      <td>Rudy Gobert</td>
      <td>Russell Westbrook</td>
      <td>Bradley Beal</td>
      <td>Rudy Gobert</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Devin Booker</td>
      <td>Russell Westbrook</td>
      <td>Rudy Gobert</td>
      <td>Russell Westbrook</td>
      <td>LaMarcus Aldridge</td>
    </tr>
    <tr>
      <th>25</th>
      <td>LaMarcus Aldridge</td>
      <td>Nikola Vucevic</td>
      <td>Bradley Beal</td>
      <td>Khris Middleton</td>
      <td>Bradley Beal</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Joel Embiid</td>
      <td>Devin Booker</td>
      <td>Nikola Vucevic</td>
      <td>Jayson Tatum</td>
      <td>Nikola Vucevic</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Khris Middleton</td>
      <td>Jayson Tatum</td>
      <td>Khris Middleton</td>
      <td>Nikola Vucevic</td>
      <td>Jayson Tatum</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Shai Gilgeous-Alexander</td>
      <td>Pascal Siakam</td>
      <td>Devin Booker</td>
      <td>LaMarcus Aldridge</td>
      <td>Khris Middleton</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Nikola Vucevic</td>
      <td>Khris Middleton</td>
      <td>Stephen Curry</td>
      <td>Devin Booker</td>
      <td>Devin Booker</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Jrue Holiday</td>
      <td>Luka Doncic</td>
      <td>Trae Young</td>
      <td>Brandon Ingram</td>
      <td>Brandon Ingram</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Bradley Beal</td>
      <td>Stephen Curry</td>
      <td>Luka Doncic</td>
      <td>Stephen Curry</td>
      <td>Luka Doncic</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Brandon Ingram</td>
      <td>Jrue Holiday</td>
      <td>Kyle Lowry</td>
      <td>Jrue Holiday</td>
      <td>Stephen Curry</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Stephen Curry</td>
      <td>Brandon Ingram</td>
      <td>Fred VanVleet</td>
      <td>Shai Gilgeous-Alexander</td>
      <td>Jrue Holiday</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Luka Doncic</td>
      <td>Bradley Beal</td>
      <td>Pascal Siakam</td>
      <td>Domantas Sabonis</td>
      <td>Shai Gilgeous-Alexander</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Gordon Hayward</td>
      <td>Shai Gilgeous-Alexander</td>
      <td>Jayson Tatum</td>
      <td>Luka Doncic</td>
      <td>Pascal Siakam</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Jonas Valanciunas</td>
      <td>Trae Young</td>
      <td>Shai Gilgeous-Alexander</td>
      <td>Mitchell Robinson</td>
      <td>Domantas Sabonis</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Fred VanVleet</td>
      <td>Fred VanVleet</td>
      <td>Brandon Ingram</td>
      <td>Pascal Siakam</td>
      <td>Gordon Hayward</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Domantas Sabonis</td>
      <td>Kristaps Porzingis</td>
      <td>Jrue Holiday</td>
      <td>Trae Young</td>
      <td>Mitchell Robinson</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Trae Young</td>
      <td>Gordon Hayward</td>
      <td>Mitchell Robinson</td>
      <td>Fred VanVleet</td>
      <td>Fred VanVleet</td>
    </tr>
    <tr>
      <th>40</th>
      <td>Kyle Lowry</td>
      <td>Kyle Lowry</td>
      <td>Domantas Sabonis</td>
      <td>Gordon Hayward</td>
      <td>Kristaps Porzingis</td>
    </tr>
    <tr>
      <th>41</th>
      <td>T.J. Warren</td>
      <td>Jonas Valanciunas</td>
      <td>Gordon Hayward</td>
      <td>Kyle Lowry</td>
      <td>Trae Young</td>
    </tr>
    <tr>
      <th>42</th>
      <td>Mitchell Robinson</td>
      <td>Domantas Sabonis</td>
      <td>Kristaps Porzingis</td>
      <td>Jonas Valanciunas</td>
      <td>Kyle Lowry</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Ricky Rubio</td>
      <td>Mitchell Robinson</td>
      <td>Ricky Rubio</td>
      <td>Kristaps Porzingis</td>
      <td>T.J. Warren</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Zach LaVine</td>
      <td>Ricky Rubio</td>
      <td>T.J. Warren</td>
      <td>Paul George</td>
      <td>Jonas Valanciunas</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Pascal Siakam</td>
      <td>Paul George</td>
      <td>Jonas Valanciunas</td>
      <td>Ricky Rubio</td>
      <td>Ricky Rubio</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Dejounte Murray</td>
      <td>Tobias Harris</td>
      <td>Paul George</td>
      <td>T.J. Warren</td>
      <td>Paul George</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Kristaps Porzingis</td>
      <td>Zach LaVine</td>
      <td>Dejounte Murray</td>
      <td>Tobias Harris</td>
      <td>Tobias Harris</td>
    </tr>
    <tr>
      <th>48</th>
      <td>Paul George</td>
      <td>Robert Covington</td>
      <td>Jamal Murray</td>
      <td>Nerlens Noel</td>
      <td>Jamal Murray</td>
    </tr>
    <tr>
      <th>49</th>
      <td>Jamal Murray</td>
      <td>T.J. Warren</td>
      <td>Nerlens Noel</td>
      <td>Dejounte Murray</td>
      <td>Dejounte Murray</td>
    </tr>
    <tr>
      <th>50</th>
      <td>Robert Covington</td>
      <td>Donovan Mitchell</td>
      <td>Robert Covington</td>
      <td>Mikal Bridges</td>
      <td>Robert Covington</td>
    </tr>
    <tr>
      <th>51</th>
      <td>OG Anunoby</td>
      <td>Nerlens Noel</td>
      <td>Zach LaVine</td>
      <td>Myles Turner</td>
      <td>Kelly Oubre</td>
    </tr>
    <tr>
      <th>52</th>
      <td>Kemba Walker</td>
      <td>Norman Powell</td>
      <td>Kris Dunn</td>
      <td>Zach LaVine</td>
      <td>Nerlens Noel</td>
    </tr>
    <tr>
      <th>53</th>
      <td>Kelly Oubre</td>
      <td>Jamal Murray</td>
      <td>Kelly Oubre</td>
      <td>Jamal Murray</td>
      <td>Zach LaVine</td>
    </tr>
    <tr>
      <th>54</th>
      <td>Jarrett Allen</td>
      <td>Kelly Oubre</td>
      <td>Will Barton</td>
      <td>Kelly Oubre</td>
      <td>Myles Turner</td>
    </tr>
    <tr>
      <th>55</th>
      <td>Norman Powell</td>
      <td>Will Barton</td>
      <td>Tobias Harris</td>
      <td>Derrick Favors</td>
      <td>Mikal Bridges</td>
    </tr>
    <tr>
      <th>56</th>
      <td>Donovan Mitchell</td>
      <td>Daniel Theis</td>
      <td>Myles Turner</td>
      <td>Robert Covington</td>
      <td>Norman Powell</td>
    </tr>
    <tr>
      <th>57</th>
      <td>Elfrid Payton</td>
      <td>Brook Lopez</td>
      <td>Norman Powell</td>
      <td>Norman Powell</td>
      <td>Eric Bledsoe</td>
    </tr>
    <tr>
      <th>58</th>
      <td>Will Barton</td>
      <td>Dejounte Murray</td>
      <td>Mikal Bridges</td>
      <td>Daniel Theis</td>
      <td>Marcus Smart</td>
    </tr>
    <tr>
      <th>59</th>
      <td>Nerlens Noel</td>
      <td>Mikal Bridges</td>
      <td>Brook Lopez</td>
      <td>Kemba Walker</td>
      <td>Donovan Mitchell</td>
    </tr>
    <tr>
      <th>60</th>
      <td>Kris Dunn</td>
      <td>Derrick Favors</td>
      <td>Derrick Favors</td>
      <td>Donovan Mitchell</td>
      <td>Kris Dunn</td>
    </tr>
    <tr>
      <th>61</th>
      <td>De'Aaron Fox</td>
      <td>Kemba Walker</td>
      <td>Donovan Mitchell</td>
      <td>OG Anunoby</td>
      <td>Al Horford</td>
    </tr>
    <tr>
      <th>62</th>
      <td>Tobias Harris</td>
      <td>Al Horford</td>
      <td>Kemba Walker</td>
      <td>Brook Lopez</td>
      <td>OG Anunoby</td>
    </tr>
    <tr>
      <th>63</th>
      <td>Montrezl Harrell</td>
      <td>Malcolm Brogdon</td>
      <td>Brandon Clarke</td>
      <td>Jabari Parker</td>
      <td>Jarrett Allen</td>
    </tr>
    <tr>
      <th>64</th>
      <td>Brook Lopez</td>
      <td>Jarrett Allen</td>
      <td>OG Anunoby</td>
      <td>Will Barton</td>
      <td>Brook Lopez</td>
    </tr>
    <tr>
      <th>65</th>
      <td>Eric Bledsoe</td>
      <td>Jabari Parker</td>
      <td>Danilo Gallinari</td>
      <td>Steven Adams</td>
      <td>Montrezl Harrell</td>
    </tr>
    <tr>
      <th>66</th>
      <td>Derrick Favors</td>
      <td>Kris Dunn</td>
      <td>Eric Bledsoe</td>
      <td>Al Horford</td>
      <td>Daniel Theis</td>
    </tr>
    <tr>
      <th>67</th>
      <td>Marcus Smart</td>
      <td>Eric Bledsoe</td>
      <td>Jaylen Brown</td>
      <td>Kris Dunn</td>
      <td>Derrick Favors</td>
    </tr>
    <tr>
      <th>68</th>
      <td>Al Horford</td>
      <td>Myles Turner</td>
      <td>Derrick Jones</td>
      <td>Marcus Smart</td>
      <td>Jeremy Lamb</td>
    </tr>
    <tr>
      <th>69</th>
      <td>Steven Adams</td>
      <td>Marcus Smart</td>
      <td>Montrezl Harrell</td>
      <td>Eric Bledsoe</td>
      <td>Kemba Walker</td>
    </tr>
    <tr>
      <th>70</th>
      <td>Mikal Bridges</td>
      <td>Derrick Rose</td>
      <td>Jabari Parker</td>
      <td>Derrick Jones</td>
      <td>Jaylen Brown</td>
    </tr>
    <tr>
      <th>71</th>
      <td>Myles Turner</td>
      <td>Jeremy Lamb</td>
      <td>Malcolm Brogdon</td>
      <td>Brandon Clarke</td>
      <td>Will Barton</td>
    </tr>
    <tr>
      <th>72</th>
      <td>Jeremy Lamb</td>
      <td>Brandon Clarke</td>
      <td>Jarrett Allen</td>
      <td>De'Aaron Fox</td>
      <td>De'Aaron Fox</td>
    </tr>
    <tr>
      <th>73</th>
      <td>Brandon Clarke</td>
      <td>De'Aaron Fox</td>
      <td>De'Aaron Fox</td>
      <td>CJ McCollum</td>
      <td>CJ McCollum</td>
    </tr>
    <tr>
      <th>74</th>
      <td>Ja Morant</td>
      <td>OG Anunoby</td>
      <td>Daniel Theis</td>
      <td>Jarrett Allen</td>
      <td>Malcolm Brogdon</td>
    </tr>
    <tr>
      <th>75</th>
      <td>Jabari Parker</td>
      <td>Jaylen Brown</td>
      <td>Steven Adams</td>
      <td>Nemanja Bjelica</td>
      <td>Jabari Parker</td>
    </tr>
    <tr>
      <th>76</th>
      <td>Daniel Theis</td>
      <td>Steven Adams</td>
      <td>Derrick Rose</td>
      <td>Montrezl Harrell</td>
      <td>Steven Adams</td>
    </tr>
    <tr>
      <th>77</th>
      <td>CJ McCollum</td>
      <td>Montrezl Harrell</td>
      <td>Al Horford</td>
      <td>Derrick Rose</td>
      <td>Brandon Clarke</td>
    </tr>
    <tr>
      <th>78</th>
      <td>Derrick Rose</td>
      <td>Elfrid Payton</td>
      <td>Evan Fournier</td>
      <td>Jeremy Lamb</td>
      <td>Derrick Rose</td>
    </tr>
    <tr>
      <th>79</th>
      <td>Derrick Jones</td>
      <td>CJ McCollum</td>
      <td>DeAndre Jordan</td>
      <td>Malcolm Brogdon</td>
      <td>Danilo Gallinari</td>
    </tr>
    <tr>
      <th>80</th>
      <td>Draymond Green</td>
      <td>Willie Cauley-Stein</td>
      <td>Marcus Smart</td>
      <td>Lonzo Ball</td>
      <td>Alec Burks</td>
    </tr>
    <tr>
      <th>81</th>
      <td>Donte DiVincenzo</td>
      <td>Lonzo Ball</td>
      <td>CJ McCollum</td>
      <td>Danilo Gallinari</td>
      <td>Zion Williamson</td>
    </tr>
    <tr>
      <th>82</th>
      <td>Danilo Gallinari</td>
      <td>Derrick Jones</td>
      <td>Jeremy Lamb</td>
      <td>Jaylen Brown</td>
      <td>Elfrid Payton</td>
    </tr>
    <tr>
      <th>83</th>
      <td>Jaylen Brown</td>
      <td>Alec Burks</td>
      <td>Willie Cauley-Stein</td>
      <td>Willie Cauley-Stein</td>
      <td>Derrick Jones</td>
    </tr>
    <tr>
      <th>84</th>
      <td>D'Angelo Russell</td>
      <td>Evan Fournier</td>
      <td>Alec Burks</td>
      <td>Alec Burks</td>
      <td>Derrick White</td>
    </tr>
    <tr>
      <th>85</th>
      <td>Kevin Love</td>
      <td>Danilo Gallinari</td>
      <td>Elfrid Payton</td>
      <td>Zion Williamson</td>
      <td>Evan Fournier</td>
    </tr>
    <tr>
      <th>86</th>
      <td>Evan Fournier</td>
      <td>Thomas Bryant</td>
      <td>Derrick White</td>
      <td>Elfrid Payton</td>
      <td>Kevin Love</td>
    </tr>
    <tr>
      <th>87</th>
      <td>Collin Sexton</td>
      <td>Draymond Green</td>
      <td>Andrew Wiggins</td>
      <td>DeAndre Jordan</td>
      <td>Thomas Bryant</td>
    </tr>
    <tr>
      <th>88</th>
      <td>Malcolm Brogdon</td>
      <td>Rui Hachimura</td>
      <td>Ja Morant</td>
      <td>D'Angelo Russell</td>
      <td>Lonzo Ball</td>
    </tr>
    <tr>
      <th>89</th>
      <td>Lonzo Ball</td>
      <td>Andrew Wiggins</td>
      <td>Nemanja Bjelica</td>
      <td>Derrick White</td>
      <td>Draymond Green</td>
    </tr>
    <tr>
      <th>90</th>
      <td>Larry Nance</td>
      <td>Wendell Carter</td>
      <td>Donte DiVincenzo</td>
      <td>Donte DiVincenzo</td>
      <td>Nemanja Bjelica</td>
    </tr>
    <tr>
      <th>91</th>
      <td>Delon Wright</td>
      <td>D'Angelo Russell</td>
      <td>Marquese Chriss</td>
      <td>Evan Fournier</td>
      <td>Andrew Wiggins</td>
    </tr>
    <tr>
      <th>92</th>
      <td>Andrew Wiggins</td>
      <td>Markelle Fultz</td>
      <td>Draymond Green</td>
      <td>Serge Ibaka</td>
      <td>DeAndre Jordan</td>
    </tr>
    <tr>
      <th>93</th>
      <td>Marquese Chriss</td>
      <td>Kevin Love</td>
      <td>D'Angelo Russell</td>
      <td>Patrick Beverley</td>
      <td>Willie Cauley-Stein</td>
    </tr>
    <tr>
      <th>94</th>
      <td>Alec Burks</td>
      <td>Larry Nance</td>
      <td>Serge Ibaka</td>
      <td>Ja Morant</td>
      <td>Serge Ibaka</td>
    </tr>
    <tr>
      <th>95</th>
      <td>Jaren Jackson</td>
      <td>Aaron Gordon</td>
      <td>Lonzo Ball</td>
      <td>Tomas Satoransky</td>
      <td>Paul Millsap</td>
    </tr>
    <tr>
      <th>96</th>
      <td>Jae Crowder</td>
      <td>Derrick White</td>
      <td>Justin Holiday</td>
      <td>Andrew Wiggins</td>
      <td>Collin Sexton</td>
    </tr>
    <tr>
      <th>97</th>
      <td>Glenn Robinson</td>
      <td>DeAndre Jordan</td>
      <td>Rui Hachimura</td>
      <td>Kevin Love</td>
      <td>Glenn Robinson</td>
    </tr>
    <tr>
      <th>98</th>
      <td>DeAndre Jordan</td>
      <td>Ja Morant</td>
      <td>Glenn Robinson</td>
      <td>Larry Nance</td>
      <td>Aaron Gordon</td>
    </tr>
    <tr>
      <th>99</th>
      <td>Nemanja Bjelica</td>
      <td>Nemanja Bjelica</td>
      <td>Wendell Carter</td>
      <td>Marquese Chriss</td>
      <td>Rui Hachimura</td>
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
