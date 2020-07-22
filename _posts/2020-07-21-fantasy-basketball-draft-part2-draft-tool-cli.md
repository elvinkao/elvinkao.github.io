# Yahoo Fantasy Basketball Draft Part 2 - Mock Draft Run

In Part 1, we ran simulations to see which players are most valuable to a team and increase category win percentage. Now we will need to create some additional functions so that we are able to draft the best available player that will increase our win percentage. In order to do this, we need to update the create_team function to factor in teams in the league player choices already, what positions they play and what positions they still need to fill. 

### Filling a team's roster positions
When a team drafts a player, the roster should fill from most specific roster spot to the more general UTIL roster spot where any player can fill. Teams will fill up roster spots based on a player being the least flexible to the most flexible. For eg. If a player only plays SG, then you would want them to fill the specific SG position over someone who plays PG, SG, SF, PF. This allows more flexibility for the rest of the roster.

In order to do this, we add an additional column to the player attribute, eligible_positions. During the create_teams_in_progress process, we see whether for a given team, if there is already a player assigned for a given position. When there are no more players for the position, then we start randomly selecting players to the team.


```python
def __init__(self, filename):
    self.__player_df = pd.read_csv(filename)
    self.__filename = filename
    self.__player_df = self.__player_df.loc[:, ~self.__player_df.columns.str.contains('^Unnamed')]
    
    # add number of positions played
    self.__player_df['num_elig_positions'] = self.__player_df['is_g'] + self.__player_df['is_pg'] \
        + self.__player_df['is_sg'] + self.__player_df['is_f'] \
        + self.__player_df['is_sf'] + self.__player_df['is_pf'] \
        + self.__player_df['is_c'] + self.__player_df['is_util']
```


```python
def select_pg(self, player_df):
    # return player with least positions played and plays PG
    df = player_df[player_df['is_pg'] == 1]
    min_positions = df.num_elig_positions.min()
    df = df[df['num_elig_positions'] == min_positions]
    if len(df) > 0:
        return df.sample()
    else:
        return df
```


```python
def create_teams_in_progress(self, num_teams):
    # For each team, find which positions left to fill and find a random player and assign to the team
    # End of player selection process, only keep drafted players
    # ----------------------
    # Get team performance compared to other teams
    # Number of categories won compared to other teams
    # Number of head to head matchups won compared to other teams
    # return summary dataframe (player, rand_team, performance metrics)
    
    draft_players = self.__player_df.copy()

    # Change order of choosing players
    # Fill more specific spots first for validation if team has filled the spot, then more general
    
    for i in range(1,num_teams + 1):
        
        # get players on team being evaluated
        curr_team_players = draft_players[draft_players['rand_team'] == i]
        
        # find a PG
        pg_returned = self.select_pg(curr_team_players)
        if(len(pg_returned) == 0):
            rand_player = draft_players[(draft_players['is_pg']==1) & (draft_players['rand_team']==0)].sample()
            rand_player = rand_player.Player.to_string(index=False).strip()
            draft_players.loc[draft_players['Player'] == rand_player, ['rand_team']] = i
        else:
            # remove PG from curr_team_players
            curr_team_players = curr_team_players[~curr_team_players.isin(pg_returned)]

        # find a SG
        sg_returned = self.select_sg(curr_team_players)
        if(len(pg_returned) == 0):
            rand_player = draft_players[(draft_players['is_sg']==1) & (draft_players['rand_team']==0)].sample()
            rand_player = rand_player.Player.to_string(index=False).strip()
            draft_players.loc[draft_players['Player'] == rand_player, ['rand_team']] = i
        else:
            # remove SG from curr_team_players
            curr_team_players = curr_team_players[~curr_team_players.isin(sg_returned)]

        # find a SF
        sf_returned = self.select_sf(curr_team_players)
        if(len(sg_returned) == 0):
            rand_player = draft_players[(draft_players['is_sf']==1) & (draft_players['rand_team']==0)].sample()
            rand_player = rand_player.Player.to_string(index=False).strip()
            draft_players.loc[draft_players['Player'] == rand_player, ['rand_team']] = i
        else:
            # remove SF from curr_team_players
            curr_team_players = curr_team_players[~curr_team_players.isin(sf_returned)]

        # find a PF
        pf_returned = self.select_pf(curr_team_players)
        if(len(pf_returned) == 0):
            rand_player = draft_players[(draft_players['is_pf']==1) & (draft_players['rand_team']==0)].sample()
            rand_player = rand_player.Player.to_string(index=False).strip()
            draft_players.loc[draft_players['Player'] == rand_player, ['rand_team']] = i
        else:
            # remove PF from curr_team_players
            curr_team_players = curr_team_players[~curr_team_players.isin(pf_returned)]

        # find a C
        c_returned = self.select_c(curr_team_players)
        if(len(c_returned) == 0):
            rand_player = draft_players[(draft_players['is_c']==1) & (draft_players['rand_team']==0)].sample()
            rand_player = rand_player.Player.to_string(index=False).strip()
            draft_players.loc[draft_players['Player'] == rand_player, ['rand_team']] = i
        else:
            # remove C from curr_team_players
            curr_team_players = curr_team_players[~curr_team_players.isin(c_returned)]

        # find a C
        c_returned = self.select_c(curr_team_players)
        if(len(c_returned) == 0):
            rand_player = draft_players[(draft_players['is_c']==1) & (draft_players['rand_team']==0)].sample()
            rand_player = rand_player.Player.to_string(index=False).strip()
            draft_players.loc[draft_players['Player'] == rand_player, ['rand_team']] = i
        else:
            # remove C from curr_team_players
            curr_team_players = curr_team_players[~curr_team_players.isin(c_returned)]
        
        # find a G
        g_returned = self.select_g(curr_team_players)
        if(len(g_returned) == 0):
            rand_player = draft_players[(draft_players['is_g']==1) & (draft_players['rand_team']==0)].sample()
            rand_player = rand_player.Player.to_string(index=False).strip()
            draft_players.loc[draft_players['Player'] == rand_player, ['rand_team']] = i
        else:
            # remove G from curr_team_players
            curr_team_players = curr_team_players[~curr_team_players.isin(g_returned)]
        
        # find a F
        f_returned = self.select_f(curr_team_players)
        if(len(f_returned) == 0):
            rand_player = draft_players[(draft_players['is_f']==1) & (draft_players['rand_team']==0)].sample()
            rand_player = rand_player.Player.to_string(index=False).strip()
            draft_players.loc[draft_players['Player'] == rand_player, ['rand_team']] = i
        else:
            # remove F from curr_team_players
            curr_team_players = curr_team_players[~curr_team_players.isin(f_returned)]

        # find a Util
        util_returned = self.select_util(curr_team_players)
        if(len(util_returned) == 0):
            rand_player = draft_players[(draft_players['is_util']==1) & (draft_players['rand_team']==0)].sample()
            rand_player = rand_player.Player.to_string(index=False).strip()
            draft_players.loc[draft_players['Player'] == rand_player, ['rand_team']] = i
        else:
            # remove Util from curr_team_players
            curr_team_players = curr_team_players[~curr_team_players.isin(util_returned)]

        # find a Util
        util_returned = self.select_util(curr_team_players)
        if(len(util_returned) == 0):
            rand_player = draft_players[(draft_players['is_util']==1) & (draft_players['rand_team']==0)].sample()
            rand_player = rand_player.Player.to_string(index=False).strip()
            draft_players.loc[draft_players['Player'] == rand_player, ['rand_team']] = i
        else:
            # remove Util from curr_team_players
            curr_team_players = curr_team_players[~curr_team_players.isin(util_returned)]

        # find a Util
        util_returned = self.select_util(curr_team_players)
        if(len(util_returned) == 0):
            rand_player = draft_players[(draft_players['is_util']==1) & (draft_players['rand_team']==0)].sample()
            rand_player = rand_player.Player.to_string(index=False).strip()
            draft_players.loc[draft_players['Player'] == rand_player, ['rand_team']] = i
        else:
            # remove Util from curr_team_players
            curr_team_players = curr_team_players[~curr_team_players.isin(util_returned)]

        # find a Util
        util_returned = self.select_util(curr_team_players)
        if(len(util_returned) == 0):
            rand_player = draft_players[(draft_players['is_util']==1) & (draft_players['rand_team']==0)].sample()
            rand_player = rand_player.Player.to_string(index=False).strip()
            draft_players.loc[draft_players['Player'] == rand_player, ['rand_team']] = i
        else:
            # remove Util from curr_team_players
            curr_team_players = curr_team_players[~curr_team_players.isin(util_returned)]

        # find a Util
        util_returned = self.select_util(curr_team_players)
        if(len(util_returned) == 0):
            rand_player = draft_players[(draft_players['is_util']==1) & (draft_players['rand_team']==0)].sample()
            rand_player = rand_player.Player.to_string(index=False).strip()
            draft_players.loc[draft_players['Player'] == rand_player, ['rand_team']] = i
        else:
            # remove Util from curr_team_players
            curr_team_players = curr_team_players[~curr_team_players.isin(util_returned)]

        # find a Util
        util_returned = self.select_util(curr_team_players)
        if(len(util_returned) == 0):
            rand_player = draft_players[(draft_players['is_util']==1) & (draft_players['rand_team']==0)].sample()
            rand_player = rand_player.Player.to_string(index=False).strip()
            draft_players.loc[draft_players['Player'] == rand_player, ['rand_team']] = i
        else:
            # remove Util from curr_team_players
            curr_team_players = curr_team_players[~curr_team_players.isin(util_returned)]
```

### MAKE IT GO FASTER!!!
I do a lot of my work in Jupyter Notebooks. That means, most of the Python programming that I do is simple scripts and I had planned to be able to use a notebook as a tool, but it performed too slow and too difficult to switch between functions of adding players and drafting during my turn. Working in command line and calling the functions while passing arguments was also taking quite a bit of time. So instead, I used the package [PyInquirer](https://github.com/CITGuru/PyInquirer), which is library for command line interface. So it is easy to toggle through functions, search name by a couple of characters, and add them to a team.

Another challenge I faced was having parallel processing end by execution time rather than a set number of iterations. What I ended up doing was creating a while loop for a certain time limit and perform parallel processing in 50 iterative chunks. Draft times can range from 30 - 120 seconds, so would have to set the while loop time accordingly. 


```python
def run_simulations(self, num_teams):
    iteration_size = 50
    starttime = time.perf_counter()

    self.save_csv(self.__filename)
    self.__player_df = pd.read_csv(self.__filename)

    # first create_team
    draft_players = self.create_teams_in_progress(num_teams)
    num_simulations = 1

    elapsed_time = round(time.perf_counter()-starttime,2)
    while elapsed_time < 25:

      with concurrent.futures.ProcessPoolExecutor() as executor:

        # list comprehension
        results = [executor.submit(self.create_teams_in_progress, num_teams) for _ in range(iteration_size)]

        for f in concurrent.futures.as_completed(results):
          draft_players = draft_players.append(f.result(), ignore_index=True)    

        num_simulations = num_simulations + iteration_size
        elapsed_time = round(time.perf_counter()-starttime,2)

    # team 1 will be our team, so we want to see the player that benefits our team the most
    sim_summary = draft_players[draft_players['rand_team']==1].groupby(['Player'])[['cat_win','cat_loss','matchup_win','matchup_loss']].apply(sum).reset_index()
    sim_summary['cat_perc'] = sim_summary['cat_win'] / (sim_summary['cat_win'] + sim_summary['cat_loss'])
    sim_summary['matchup_perc'] = sim_summary['matchup_win'] / (sim_summary['matchup_win'] + sim_summary['matchup_loss'])
    print(sim_summary.sort_values(by=['cat_perc'], ascending=False).head(20))

    endtime = time.perf_counter()
    print(f'Finished in {round(endtime-starttime,2)} seconds(s) with {num_simulations} simulations')
```

### Test Run
Unfortunately, because of Covid, a lot of the mock nba fantasy draft applications are currently down. I have did a short test run.
> youtube: https://youtu.be/O2Cq3FFWLkA

I drafted the players in order in last post with 10,000 simulations in a 6 person league. I have set the time limit to 30 seconds and with the 2nd pick, the best player in 201 simulations was Ben Simmons with a category win percentage of 72% against random sample for other teams. Then going through the ranks under 10,000 simulations, the 3rd pick with 151 simulations was Mikal Bridges with category win percentage of 43%. That would mean I would now have a losing winning percentage. That changed rather quickly. It is clear that running under simulations under this current format is just not fast enough to choose the right person and under small sample, at pick 2, choosing the wrong person was very consequential. This tool will be more effective with 2 minute draft times and during later stages when optimizing for player categories. 

### Final Thoughts
I got to learn some interesting tricks with concurrent.futures to parallel process and workarounds when time constrained and you want to run as much parallel simulations as possible. It was also fun to play around with the CLI tool to possibly make some small applications in the future.


```python

```
