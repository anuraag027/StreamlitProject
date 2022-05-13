import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from soccerplots.radar_chart import Radar
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import math

import warnings
warnings.filterwarnings('ignore')

# ADDING CACHE OPTION
@st.cache
def load_data():
    return pd.read_csv('Data/All_stats_combined_with_positions.csv')
df = load_data()

st.header('Use Machine Learning to Find Similar Players')
st.caption('Select a player, and my Machine Learning model will suggest players similar to the selected player.')
st.caption('Currently works for outfield positions for players from Europe\'s Top 5 Leagues - Premier League, La Liga, Seria A, Ligue 1 and Bundesliga.')
st.caption('Made by: Anuraag Kulkarni   |   Twitter: @Anuraag027')

#Function for Radar Plot
def plot_radar(bkup,df,n):
    
    #Perform a merge on scaled dataframe and original dataframe
    new = pd.merge(df,bkup,on=['Player'],indicator=True)
    #Remove the scaled columns
    new = new.loc[:, ~new.columns.str.contains("_x")]

    new.columns = new.columns.str.rstrip('_y')
    df = new.copy()
    
    #get list of column names as parameters
    params = list(df.columns)
    #Remove unwanted columns and keep only metric columns
    params = params[5:-1]
    new_params = []
    for i in range(len(params)):
        new_params.append(params[i].split('(')[0].strip())
    
    #Update column names by adding '\n' to prevent clatter in radar plot
    new = []
    for c in new_params:
        if len(c.split(' ')) >= 2:
            temp = c.split(' ')
            col_str = ''
            for i in range(len(temp)):
                if (i == 0) or (i == 2) or (i == 4):# or (i == 4):
                    col_str += temp[i] + '\n'
                else:
                    col_str += temp[i]
                col_str += ' '
            new.append(col_str)
        else:
            new.append(c)

    new_params = new
    try_list = new
    
    #add ranges to list of tuple pairs
    ranges = []
    values = []

    #Form minimum and maximum values for radar plot
    for x in params:
        a = min(df[params][x])
        a = a - (a*.15)

        b = max(df[params][x])
        b = b + (b*.15)

        ranges.append((a,b))

    for i in range(n):
        values.append(df.iloc[i].values.tolist()[5:-1])
        
    colors_list = ['red','blue','yellow','pink','green','orange']
    colors_list = colors_list[:n]

    #List of alphas for every player's chart. Currently keeping as the same
    alpha_list = [0.7,0.7,0.7,0.7,0.7,0.7]
    alpha_list = alpha_list[:n]

    endnote = 'Twitter: @Anuraag027'

    radar = Radar()

    fig,ax = radar.plot_radar(ranges=ranges,params=new_params,values=values,
                             radar_color=colors_list,
                             alphas=alpha_list,endnote=endnote,#title=title,endnote=endnote,
                             compare=True)
    
    #Get player name with color used for them in bracket
    temp_str = ''
    for i in range(n):
        if i != n - 1:
            temp_str += df['Player'][i] + ' (' + colors_list[i] + ') vs '
        else:
            temp_str += df['Player'][i] + ' (' + colors_list[i] + ')'
    
    plt.title(temp_str,color='black',size=20,fontfamily='Candara')
    st.write(fig)
    
    #This try_list will be needed for metric descriptions
    return try_list
    
# df = pd.read_csv('./All_stats_combined_with_positions.csv')

#Remove the Unnamed & rank columns
df = df.iloc[:,3:]

# Widget sidebar, "with" notation
with st.sidebar:
    #Select the player
    player = st.selectbox(
         'Choose Player',
        df['Player'])

    #Get the position for that player
    pos = df[df['Player'] == player]['TMPosition'].values[0]
    
    st.write(player,"plays at position:",pos)
    
    #Age slider
    start_age, end_age = st.slider(
     'Select a range of age',
     value=[17, 25],step=1,min_value=17,max_value=int(df['Age'].max()))

    st.write('Suggested players will be between the ages',str(start_age),'and',str(end_age)+'.')

    #Choose the threshold as the minimum of 12 and the 90s played by that player
    ninties = min(12,df['90s'][df[df['Player'] == player].index[0]])

    df = df[df['90s'] >= ninties]
    st.write("90s played by player:",df['90s'][df[df['Player'] == player].index[0]])
    st.write("90s selected as threshold:",ninties)
    
    #Display warning for 90s less than 8 as sample size is too small
    if ninties <= 8:
        st.caption("Minutes played by player is low, this can lead to inaccurate results due to small sample size")
    
    #Reset index
    df.reset_index(inplace=True)
    df.drop(['index'],axis=1,inplace=True)
    
    #Filter all players that play in that position
    df = df[df['TMPosition'] == pos]
    
    #Metric usage for every position
    if (pos == 'Right-Back') or (pos == 'Left-Back') or (pos == 'Left WingBack') or (pos == 'Right WingBack'):
        df = df[['Player','Squad','Age','Padj Tkl+Int p90 (defense_Padj_p90)','% of Dribblers Tackled (possession_p90)','Crs_p90 (passing_types_p90)',
                 'SCA_p90 (gca_p90)',
                 'xA_p90 (passing_p90)','Prog Actions_p90 (possession_p90)','Passes into Final 1/3_p90 (possession_p90)',
                 'Carries into Final 1/3_p90 (possession_p90)','KP_p90 (possession_p90)','TB_p90 (passing_types_p90)',
                 'True Interceptions_p90 (possession_p90)','Prog Carries_p90 per 100 touches (possession_p90)']]

    elif (pos == 'Right Winger') or (pos == 'Left Winger'):
        df = df[['Player','Squad','Age','xA_p90 (passing_p90)','KP_p90 (possession_p90)','Successful Dribbles_p90 (possession_p90)',
                 'Attempted Dribbles_p90 (possession_p90)','Passes into Penalty Area_p90 (possession_p90)',
                 'Carries into Penalty Area_p90 (possession_p90)','npxG_p90 (shooting_p90)', 'npxG/Shot (possession_p90)',
                 'Prog Actions_p90 (possession_p90)','Crs_p90 (passing_types_p90)','TB_p90 (passing_types_p90)',
                 'Prog Carries_p90 per 100 touches (possession_p90)', 'Sh_p90 (possession_p90)', 'SoT_p90 (possession_p90)', 
                 'Receiving Prog_p90 (possession_p90)','Prog Carries_p90 (possession_p90)','Prog Passes_p90 (possession_p90)',
                 'Prog Passes_p90 per 50 passes (possession_p90)']]

    elif pos == 'Defensive Midfield':
        df = df[['Player','Squad','Age','Successful Dribbles_p90 (possession_p90)',
                 'Padj Tkl+Int p90 (defense_Padj_p90)','% of Dribblers Tackled (possession_p90)',
                 'Completed Passes_p90 (passing_p90)','Long Cmp_p90 (passing_p90)','Long Att_p90 (passing_p90)',
                 'Press_p90 (passing_types_p90)','Prog Passes_p90 per 50 passes (possession_p90)',
                 'Prog Carries_p90 per 100 touches (possession_p90)',
                 'Passes into Final 1/3_p90 (possession_p90)','Prog Carries_p90 (possession_p90)',
                 'Prog Passes_p90 (possession_p90)','Clr_p90 (defense_Padj_p90)','True Interceptions_p90 (possession_p90)']]

    elif pos == 'Attacking Midfield':
        df = df[['Player','Squad','Age','Successful Dribbles_p90 (possession_p90)',
                 'Completed Passes_p90 (passing_p90)','Press_p90 (passing_types_p90)','Prog Carries_p90 per 100 touches (possession_p90)',
                 'Passes into Final 1/3_p90 (possession_p90)','Prog Carries_p90 (possession_p90)',
                 'Prog Passes_p90 (possession_p90)','Carries into Final 1/3_p90 (possession_p90)',
                 'Passes into Penalty Area_p90 (possession_p90)','Carries into Penalty Area_p90 (possession_p90)',
                 'xA_p90 (passing_p90)','KP_p90 (possession_p90)','npxG_p90 (shooting_p90)','TB_p90 (passing_types_p90)', 
                 'Receiving Prog_p90 (possession_p90)', 'npxG/Shot (possession_p90)']]

    elif pos == 'Central Midfield':
        df = df[['Player','Squad','Age','Successful Dribbles_p90 (possession_p90)',
                 'Padj Tkl+Int p90 (defense_Padj_p90)','Completed Passes_p90 (passing_p90)','Press_p90 (passing_types_p90)',
                 'Passes into Final 1/3_p90 (possession_p90)','Prog Carries_p90 (possession_p90)',
                 'Prog Carries_p90 per 100 touches (possession_p90)',
                 'Prog Passes_p90 (possession_p90)','Carries into Final 1/3_p90 (possession_p90)',
                 'Passes into Penalty Area_p90 (possession_p90)','xA_p90 (passing_p90)','KP_p90 (possession_p90)',
                 'TB_p90 (passing_types_p90)','Receiving Prog_p90 (possession_p90)']]

    elif pos == 'Centre-Forward':
        df = df[['Player','Squad','Age','xA_p90 (passing_p90)','Passes into Penalty Area_p90 (possession_p90)',
                 'Carries into Penalty Area_p90 (possession_p90)','npxG_p90 (shooting_p90)', 'npxG/Shot (possession_p90)',
                 'Sh_p90 (possession_p90)', 'SoT_p90 (possession_p90)','Aerial Win % (possession_p90)','Won_p90 (misc_p90)',
                 'Receiving Prog_p90 (possession_p90)','SCA_p90 (gca_p90)','KP_p90 (possession_p90)']]

    elif pos == 'Goalkeeper':
        df = df[['Player','Squad','Age','Completed Passes_p90 (passing_p90)','SoTA_p90 (keepers_p90)', 'Saves_p90 (keepers_p90)','Stp_p90 (keepersadv_p90)',
                 '#OPA_p90 (keepersadv_p90)', 'AvgDist_p90 (keepersadv_p90)','PSxG+/-_p90 (keepersadv_p90)', 'Cmp_p90 (keepersadv_p90)', 
                 'Att_p90 (keepersadv_p90)', 'Passes Att_p90 (keepersadv_p90)', 'Thr_p90 (keepersadv_p90)', 'AvgLen_p90 (keepersadv_p90)']]

    elif pos == 'Second Striker':
        df = df[['Player','Squad','Age','Successful Dribbles_p90 (possession_p90)', 'Sh_p90 (possession_p90)', 'SoT_p90 (possession_p90)',
                 'Passes into Penalty Area_p90 (possession_p90)','Carries into Penalty Area_p90 (possession_p90)',
                 'xA_p90 (passing_p90)','KP_p90 (possession_p90)','npxG_p90 (shooting_p90)','SCA_p90 (gca_p90)',
                 'Aerial Win % (possession_p90)','Won_p90 (misc_p90)', 'Receiving Prog_p90 (possession_p90)',
                 'npxG/Shot (possession_p90)']]

    elif pos == 'Centre-Back':
        df = df[['Player','Squad','Age',
                 'Padj Tkl+Int p90 (defense_Padj_p90)','% of Dribblers Tackled (possession_p90)',
                 'Completed Passes_p90 (passing_p90)','Long Cmp_p90 (passing_p90)','Long Att_p90 (passing_p90)',
                 'Prog Carries_p90 per 100 touches (possession_p90)',
                 'Prog Passes_p90 per 50 passes (possession_p90)','Passes into Final 1/3_p90 (possession_p90)',
                 'Prog Carries_p90 (possession_p90)','Prog Passes_p90 (possession_p90)','Clr_p90 (defense_Padj_p90)',
                 'Shots Blocked_p90 (possession_p90)', 'Pass_p90 (defense_Padj_p90)',
                 'Aerial Win % (possession_p90)','Won_p90 (misc_p90)','True Interceptions_p90 (possession_p90)']]        
   
    df = df.dropna()
    #Reset index
    df.reset_index(inplace=True)
    df.drop(['index'],axis=1,inplace=True)
    bkup_df = df.copy()
    
    #Scale all the required features, as some may be absolute values and some may be percentages
    scaler = StandardScaler()

    scaler.fit(df.drop(['Player','Squad','Age'],axis=1))
    scaled_features = scaler.transform(df.drop(['Player','Squad','Age'],axis=1))
    scaled_feat_df = pd.DataFrame(scaled_features,columns=df.columns[3:])
    
    df = pd.concat([df[['Player','Squad','Age']],scaled_feat_df],axis=1)
    
    #Get the scaled features
    X = np.array(df.iloc[:,3:])
    
    #Use the elbow method to find ideal number of clusters
    wcss = [] #within cluster sum of squares

    for i in range(1, 10):
        kmeans = KMeans(n_clusters = i)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    #Plotting the results onto a line graph, allowing us to observe 'The elbow'
    fig, ax = plt.subplots()
    ax.plot(range(1, 10), wcss)
    ax.scatter(range(1, 10), wcss)

    #Create a k-means model, fit the features and get cluster predictions. Add the predictions as a column
    #Currently using number of clusters as 4. SUBJECT TO CHANGE.
    kmeans = KMeans(n_clusters = 4,random_state=100)
    kmeans.fit(X)
    df['cluster'] = kmeans.predict(X)

    clus = df[df['Player'] == player]['cluster']

    df = df[df['cluster'] == int(clus)]    

    df.reset_index(inplace=True)
    df.drop(['index'],axis=1,inplace=True)

    #Get the list of values of the same cluster players for calculating similarity score
    player_list = df[df['Player'] == player].values.tolist()
    others_list = df[df['Player'] != player].values.tolist()

    ind = df[df['Player'] == player_list[0][0]].index[0]
    df['Similarity Score'] = ''
    df['Similarity Score'][ind] = 0
    for elem in others_list:
        sim_score = 0
        #Calculate similarity score using Euclidian distance
        for i in range(3,len(player_list[0])-1):
            sim_score += pow(player_list[0][i] - elem[i],2)
        sim_score = math.sqrt(sim_score)
        ind = df[df['Player'] == elem[0]].index
        df['Similarity Score'][ind] = sim_score
    
    #Filter for the players between the selected ages
    df = df[(df['Age'] >= start_age) & (df['Age'] <= end_age)]
    #Sort the dataframe according to similarity score
    df = df.sort_values('Similarity Score')

radar_df = df.copy()
df = df[['Player','Similarity Score']]
df.reset_index(inplace=True)
df.drop(['index'],axis=1,inplace=True)

#Left column for result dataframe
col1, col2 = st.columns([2,4])
with col1:
    st.subheader("Results")
    st.write(df)
#Right column for radar plot
with col2:
    st.subheader("Radar Chart")
    radar_df.reset_index(inplace=True)
    radar_df.drop(['index'],axis=1,inplace=True)
    
    #Radar will be plotted for 3 players APART from the selected player
    if df['Player'][0] == player:
        n = min(4,len(radar_df))
    else:
        n = min(3,len(radar_df))
    
    #Plot the radar
    try_list = plot_radar(bkup = bkup_df,df = radar_df,n = n)

#Metric descriptions
try_list = [ele.rstrip().replace('\n','') for ele in try_list]

test_dic = {
'Long Att_p90 (passing_p90)' : 'Long Passes Attempted per 90',
'Completed Passes_p90 (passing_p90)': 'Completed Passes per 90',
'Successful Dribbles_p90 (possession_p90)' : 'Successful Dribbles per 90',
'npxG_p90 (shooting_p90)' : 'Non Penalty xG per 90',
'% of Dribblers Tackled (possession_p90)' : 'Percentage of Dribblers Tackled',
'Receiving Prog_p90 (possession_p90)' : 'Progressive Passes Received per 90',
'Pass_p90 (defense_Padj_p90)' : 'Blocked Passes per 90',
'Prog Passes_p90 (possession_p90)' : 'Progressive Passes per 90',
'Crs_p90 (passing_types_p90)' : 'Crosses per 90',
'True Interceptions_p90 (possession_p90)' : 'True Interceptions = Interceptions per 90 + Blocked Shots per 90 + Blocked Passes per 90',
'xA_p90 (passing_p90)' : 'Expected Assists per 90',
'Carries into Penalty Area_p90 (possession_p90)' : 'Carries Into Penalty Area per 90',
'Padj Tkl+Int p90 (defense_Padj_p90)' : 'Possession Adjusted Tackles + Interceptions per 90',
'Attempted Dribbles_p90 (possession_p90)' : 'Attempted Dribbles per 90',
'Aerial Win % (possession_p90)' : '% of Aerial Duals Won',
'Prog Carries_p90 per 100 touches (possession_p90)' : 'Progressive Carries per 100 touches, per 90',
'Carries into Final 1/3_p90 (possession_p90)' : 'Carries into The Final Third per 90',
'Won_p90 (misc_p90)' : 'Aerial Duals Won per 90',
'Long Cmp_p90 (passing_p90)' : 'Long Passes Completed per 90',
'SoT_p90 (possession_p90)' : 'Shots on Target per 90',
'SCA_p90 (gca_p90)' : 'Shot Creating Actions per 90',
'Prog Actions_p90 (possession_p90)' : 'Progressive Actions = Progressive Passes per 90 + Progressive Carries per 90',
'Prog Carries_p90 (possession_p90)' : 'Progressive Carries per 90',
'Passes into Final 1/3_p90 (possession_p90)' : 'Passes into The Final Third per 90',
'Press_p90 (passing_types_p90)' : 'Passes Made Under Pressure per 90',
'Shots Blocked_p90 (possession_p90)' : 'Shots Blocked per 90',
'Prog Passes_p90 per 50 passes (possession_p90)' : 'Progressive Passes per 50 Completed Passes, per 90',
'TB_p90 (passing_types_p90)' : 'Through Balls per 90',
'Passes into Penalty Area_p90 (possession_p90)' : 'Passes Into Penalty Area per 90',
'KP_p90 (possession_p90)' : 'Key Passes per 90',
'Sh_p90 (possession_p90)' : 'Shots Taken per 90',
'npxG/Shot (possession_p90)' : 'Non Penalty xG Generated per Shot',
'Clr_p90 (defense_Padj_p90)' : 'Clearances Made per 90'}

st.caption('The lower the similarity score, the higher the similarity between the players. A similarity score of above 4.5 should be taken with a pinch of salt.')
    
st.subheader('Description for Metrics Used')

#Look for every column name in above dictionary (key) and print the appropriate description (value)
for ele in try_list:
    for k,v in test_dic.items():
        if ele in k:
            if ele == 'Prog Passes_p90':
                st.write(ele,':','Progressive Passes per 90')
                break
            if ele == 'Prog Carries_p90':
                st.write(ele,':','Progressive Carries per 90')
                break
            else:
                st.write(ele,' : ',v)
    
st.caption('Names of players are as taken as provided by Statsbomb on FBRef.com')
st.caption('Data has been taken from FBRef (Statsbomb): https://fbref.com/en/')
st.caption('Feel free to reach out to me on Twitter via DMs if you have any questions or if you face any issues with the app.')
st.caption('If you like my work, check out my articles on: https://thebeautifulgamereviewed.blogspot.com/')
