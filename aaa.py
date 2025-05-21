import numpy as np
import streamlit as st 
import pickle

data = pickle.load(open('scaler.pkl', 'rb'))
data2 = pickle.load(open('iplmodel_ridge.sav','rb'))


Batting_team = st.selectbox("Select a Batting team", options=['Options','Chennai Super Kings','Delhi Daredevils', 
        'Kings XI Punjab','Kolkata Knight Riders', 'Mumbai Indians',
        'Rajasthan Royals', 'Royal Challengers Bangalore','Sunrisers Hyderabad','Lucknow Super Giants','Gujrat Titans'])


Bowling_team = st.selectbox("Select a Bowling team", options=['Options','Chennai Super Kings','Delhi Daredevils', 
        'Kings XI Punjab','Kolkata Knight Riders', 'Mumbai Indians',
        'Rajasthan Royals', 'Royal Challengers Bangalore',
        'Sunrisers Hyderabad','Lucknow Super Giants','Gujrat Titans'])


Venue = st.selectbox("Select a Venue", options=['Options','ACA-VDCA Stadium, Visakhapatnam',
       'Barabati Stadium, Cuttack', 'Dr DY Patil Sports Academy, Mumbai',
       'Dubai International Cricket Stadium, Dubai',
       'Eden Gardens, Kolkata', 'Feroz Shah Kotla, Delhi',
       'Himachal Pradesh Cricket Association Stadium, Dharamshala',
       'Holkar Cricket Stadium, Indore',
       'JSCA International Stadium Complex, Ranchi',
       'M Chinnaswamy Stadium, Bangalore',
       'MA Chidambaram Stadium, Chepauk',
       'Maharashtra Cricket Association Stadium, Pune',
       'Punjab Cricket Association Stadium, Mohali',
       'Raipur International Cricket Stadium, Raipur',
       'Rajiv Gandhi International Stadium, Uppal',
       'Sardar Patel Stadium, Motera',
       'Sawai Mansingh Stadium, Jaipur',
       'Sharjah Cricket Stadium, Sharjah',
       'Sheikh Zayed Stadium, Abu-Dhabi',
       'Wankhede Stadium, Mumbai','Ekana Sports City, Lucknow','Narendra Modi Stadium, Gujrat'])


overs = st.number_input("Overs (>= 5.0)", max_value=20)

runs = st.number_input("Curr Runs")

wickets = st.number_input("Curr Wickets")

runs_last_5 = st.number_input("Runs scored in previous 5 Overs")

wickets_last_5 = st.number_input("Wickets taken in previous 5 Overs")


if st.button("Predict Score"):
    pipe = np.array([[runs, wickets, overs, runs_last_5, wickets_last_5]])
    predicted_score = data.predict(pipe)
    st.write("The predicted score is", predicted_score)
