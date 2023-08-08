import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from pycaret.regression import *


@st.cache_data
def load_data():
    # Load your data here
    data = pd.read_csv("bike-sharing_hourly.csv")
    data.to_csv("feature_data.csv")
    data.to_csv("prep_data.csv")
    return data

data = load_data()

#SIDEBAR
st.sidebar.header('Utilities')
# Add a button to reset the data
if st.sidebar.button('Reset data'):
    data = pd.read_csv("bike-sharing_hourly.csv")
    data.to_csv("feature_data.csv")
    data.to_csv("prep_data.csv")
st.sidebar.header('Original Data Info')
st.sidebar.markdown("""- `instant`: record index
- `dteday` : date
- `season` : season flag 
    - 1: springer
    - 2: summer 
    - 3: fall
    - 4: winter
- `yr` : year (0: 2011, 1:2012)
- `mnth` : month ( 1 to 12)
- `hr` : hour (0 to 23)
- `holiday` : flag holiday or not
- `weekday` : day of the week
- `workingday` : 
     - 1: weekend nor holiday
     - 0: weekend or holiday
+ `weathersit` : 
	- 1: Clear, Few clouds, Partly cloudy, Partly cloudy
	- 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
	- 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
	- 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
- `temp` : Normalized temperature in Celsius. The values are divided to 41 (max)
- `atemp`: Normalized feeling temperature in Celsius. The values are divided to 50 (max)
- `hum`: Normalized humidity. The values are divided to 100 (max)
- `windspeed`: Normalized wind speed. The values are divided to 67 (max)
- `casual`: count of casual users
- `registered`: count of registered users
- `cnt`: count of total rental bikes including both casual and registered (target)""")


#HEADER
#Image for the header
header_image = Image.open("header_image.jpg")
st.image(header_image, use_column_width=True)
#Title
st.title("Group 5 Assignment - Python for Data Analysis II")

#Section I - Exploratory analysis
st.header("Exploratory Data Analysis")
st.markdown("The data provided contains information about bike rentals in a city during the years 2011 and 2012. This is the data provided:")
data

st.markdown("The target variable in this case is the number of bikes rented (*cnt*).")
#Plot - Bike Rentals Over Time
#Time series for the number of bikes rented.
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['dteday'], y=data['cnt'], name='Bike Rentals', line=dict(color='orange')))
fig.update_layout(title='Bike Rentals Over Time', xaxis_title='Date', yaxis_title='Number of bikes rented', width=1200, height=500)
st.plotly_chart(fig, use_container_width = True)
st.markdown("It seems that there were **more bikes rented in 2012 that 2011 overall** and, there is some seasonality since there is a **drop on bikes rented around January**.")

#We need to fix the variable hum because of some inconsistency show in the selectbx object. We are going to do it here to make sure it is done even though
#the user does not select the hum variable
data_hum_0 = data[data["hum"] == 0]
#Mean from previou and following day
mean_prev_day = data[data["dteday"] == "2011-03-09"]["hum"].mean()
mean_foll_day = data[data["dteday"] == "2011-03-11"]["hum"].mean()
mean_hum = (mean_prev_day + mean_foll_day) / 2
#Impute the values of the dat 2011-03-10
data.loc[data["dteday"] == "2011-03-10", "hum"] = mean_hum

#Exploratory analysis of each of the features using a tab selector
st.markdown("---")
st.markdown("Let's go for a more in depth analysis on each of the features of the data.")
#Select box
tab = st.selectbox("Please, select the feature", ["Time based features - season, yr, mnth, hr", "holiday", "weekday", "weathersit", "temp", "atemp", "hum", "windspeed", "registered & casual"])
#Content of each tab
if tab == "Time based features - season, yr, mnth, hr":
    st.markdown("First, we are going to take a look on how the variable *season* affects the number of bikes rented")
    
    #Plot - Bike Rentals per season
    #Dic to create a new column with the labels of the seasons
    season_dict = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
    #Temp season label column
    data['season_labels'] = data['season'].replace(season_dict)
    #Groupby to obtain the total number of bikes rented per season
    season_counts = data.groupby('season_labels')['cnt'].sum().reset_index()
    #Plot the result
    colors = ["sandybrown", "seagreen", "coral", "steelblue"]
    fig = px.bar(season_counts, x='season_labels', y='cnt',color='season_labels',
             color_discrete_sequence=colors, labels={'season_labels': 'Season', 'cnt': 'Count'})
    fig.update_layout(title='Bike Rentals by Season', width=800, height=500)
    st.plotly_chart(fig, use_container_width = True)
    st.markdown("It can be seen that the season with the highest amount of bikes rented is **Fall, then Summer, Winter and Spring**.")
    
    st.markdown("Let's do the same thing but including the year (*yr*).")
    #Plot - Bike Rentals per Season and Year
    season_yr_count = data.groupby(["yr", "season_labels"])["cnt"].sum()
    #Plot the result
    fig = px.bar(season_yr_count.reset_index(), x='season_labels', y='cnt', color='yr',
                 labels={'season_labels': 'Season', 'cnt': 'Count', 'yr': 'Year'})
    fig.update_layout(title='Bike Rentals by Season and Year', width=800, height=500)
    st.plotly_chart(fig, use_container_width = True)
    st.markdown("There is no change in the distribution of the season between 2011 and 2012 (Fall, then Summer, Winter and Spring)")
    
    st.markdown("Lastly, let's study which is the distribution by hour of the day (*hr*).")
    #Group by hour and season
    data_season_hr = data.groupby(["season_labels", "hr"])["cnt"].sum()
    #Plot - Bike Rentals per Season and Hour
    fig = px.bar(data_season_hr.reset_index(), x='hr', y='cnt', color='season_labels', color_discrete_sequence=colors, barmode='group',
                 labels={'hr': 'Hour', 'cnt': 'Count', 'season_labels': 'Season'})
    fig.update_layout(title='Bike Rentals by Hour and Season', width=1200, height=400)
    st.plotly_chart(fig, use_container_width = True)
    st.markdown("After analysing each season in particular, the conclusion is that there are no real differences on the hours between seasons. There top and worst hours are the same across all seasons.")
    st.markdown("""- Top hours: 16 - 19 (job exits) and 8 (job enter).
- Worst hours: 0-6 (night).""")
    
    #Drop of the temporal column season_label
    data = data.drop(columns = "season_labels")

if tab == "holiday":
    st.markdown("Amount of bikes rented per *holiday*.")
    #Plot - Bike Rentals per Holiday
    #Groupby to obtain the total number of bikes rented per holiday
    holiday_count = data.groupby("holiday")["cnt"].sum().reset_index()
    #Plot the result
    fig = px.bar(holiday_count, x='holiday', y='cnt', labels={'holiday': 'Is it holiday ? ', 'cnt': 'Count'})
    fig.update_layout(title='Bike Rentals by Holiday', width=500, height=500)
    st.plotly_chart(fig, use_container_width = True)
    st.markdown("From the total amount of bikes rented, only **2.3% are rented during holidays**.")
    
if tab == "weekday":
    st.markdown("Amount of bikes rented per *weekday*.")
    #Plot - Bike Rentals per Weekday
    #Dic to create a new column with the labels of the weekdays
    weekday_dict = {0: 'Monday', 1: 'Tuesday',2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    #Temp season label column
    data['weekday_labels'] = data['weekday'].replace(weekday_dict)
    #Groupby to obtain the total number of bikes rented per weekday
    weekday_count = data.groupby("weekday_labels")["cnt"].sum().sort_values(ascending = False).reset_index()
    #Plot the result
    colors = ["sandybrown", "seagreen", "coral", "steelblue", "plum", "tan","darkseagreen" ]
    fig = px.bar(weekday_count, x='weekday_labels', y='cnt', labels={'weekday_labels': 'Weekday', 'cnt': 'Count'}, color = "weekday_labels",color_discrete_sequence = colors)
    fig.update_layout(title='Bike Rentals by Weekday', width=800, height=500)
    st.plotly_chart(fig, use_container_width = True)
    st.markdown("The days with more bikes rented are **Friday, Saturday and Sunday**. In any case, the differences are not extremelly relevant across the different days.")
    #Drop of the temporal column weekday_labels
    data = data.drop(columns = "weekday_labels")

if tab == "weathersit":
    st.markdown("Amount of bikes rented per *weathersit*.")
    #Plot - Bike Rentals per Weathersit
    #Dic to create a new column with the labels of the weathersit
    weathersit_dict = {1: 'Clear', 2: 'Cloudy/Mist', 3: 'Light Rain/Light Snow', 4: 'Heavy Rain/Thunderstorm'}
    #Temp season label column
    data['weathersit_labels'] = data['weathersit'].replace(weathersit_dict)
    #Groupby to obtain the total number of bikes rented per weathersit
    weekday_count = data.groupby("weathersit_labels")["cnt"].sum().sort_values(ascending = False).reset_index()
    #Plot the result
    colors = ["#e1b18e","#b0c9db", "#6794b5", "#28465c"]
    fig = px.bar(weekday_count, x='weathersit_labels', y='cnt', labels={'weathersit_labels': 'Weather', 'cnt': 'Count'},color = "weathersit_labels",color_discrete_sequence = colors)
    fig.update_layout(title='Bike Rentals by Weather', width=800, height=500)
    st.plotly_chart(fig, use_container_width = True)
    st.markdown("""- Clearly, the better the weather, the more bikes that are rented. From all the bikes rented, 71% have been rented when the weather was clear or with little clouds.
- On the days with really bad weather, bikes are almost never rented. Only 223 bikes in total.""")
    
if tab == "temp":
    st.markdown("Study of the distributtion of the column *temp*.")
    #Plot - Distributtion of temp
    fig = go.Figure()
    fig.add_trace(go.Box(x=data['temp'], orientation='h'))
    fig.update_layout(title='temp Distribution', xaxis_title='temp')
    # Show the plot
    st.plotly_chart(fig, use_container_width = True)
    st.markdown(""" - The median temp is 0.5, with a min of 0.02 and a max of 1.
 - There are not outliers on the temp variable.""")
    
    #Plot - Bike Rentals by Temperature
    st.markdown("We also going to try to plot the distributtion of bike rentals by *temp* and *season*.")
    fig = px.scatter(data, x='temp', y='cnt', color='season', hover_data=['season', 'temp', 'cnt'], color_continuous_scale=[[0, "sandybrown"], [0.5, "seagreen"], [0.75, "coral"], [1, "steelblue"]])
    fig.update_layout(title='Bike Rentals by Temperature', xaxis_title='Temperature', yaxis_title='Count', width = 800, height = 600)
    st.plotly_chart(fig, use_container_width = True)
    st.markdown("We can see how there is a trend of bikes rented with the temperature. As temperature increases, there is more probability that the number of bikes rented also increases. ")
    
if tab == "atemp":
    st.markdown("Study of the distributtion of the column *atemp*.")
    #Plot - Distributtion of atemp
    fig = go.Figure()
    fig.add_trace(go.Box(x=data['atemp'], orientation='h'))
    fig.update_layout(title='atemp Distribution', xaxis_title='temp')
    # Show the plot
    st.plotly_chart(fig, use_container_width = True)
    st.markdown("""- The median atemp is 0.4848, with a min of 0 and a max of 1.
 - There are not outliers on the temp variable.""")    
    
if tab == "hum":
    st.markdown("Study of the distributtion of the column *hum*.")
    #Plot - Distributtion of hum
    fig = go.Figure()
    fig.add_trace(go.Box(x=data['hum'], orientation='h'))
    fig.update_layout(title='hum Distribution', xaxis_title='temp')
    # Show the plot
    st.plotly_chart(fig, use_container_width = True)
    st.markdown(""" - The median hum is 0.63, with a min of 0 and a max of 1.
 - The are outliers in the hum variable with value 0. We will study this more in depth.""")
    
    st.markdown("Let's focus on the case when the *hum* value is 0.")
    data_hum_0
    st.markdown("""Apparently, during the 3rd of March of 2011 something appears to have happened with the humidity measure. We need to fix this data. We have studied if there are any correlations with the *hum* variable in order to help us impute the values but there are none. Taking this into consideration, we are going to impute it by using the mean values from the previous and the following day since, normally, weather does not change drastically from one day to another.""")
    data[data["dteday"] == "2011-03-10"]
    
if tab == "windspeed":
    st.markdown("Study of the distributtion of the column *windspeed*.")
    #Plot - Distributtion of windspeed
    fig = go.Figure()
    fig.add_trace(go.Box(x=data['windspeed'], orientation='h'))
    fig.update_layout(title='windspeed Distribution', xaxis_title='temp')
    # Show the plot
    st.plotly_chart(fig, use_container_width = True)
    st.markdown("""- The median windspeed is 0.194 with a min of 0 and a max of 0.8507.
- There are multiple outliers in the upper fence of the windspeed variable.""")
    
    st.markdown("Let's explore the outliers of the windspeed variable.")
    data[data["windspeed"] > 0.4627]
    st.write("Mean bikes rented for windspeed outliers: ", data[data["windspeed"] > 0.4627]["cnt"].mean())
    st.write("Mean bikes rented for windspeed non-outliers: ", data[data["windspeed"] <= 0.4627]["cnt"].mean())
    st.markdown("There are 342 outliers that, when looking into them, from the real-world perspective they do make sense. Moreover, as it can be seen above the mean of bikes rented is pretty much the same for the outliers and the non-outliers. For this reason, we are going to keep them.")
    
if tab == "registered & casual":
    #Time series for the number of bikes rented of registered vs casual
    fig = px.line(data, x='dteday', y=['registered', 'casual'], title='Bike Rentals Registered vs Casual Over Time', color_discrete_sequence = ["steelblue", "coral"], width = 1200, height = 500)
    st.plotly_chart(fig, use_container_width = True)
    st.markdown("We can conclude that the majority of bikes are rented by *registered* clients.")
    
st.markdown("---")
#Correlation of the original features    
st.subheader("Correlation of the original features")
st.markdown("We are going to study the correlations between the original features provided in the original data to have a basic idea. In any case, we will run this study again after the feature engineering process to also include the new columns created.")
# check correlation between features
corr = data.corr()
# plot correlation matrix with plotly
fig = go.Figure(data=go.Heatmap(
    z=corr.values,
    x=corr.columns,
    y=corr.columns,
    colorscale='Viridis',
    colorbar=dict(
        title="Correlation",
        titleside="right",
        tickmode="array",
        tickvals=[-1, -0.5, 0, 0.5, 1],
        ticktext=["-1", "-0.5", "0", "0.5", "1"]
    )
))
fig.update_layout(
    title="Correlation Matrix",
    xaxis_title="Features",
    yaxis_title="Features",
    width=800,
    height=800
)
st.plotly_chart(fig, use_container_width = True)
st.markdown("""- High correlation between all the variables for the user count. Probably we are going to drop the *casual* and *registered* to only have *cnt* which is going to be the target variable.
- High correlation between *mnth* and *season*. It makes sense since the months are what determined the seasons. We need to think to only use one. 
- Both temperatures are really correlated (*temp* and *atemp*). We need to think to only use of them.
- The correlation with the variable *instant* we do not care about them since, we are not going to use *instant* for the models. It is basically an index.""")

#Nan values
st.subheader("NaN Values")
st.markdown("Let's check if there are any NaN Values on the data.")
st.table(data.isna().sum())
st.markdown("There are no null values. Nothing else to be done here.")

#Feature engineering
st.header("Feature Engineering")
st.markdown("In this section you can select the new features you want to include on the model")
st.markdown("Please, select the features you want to include on the model")

col1, col2 = st.columns(2)

with col1:
    temp_range_flag = st.checkbox("temp_range", help = "This feature converts the temp variable into a 4 categories: cold, cool, warm or hot")
    atemp_range_flag = st.checkbox("atemp_range", help = "This feature converts the atemp variable into a 4 categories: cold, cool, warm or hot")
    hum_range_flag = st.checkbox("hum_range", help = "This feature converts the hum variable into a 5 categories: Very Dry, Dry, Normal, Humid, Very Humid")
    windspeed_range_flag = st.checkbox("windspeed_range", help = "This feature bins the windspeed variable: 0-0.25, 0.25-0.5, 0.5-0.75, 0.75-1")
    diff_temp_flag = st.checkbox("diff_temp", help = "Difference between the temperature and the feeling temperature (atemp - temp)")
    time_since_rain_flag = st.checkbox("time_since_rain", help = "Time since it last rainned")
    time_of_day_flag = st.checkbox("time_of_day", help = "Transform the hours of the day into categories: night, morning, afternoon and evening")
    rush_hour_flag = st.checkbox("rush_hour", help = "Flag if it is one of the most busy hours of the day or not")

with col2:
    temp_hum_ratio_flag = st.checkbox("temp_hum_ratio", help = "temp / hum")
    atemp_hum_ratio_flag = st.checkbox("atemp_hum_ratio", help = "atemp / hum")
    wind_temp_ratio_flag = st.checkbox("wind_temp_ratio", help = "windspeed / temp")
    wind_atemp_ratio_flag = st.checkbox("wind_atemp_ratio", help = "windspeed / atemp")
    wind_hum_ratio_flag = st.checkbox("wind_hum_ratio", help = "windspeed / hum")

#Apply of the selection to the data
if st.button("Apply"):
    #temp_hum_ratio
    if temp_hum_ratio_flag:
        #Nem colunm temp_hum_ratio
        data['temp_hum_ratio'] = (data['temp'] * 41) / (data['hum'] * 100)
    #atemp_hum_ratio
    if atemp_hum_ratio_flag:
        #Nem colunm atemp_hum_ratio
        data['atemp_hum_ratio'] = (data['atemp'] * 50) / (data['hum'] * 100)
    #wind_temp_ratio
    if wind_temp_ratio_flag:
        #Nem colunm wind_temp_ratio
        data['wind_temp_ratio'] = (data["windspeed"] * 67)/ (data['temp'] * 41)
    #wind_atemp_ratio
    if wind_atemp_ratio_flag:
        #Nem colunm wind_atemp_ratio
        data['wind_atemp_ratio'] = (data['windspeed'] * 67) / (data['atemp'] * 50)
    #wind_hum_ratio
    if wind_hum_ratio_flag:
        #Nem colunm wind_hum_ratio
        data['wind_hum_ratio'] = (data['windspeed'] * 67) / (data['hum'] * 100) 
    #diff_temp
    if diff_temp_flag:
        #New column 
        data['diff_temp'] = data['atemp'] - data['temp']
    #temp_range
    if temp_range_flag:
        #Temperature range categories
        temp_bins = [-float('inf'), 0.2, 0.5, 0.8, float('inf')]
        temp_labels = ['cold', 'cool', 'warm', 'hot']
        #New column temp_range
        data['temp_range'] = pd.cut(data['temp'], bins=temp_bins, labels=temp_labels)
        data = data.drop(columns = "temp")
    #atemp_range
    if atemp_range_flag:
        #Temperature range categories
        temp_bins = [-float('inf'), 0.2, 0.5, 0.8, float('inf')]
        temp_labels = ['cold', 'cool', 'warm', 'hot']
        #New column temp_range
        data['atemp_range'] = pd.cut(data['atemp'], bins=temp_bins, labels=temp_labels)
        data = data.drop(columns = "atemp")
    #hum_range
    if hum_range_flag:
        #New column hum_range
        data['hum_range'] = data['hum'].apply(lambda x: 'Very Dry' if x <= 0.2 else 'Dry' if x <= 0.4 else 'Normal' if x <= 0.6 else 'Humid' if x <= 0.8 else 'Very Humid')
        data = data.drop(columns = "hum")
    #windspeed_range
    if windspeed_range_flag:
        #New column windspeed_range
        data['windspeed_range'] = pd.cut(data['windspeed'], bins=[0.0, 0.25, 0.5, 0.75, 1.0], labels=['0-0.25', '0.25-0.5', '0.5-0.75', '0.75-1'])
        data["windspeed_range"] = data["windspeed_range"].fillna('0-0.25')
        data = data.drop(columns = "windspeed")
    #rush_hour
    if rush_hour_flag:
        #New rush_hour column
        data['rush_hour'] = data['hr'].apply(lambda x: 1 if (x >= 7 and x <= 9) or (x >= 16 and x <= 18) else 0)
    #time_of_day
    if time_of_day_flag:
        #Time of day categories
        time_bins = [-1, 6, 12, 18, 24]
        time_labels = ['night', 'morning', 'afternoon', 'evening']
        #New columns time_of_day
        data['time_of_day'] = pd.cut(data['hr'], bins=time_bins, labels=time_labels)
        data = data.drop(columns = "hr")
    #time_since_rain
    if time_since_rain_flag:
        #New column time_since_rain
        #Define variables
        data['time_since_rain'] = 0
        hours_since_rain = 0
        #Loop
        for i in range(len(data)):
            if (data.loc[i, 'weathersit'] == 3) | (data.loc[i, 'weathersit'] == 4) :
                hours_since_rain = 0
            else:
                hours_since_rain += 1
            data.loc[i, 'time_since_rain'] = hours_since_rain
    
    #Display the final 
    st.markdown("This is the final dataset with the features selected added")
    data.to_csv("feature_data.csv")
    data
    
#Data Preparation
st.header("Data Preparation")
st.markdown("This is the final section before applying the model.")
st.markdown("The final correlation matrix is going to be display and, in case you want to drop any features, please select them and click on Drop")
#Plot - Final Correlation
# check correlation between features
# Drop the first column by position
feature_data = pd.read_csv("feature_data.csv").iloc[:,1:]
corr = feature_data.corr()
# plot correlation matrix with plotly
fig = go.Figure(data=go.Heatmap(
    z=corr.values,
    x=corr.columns,
    y=corr.columns,
    colorscale='Viridis',
    colorbar=dict(
    title="Correlation",
        titleside="right",
        tickmode="array",
        tickvals=[-1, -0.5, 0, 0.5, 1],
        ticktext=["-1", "-0.5", "0", "0.5", "1"]
    )
))
fig.update_layout(
    title="Correlation Matrix",
    xaxis_title="Features",
    yaxis_title="Features",
    width=800,
    height=800
)
st.plotly_chart(fig, use_container_width = True)

#Select box
feature_data = feature_data.drop(columns = ["instant", "dteday", "casual", "registered"])
drop_features = st.multiselect("Select the features to drop", feature_data.columns.sort_values(), help = "The columns 'instant', 'dteday', 'casual' and 'registered' are going to be drop by default since they can not be use in the model")
#Drop button
if st.button("Drop"):
    prep_data = feature_data.drop(columns = drop_features)
    prep_data.to_csv("prep_data.csv")
    st.markdown("This is the dataset after the selected features have been dropped:")
    prep_data
    
#Prediction Model
st.header("Prediction Model")
st.markdown("In order to find the best model for this data, we are going to use *pycaret*. This is a library that allows us to do autoML in order to find the best model for our data and predict with it.")
st.markdown("By clicking the button below, we will extract 20% of the data to simulate new unseen data and the rest will be used in order to run the pycaret pipeline to find the best model, train it and validate it.")
st.markdown("Please, click the button to create the model.")
if st.button("Let's Rock"):
    prep_data = pd.read_csv("prep_data.csv").iloc[:,1:]
    if "instant" in prep_data.columns:
        prep_data = prep_data.drop(columns = ["instant", "dteday", "casual", "registered"])
    #Unseen datasets
    unseen_data_pycaret = prep_data.sample(n=3476)
    unseen_data_pycaret = unseen_data_pycaret.drop(columns = 'cnt')
    #Remove the extracted rows from the original dataframe to avoid data leakage
    prep_data_pycaret = prep_data.drop(unseen_data_pycaret.index)
    #Setup
    setup = setup(data=prep_data_pycaret, target='cnt', silent=True)
    #Compare models
    best = compare_models(fold=5, sort = 'R2')
    st.write("Best model: ", best)
    #Create the model
    ctb = create_model(best, fold=5)
    dt_results = pull()
    #Tune the model
    tuned_ctb = tune_model(ctb, optimize='MSE')
    st.write("Best param: ", tuned_ctb.get_params())
    #Score of the model
    st.markdown("Model scores during validation:")
    st.table(dt_results)
    #Plot - Residuals
    st.markdown("Plot of the model residuals during validation:")
    plot_model(best, plot = 'residuals', save = True)
    residuals_image = Image.open("Residuals.png")
    st.image(residuals_image, use_column_width=True)
    #Plot - Error
    st.markdown("Plot of the model errors during validation:")
    plot_model(best, plot = 'error', save = True)
    error_image = Image.open("Prediction Error.png")
    st.image(error_image, use_column_width=True)
    #Predictions using the model
    # training model with all the data
    model = finalize_model(tuned_ctb)
    #Predict
    final_pred = predict_model(model, unseen_data_pycaret)
    st.markdown("Predicted results from the unseendata: ")
    final_pred = final_pred.rename(columns = {"Label": "Predicted_cnt"})
    final_pred
    #if st.button("Save Model"):
        #Saving the model
        #save_model(model, 'ctb_python_assignment')
    
    
