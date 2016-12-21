
# CS5010 - Analysis of MetaCritic Movie Data
# Jordan Baker (jmb4ax), Matt DaVolio (md3es), Brady Fowler (dbf5sd), Andrew Pomykalski (ajp5sb)
# 8/1/2016

import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
import pylab as pl
import matplotlib as plt
from matplotlib import figure


# Establish column names for the data frame
colname = ['critic_rating_val', 'critic_rating_num', 'user_rating', 'user_rating_num', 'mpaa_rating', 'genres', 'movie_date', 'movie_name', 'description', 'current_url', 'c_desc']

# Read in the movie data
rawdata = pd.read_table('C:/Users/Jordan/Documents/School/University of Virginia/Summer 2016/CS5010 - Programming and Systems for Data Analysis/Project/movie_details.txt', sep='|', names=colname)

# Read in dates into 'YYYY-MM-DD' format with no timestamps
rawdata['movie_date'] = pd.to_datetime(rawdata['movie_date'], errors='coerce').dt.date

# Strip out punctuation from description
rawdata['c_desc'] = rawdata['description'].str.replace('[^\w\s]','')

# Output the rawdata to a CSV file
rawdata.to_csv("rawdata.csv")

# Remove user ratings that are "Unknown" or "tbd"
# Remove mpaa_ratings that are "Unknown", "Not Rated", or TV-related
rawdata = rawdata[rawdata["user_rating"] != "Unknown"]
rawdata = rawdata[rawdata["user_rating"] != "tbd"]
rawdata = rawdata[rawdata["mpaa_rating"] != "Unknown"]
rawdata = rawdata[rawdata["mpaa_rating"] != "Not Rated"]
rawdata = rawdata[rawdata["mpaa_rating"].str.contains("TV") == False]

# Change the ratings columns to int and float types
rawdata[["critic_rating_val"]] = rawdata[["critic_rating_val"]].convert_objects(convert_numeric=True)
rawdata[["critic_rating_num"]] = rawdata[["critic_rating_num"]].convert_objects(convert_numeric=True)
rawdata[["user_rating"]] = rawdata[["user_rating"]].convert_objects(convert_numeric=True)
rawdata[["user_rating_num"]] = rawdata[["user_rating_num"]].convert_objects(convert_numeric=True)
rawdata['movie_date'] = pd.to_datetime(rawdata['movie_date'], errors='coerce')



# Create subset files that contain the genre and ratings obs we want
genres_sub = rawdata[rawdata["genres"].str.contains("Action|Comedy|Drama|Romance|Family|Horror")]
ratings_dict = {"G":"G", "PG":"G", "GP":"G", "PG-13":"PG13", "PG--13":"PG13", "R":"R", "NC-17":"R", "X":"X", "Unrated":"Unrated"}
ratings_sub = rawdata[rawdata["mpaa_rating"].str.contains("G|PG|GP|PG-13|PG--13|R|NC-17|X|Unrated")]
ratings_sub["rating_group"] = ratings_sub.apply(lambda x: ratings_dict[x["mpaa_rating"]], axis=1)
genres_sub['movie_date'] = pd.to_datetime(genres_sub['movie_date'], errors='coerce')


# Create a list of means by genre (for critics and users)
genre_list = ["Action", "Comedy", "Drama", "Romance", "Family", "Horror"]
genre_means = []
for genre in genre_list:
    temp_sub = genres_sub[genres_sub["genres"].str.contains(genre) == True]
    critic_mean = np.mean(temp_sub["critic_rating_val"])/10
    user_mean = np.mean(temp_sub["user_rating"])
    genre_means.append([genre,critic_mean, user_mean])

# Create a list of means by rating (for critics and users)
rating_list = ["G", "PG13", "R", "X", "Unrated"]
rating_means = []
for rating in rating_list:
    temp_sub = ratings_sub[ratings_sub["rating_group"] == rating]
    critic_mean = np.mean(temp_sub["critic_rating_val"])/10
    user_mean = np.mean(temp_sub["user_rating"])
    rating_means.append([rating,critic_mean,user_mean])
    
# Create a list of means by year (for critics and users)
year_list = list(range(1996,2016))
year_means = []
for year in year_list:
    temp_sub = rawdata.loc[rawdata["movie_date"].dt.year == int(year)]
    critic_mean = np.mean(temp_sub["critic_rating_val"])/10
    user_mean = np.mean(temp_sub["user_rating"])
    year_means.append([year,critic_mean, user_mean])

# Create a list of means by year and genre (for critics and users)
genre_year_means = []
for genre in genre_list:
    for year in year_list:
        temp_sub = genres_sub[(genres_sub["genres"].str.contains(genre) == True) & (genres_sub["movie_date"].dt.year == int(year))]
        critic_mean = np.mean(temp_sub["critic_rating_val"])/10
        user_mean = np.mean(temp_sub["user_rating"])
        genre_year_means.append([year, genre ,critic_mean, user_mean])

top5_per_year = []
for year in year_list:
    temp_list = []
    for i in list(range(0,5)): 
        temp_sub = rawdata.loc[rawdata["movie_date"].dt.year == int(year)]
        temp_sub = temp_sub.sort("critic_rating_val", ascending = False)
        movie = temp_sub['movie_name'].iloc[i]
        temp_list.append([movie])
    top5_per_year.append(temp_list)

# Look at the minimum user reviews that we have
list(set(list(rawdata["user_rating_num"])))[:10]

# Convert the mean lists to data frames
genre_means_df = pd.DataFrame(genre_means)
rating_means_df = pd.DataFrame(rating_means)
year_means_df = pd.DataFrame(year_means)
genre_year_means_df = pd.DataFrame(genre_year_means)
top_5_df = pd.DataFrame(top5_per_year)
top_5_df = top_5_df.set_index([year_list])


# Change the names of columns in the recently created data frames
genre_means_df = genre_means_df.rename(columns= {0:'genre', 1:'critic_score', 2:'user_score'})
rating_means_df = rating_means_df.rename(columns= {0:'rating', 1:'critic_score', 2:'user_score'})
year_means_df = year_means_df.rename(columns= {0:'year', 1:'critic_score', 2:'user_score'})
genre_year_means_df = genre_year_means_df.rename(columns= {0:'year', 1:'genre', 2:'critic_score', 3:'user_score'})
top_5_df = top_5_df.rename(columns= {0:'#1', 1:'#2', 2:'#3', 3:'#4', 4:'#5',})

# Set the index of year_means_df to year - RUN INDIVIDUALLY
year_means_df = year_means_df.set_index('year')

#Summaries
rating_summary = pd.DataFrame(ratings_sub['rating_group'].value_counts())



# Plotting Critic Score and User Score by Genre
genre_plot = genre_means_df.plot.bar(x=genre_means_df['genre'], ylim = (0,10), color = 'br', title="Critic and User Ratings by Genre")
genre_plot.set_xlabel("Genre")
genre_plot.set_ylabel("Average Rating")

# Plotting the Critic Score and User Score by MPAA Rating
rating_plot = rating_means_df.plot.bar(x=rating_means_df['rating'], ylim = (0,10), color = 'br', title="Critic and User Ratings by MPAA Rating")
rating_plot.set_xlabel("MPAA Rating")
rating_plot.set_ylabel("Average Rating")


# Plotting the Critic SCore and User Score Over Time
year_plot = year_means_df.plot.line(color = 'br', ylim = (4.5,8), title="Critic and User Scores Over 20 Years")
year_plot.set_xlabel("Year")
year_plot.set_ylabel("Average Rating")

# Density Plot for Critic and User Scores
rawdata['critic_rating_val'] = rawdata['critic_rating_val']/10
critic_density_plot = rawdata['critic_rating_val'].plot.kde(xlim=(0,10), title="Density Plot of Critic Scores")
critic_density_plot.set_xlabel("Rating")
user_density_plot = rawdata['user_rating'].plot.kde(xlim=(0,10), title="Density Plot of User Scores", color='r')
user_density_plot.set_xlabel("Rating")









