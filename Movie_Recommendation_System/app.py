import pickle
import pandas as pd
import streamlit as st

#Load the data

data=pickle.load(open('movie_dict.pkl',mode='rb'))

data = pd.DataFrame(data)
#print(data)
print(data['title'].values)


# Load Similarity score
similarity = pickle.load(open('similarity.pkl',mode='rb'))
#print(similarity)


# Final Function
def recommend(movie):

    recommend_movies =[]

    movie_index = data[data['title'] == movie].index[0]
    distance = similarity[movie_index]
    movie_list = sorted(list(enumerate(distance)),reverse=True, key=lambda x: x[1])[1:6]

    for i in movie_list:
        recommend_movies.append(data.iloc[i[0]].title)

    return recommend_movies


# Streamlit Web App

st.title('Movie Recommendation System')

selected_movie = st.selectbox("Select a movie to get a recommendation",data['title'].values)
 
btn=st.button('Recommend')

if btn:

    list_of_movie=recommend(selected_movie)

    for movie in list_of_movie:
        st.write(movie)
 