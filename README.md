# Movie-Recommendation-System

Overview

The Movie Recommendation System is a Python-based application that provides personalized movie suggestions based on user preferences and historical ratings. With an intuitive Tkinter GUI, users can easily input their ratings and get movie recommendations without dealing with the code or datasets directly.

Features

->Personalized movie recommendations based on user ratings.

->Tkinter GUI for easy and interactive use.

->Handles datasets of movies, users, and ratings.

->Can be extended to include content-based filtering, hybrid recommendations, or web deployment.

Dataset:

The system uses a dataset containing:

->userId: Unique ID of the user

->movieId: Unique ID of the movie

->title: Movie title

->rating: User rating for the movie

Technologies Used:

->Python

->Tkinter (GUI)

->Pandas, NumPy (data handling)

->Scikit-learn (recommendation algorithms)

How It Works

1.Load the dataset of users, movies, and ratings.

2.Preprocess the data to create a user-movie matrix.

3.Apply collaborative filtering to predict user preferences.

4.Users input ratings through the Tkinter GUI.

5.Generate a list of top recommended movies for the user.
