# nhl_points_predictor
AI model made using tensorflow/keras to predict how many points an NHL player will record in a given season. Data was obtained from www.hockey-reference.com using a webscraper included here. All the data is saved in a csv file that the model was trained on, so you can predict points from any player since the 2009-2010 season. The model is currently in beta and needs tweaking as it vastly under-predicts the points for players.
The model works by prompting the user for a player, looking at that players stats in the previous year, and using these stats to predict how many points the player will score next year.
The year you use for prediction can be changed for testing purposes, but remains constant in the source code to make predictions for next season (2026-2027)
