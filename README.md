# STA9890-Player-Value-Prediction-Project
This is the repository for containing materials for STA9890 FIFA Football Player Value Prediction Project.


## Data Source and Descriptions
This dataset is directly from Kaggle (https://www.kaggle.com/karangadiya/fifa19). I uploaded the data in the repository, in case you can't download it, you can download data directly from the Kaggle URL link above. It contains all the statistics and playing attributes of all the players in the FIFA 2019. The data is scraped from the website https://sofifa.com by extracting the player personal data, followed by player IDs and their playing and style statistics.

The response variable is the value for each player. The predictors are the demographic ones (age, nationality, height, weight, body type, real face), playing-related ones (position, preferred foot, weak foot, skill move date), value-related ones (wage, international reputation, attacking / defensive work rate), attacking-related ones (crossing, finishing, heading accuracy, short passing, volleys scores), skill-related ones (dribbling, curve, free kick accuracy, long passing, ball control scores), movement-related ones (acceleration, sprint speed, agility, reactions, balance scores), power-related ones (shot power, jumping, stamina, strength, long shots scores), mentality-related scores (aggression, interceptions, positioning, vision, penalties, composure scores), defending-related ones (defensive awareness, standing tackle, sliding tackle scores), goalkeeping-related ones (goalkeeping diving, handing, kicking, positioning, reflexes scores).

## Number of Observations and Predictors
After pre-processing the dataset, only release clause value predictor has missing values, so I impute the predictor with its mean value. There are n=18,159 observations (17,907 after pre-processing) in the dataset. Since the dataset has categorical predictors, before converting them to dummies, the dataset has p=48 predictors, but after that, it has p=55 predictors.

