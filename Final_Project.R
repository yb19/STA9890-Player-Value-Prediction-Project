library(glmnet)
library(stringr)
library(rmarkdown)
library(tidyverse)
library(fastDummies)
library(randomForest)
library(gridExtra)

# Set the working directory which contains the dataset
setwd("~/Dropbox/Education/Baruch College/Statistical Learning for Data Mining (STA 9890)/Project/")


### Read in data
data <- read.csv("data.csv")
head(data)
glimpse(data)


### Preprocess some predictors
data <- data %>% 
  select(-c(1:3, 5, 7, 10:11, 14, 23:26, 29:54)) %>% # Delete unnecessary columns
  mutate(
    Value = gsub("€", "", Value),
    Value = ifelse(grepl("M", Value, fixed = TRUE), as.numeric(gsub("M", "", Value)) * 1000, 
                   ifelse(grepl("K", Value, fixed = TRUE), as.numeric(gsub("K", "", Value)), as.numeric(Value))),
    Wage = gsub("€", "", Wage),
    Wage = as.numeric(gsub("K", "", Wage)), 
    Height = as.numeric(gsub("\'", ".", Height)),
    Weight = as.numeric(gsub("lbs", "", Weight)),
    Release.Clause = gsub("€", "", Release.Clause),
    Release.Clause = ifelse(grepl("M", Release.Clause, fixed = TRUE), as.numeric(gsub("M", "", Release.Clause)) * 1000, 
                            ifelse(grepl("K", Release.Clause, fixed = TRUE), as.numeric(gsub("K", "", Release.Clause)), 
                                   as.numeric(Release.Clause)))
        )


### Drop observations with too many missing values (48 obs dropped)
data$missing1 <- apply(data == '', 1, sum)
data$missing1[is.na(data$missing1)] <- 0
data$missing2 <- apply(is.na(data), 1, sum)
data$missing1[is.na(data$missing1)] <- 0
data$missing <- data$missing1 + data$missing2

data <- data %>% filter(missing < 40) %>% select(c(1:(ncol(data)-3))) 


### Transform categorical predictors
# Convert large countries (defined as the largest 20 number of players) to 1, 0 otherwise
head(as.data.frame(table(data$Nationality)), 10)
temp <- as.data.frame(table(data$Nationality))
temp <- head(temp[order(-temp$Freq),], 20)
big.country <- as.vector(temp$Var1)

data$Nationality <- ifelse(data$Nationality %in% big.country, 1, 0)
data$Nationality <- factor(data$Nationality, levels = c(1, 0))

# Convert Real.Face == 'Yes' to 1, 0 otherwise
as.data.frame(table(data$Real.Face))
data$Real.Face <- ifelse(data$Real.Face == 'Yes', 1, 0)
data$Real.Face <- factor(data$Real.Face, levels = c(1, 0))

# Convert Preferred.Foot == 'Right' to 1, 0 otherwise
as.data.frame(table(data$Preferred.Foot))
data$Preferred.Foot <- ifelse(data$Preferred.Foot == 'Right', 1, 0)
data$Preferred.Foot <- factor(data$Preferred.Foot, levels = c(1, 0))

# Split work rate into two predictors: attacking rate and defensing rate
data$attack.work  <- str_split_fixed(data$Work.Rate, "/ ", 2)[,1]
data$defense.work <- str_split_fixed(data$Work.Rate, "/ ", 2)[,2]

data$attack.work  <- factor(data$attack.work, levels = c('High', 'Medium', 'Low'))
data$defense.work <- factor(data$defense.work, levels = c('High', 'Medium', 'Low'))

data <- data[ , !(names(data) %in% c('Work.Rate'))]

# Assign categories with very few frequencies to categories with largest frequency
as.data.frame(table(data$Body.Type))
categories <- c('Akinfenwa', 'C. Ronaldo', 'Courtois', 'Messi', 'Neymar', 'PLAYER_BODY_TYPE_25', 'Shaqiri')
data$Body.Type[data$Body.Type %in% categories] <- 'Normal'

data$Body.Type <- factor(data$Body.Type, levels = c('Normal', 'Lean', 'Stocky'))
as.data.frame(table(data$Body.Type))

# Merge categories with similar meaning
as.data.frame(table(data$Position))
data$Position <- ifelse(data$Position == 'GK', 'GK',
                        ifelse(data$Position %in% c('RB', 'LB', 'CB', 'LCB', 'RCB', 'RWB', 'LWB'), 'DF',
                               ifelse(data$Position %in% c('LDM', 'CDM', 'RDM'), 'DM',
                                      ifelse(data$Position %in% c('LM', 'LCM', 'CM', 'RCM', 'RM'), 'MF',
                                             ifelse(data$Position %in% c('LAM', 'CAM', 'RAM', 'LW', 'RW'), 'AM',
                                                    'ST')))))

data$Position <- factor(data$Position, levels = c('GK', 'DF', 'DM', 'MF', 'AM', 'ST'))
as.data.frame(table(data$Position))


### Deal with missing values
# Check missing values
apply(is.na(data), 2, sum)

# Release.Clause has missing values, so we use mean value to impute missing values.
data <- data %>% mutate(Release.Clause = replace_na(Release.Clause, mean(Release.Clause, na.rm = TRUE)))
sum(is.na(data$Release.Clause))


### Standardize numeric predictors
numerics <- names(data)[c(1, 3:4, 6, 8:10, 14:50)]
data <- data %>% mutate_at(numerics, ~(scale(.) %>% as.vector))
apply(data[numerics], 2, sd)


### Convert categorical predictors to dummy predictors
categories <- c('Body.Type', 'Position', 'attack.work', 'defense.work')
data <- fastDummies::dummy_cols(data, select_columns = categories, remove_first_dummy = TRUE)
data <- data[ , !(names(data) %in% categories)]

col_names <- names(data)[c((ncol(data)-10):ncol(data))]
data[col_names] <- lapply(data[col_names], factor)


### Modeling of the Relationships between Predictors and Response
# Construct predictors and response
data <- data[data$Value != 0, ] # for converting data$Value to log
y    <- log(data[, 5]) # distribution of data$Value is highly skewed
X    <- data.matrix(data[, -c(3:5, 48)]) # delete response and its co-dependent variables

# Set the basic parameters
set.seed(1)

n <- nrow(data) # number of total observations
p <- ncol(X)    # number of predictors
n.train <- floor(0.8 * n)
n.test  <- n - n.train

M <- 100
# R squared
Rsq.test.la  <- rep(0, M)  # la = lasso
Rsq.train.la <- rep(0, M)
Rsq.test.en  <- rep(0, M)  # en = elastic net
Rsq.train.en <- rep(0, M)
Rsq.test.ri  <- rep(0, M)  # ri = ridge
Rsq.train.ri <- rep(0, M)
Rsq.test.rf  <- rep(0, M)  # rf= randomForest
Rsq.train.rf <- rep(0, M)

for (m in c(1:M)) {
  
  shuffled_indexes <-     sample(n)
  train            <-     shuffled_indexes[1:n.train]
  test             <-     shuffled_indexes[(1+n.train):n]
  X.train          <-     X[train, ]
  y.train          <-     y[train]
  X.test           <-     X[test, ]
  y.test           <-     y[test]
  
  # Fit lasso and calculate and record the train and test R squares, and estimated coefficients 
  cv.fit           <-     cv.glmnet(X.train, y.train, alpha = 1, nfolds = 10)
  fit              <-     glmnet(X.train, y.train, alpha = 1, lambda = cv.fit$lambda.min)
  y.train.hat      <-     predict(fit, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  y.test.hat       <-     predict(fit, newx = X.test, type = "response")  # y.test.hat=X.test %*% fit$beta  + fit$a0
  Rsq.test.la[m]   <-     1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  Rsq.train.la[m]  <-     1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)
  
  # Fit elastic-net and calculate and record the train and test R squares, and estimated coefficients  
  cv.fit           <-     cv.glmnet(X.train, y.train, alpha = 0.5, nfolds = 10)
  fit              <-     glmnet(X.train, y.train, alpha = 0.5, lambda = cv.fit$lambda.min)
  y.train.hat      <-     predict(fit, newx = X.train, type = "response")
  y.test.hat       <-     predict(fit, newx = X.test, type = "response") 
  Rsq.test.en[m]   <-     1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  Rsq.train.en[m]  <-     1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)
  
  # Fit ridge and calculate and record the train and test R squares, and estimated coefficients 
  cv.fit           <-     cv.glmnet(X.train, y.train, alpha = 0, nfolds = 10)
  fit              <-     glmnet(X.train, y.train, alpha = 0, lambda = cv.fit$lambda.min)
  y.train.hat      <-     predict(fit, newx = X.train, type = "response") 
  y.test.hat       <-     predict(fit, newx = X.test, type = "response") 
  Rsq.test.ri[m]   <-     1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  Rsq.train.ri[m]  <-     1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)
  
  # Fit RF and calculate and record the train and test R squares, and estimated coefficients  
  rf               <-     randomForest(X.train, y.train, mtry = sqrt(p), importance = TRUE, ntree = 100, nodesize = 100)
  y.test.hat       <-     predict(rf, X.test)
  y.train.hat      <-     predict(rf, X.train)
  Rsq.test.rf[m]   <-     1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  Rsq.train.rf[m]  <-     1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)
  
  cat(sprintf("m=%3.f| Rsq.test.rf=%.2f,  Rsq.test.en=%.2f| Rsq.train.rf=%.2f,  Rsq.train.en=%.2f| \n", m,  Rsq.test.rf[m], Rsq.test.en[m],  Rsq.train.rf[m], Rsq.train.en[m]))
  
}


### Plot results
## Show the side-by-side boxplots of R2_test, R2_train
model <- c(rep('Lasso', 2*M), rep('ElasticNet', 2*M), rep('Ridge', 2*M), rep('RandomForest', 2*M))
type  <- c(rep('train', M), rep('test', M), rep('train', M), rep('test', M), rep('train', M), rep('test', M), rep('train', M), rep('test', M))
rsq   <- c(Rsq.train.la, Rsq.test.la, Rsq.train.en, Rsq.test.en, Rsq.train.ri, Rsq.test.ri, Rsq.train.rf, Rsq.test.rf)

boxdata <- data.frame(model, type, rsq)

model_order <- c('Lasso', 'ElasticNet', 'Ridge', 'RandomForest') 

ggplot(boxdata, aes(x = factor(model, level = model_order), y = rsq, fill = type)) +
  geom_boxplot(alpha = 0.7) +
  scale_y_continuous(name = "R Square") +
  scale_x_discrete(name = "Model") +
  ggtitle("Boxplot of R2_train and R2_test for Each Model") +
  theme_bw() +
  theme(plot.title = element_text(size = 12, family = "Tahoma", face = "bold", hjust = 0.5),
        text = element_text(size = 12, family = "Tahoma"),
        axis.title = element_text(face="bold"),
        axis.text.x = element_text(size = 10),
        legend.position = "bottom") +
  scale_fill_brewer(palette = "Accent") +
  labs(fill = "")

## For one on the 100 samples, create 10-fold CV curves for lasso, elastic-net α = 0.5, ridge.
shuffled_indexes <-     sample(n)
train            <-     shuffled_indexes[1:n.train]
test             <-     shuffled_indexes[(1+n.train):n]
X.train          <-     X[train, ]
y.train          <-     y[train]

cv.lasso <- cv.glmnet(X.train, y.train, alpha = 1, nfolds = 10)
cv.elast <- cv.glmnet(X.train, y.train, alpha = 0.5, nfolds = 10)
cv.ridge <- cv.glmnet(X.train, y.train, alpha = 0, nfolds = 10)

par(mfrow = c(1, 3), cex=0.7, mai=c(0.6,0.4,0.6,0.4))
plot(cv.lasso)
title('Lasso', line = 2.5)
plot(cv.elast)
title('ElasticNet', line = 2.5)
plot(cv.ridge)
title('Ridge', line = 2.5)


## For one on the 100 samples, show the side-by-side boxplots of train and test residuals. 
## Comment on the distribution and size of the residuals.
shuffled_indexes <-     sample(n)
train            <-     shuffled_indexes[1:n.train]
test             <-     shuffled_indexes[(1+n.train):n]
X.train          <-     X[train, ]
y.train          <-     y[train]
X.test           <-     X[test, ]
y.test           <-     y[test]

# Lasso
cv.fit           <-     cv.glmnet(X.train, y.train, alpha = 1, nfolds = 10)
fit              <-     glmnet(X.train, y.train, alpha = 1, lambda = cv.fit$lambda.min)
y.train.hat      <-     predict(fit, newx = X.train, type = "response") 
y.test.hat       <-     predict(fit, newx = X.test, type = "response")  
Res.test.la      <-     y.test - y.test.hat
Res.train.la     <-     y.train - y.train.hat

# Elastic net
cv.fit           <-     cv.glmnet(X.train, y.train, alpha = 0.5, nfolds = 10)
fit              <-     glmnet(X.train, y.train, alpha = 0.5, lambda = cv.fit$lambda.min)
y.train.hat      <-     predict(fit, newx = X.train, type = "response") 
y.test.hat       <-     predict(fit, newx = X.test, type = "response")  
Res.test.en      <-     y.test - y.test.hat
Res.train.en     <-     y.train - y.train.hat

# Ridge
cv.fit           <-     cv.glmnet(X.train, y.train, alpha = 0, nfolds = 10)
fit              <-     glmnet(X.train, y.train, alpha = 0, lambda = cv.fit$lambda.min)
y.train.hat      <-     predict(fit, newx = X.train, type = "response") 
y.test.hat       <-     predict(fit, newx = X.test, type = "response")  
Res.test.ri      <-     y.test - y.test.hat
Res.train.ri     <-     y.train - y.train.hat

# RF
rf               <-     randomForest(X.train, y.train, mtry = sqrt(p), importance = TRUE, ntree = 100, nodesize = 100)
y.test.hat       <-     predict(rf, X.test)
y.train.hat      <-     predict(rf, X.train)
Res.test.rf      <-     y.test - y.test.hat
Res.train.rf     <-     y.train - y.train.hat

model <- c(rep('Lasso', n), rep('ElasticNet', n), rep('Ridge', n), rep('RandomForest', n))
type  <- c(rep('train', n.train), rep('test', n.test), rep('train', n.train), rep('test', n.test), rep('train', n.train), rep('test', n.test), rep('train', n.train), rep('test', n.test))
res   <- c(Res.train.la, Res.test.la, Res.train.en, Res.test.en, Res.train.ri, Res.test.ri, Res.train.rf, Res.test.rf)

boxdata <- data.frame(model, type, res)

model_order <- c('Lasso', 'ElasticNet', 'Ridge', 'RandomForest') 

ggplot(boxdata, aes(x = factor(model, level = model_order), y = res, fill = type)) +
  geom_boxplot(alpha = 0.7) +
  scale_y_continuous(name = "Residual") +
  scale_x_discrete(name = "Model") +
  ggtitle("Boxplot of Residual_train and Residual_test for Each Model") +
  theme_bw() +
  theme(plot.title = element_text(size = 12, family = "Tahoma", face = "bold", hjust = 0.5),
        text = element_text(size = 12, family = "Tahoma"),
        axis.title = element_text(face="bold"),
        axis.text.x = element_text(size = 10),
        legend.position = "bottom") +
  scale_fill_brewer(palette = "Accent") +
  labs(fill = "")


## Present bar-plots (with bootstrapped error bars) of the estimated coefficients, and the 
## importance of the parameters.
bootstrapSamples <-     100
beta.la.bs       <-     matrix(0, nrow = p, ncol = bootstrapSamples)   
beta.en.bs       <-     matrix(0, nrow = p, ncol = bootstrapSamples)
beta.ri.bs       <-     matrix(0, nrow = p, ncol = bootstrapSamples)
beta.rf.bs       <-     matrix(0, nrow = p, ncol = bootstrapSamples) 

for (m in 1:bootstrapSamples){
  bs_indexes       <-     sample(n, replace = T)
  X.bs             <-     X[bs_indexes, ]
  y.bs             <-     y[bs_indexes]
  
  # fit bs lasso
  cv.fit           <-     cv.glmnet(X.bs, y.bs, alpha = 1, nfolds = 10)
  fit              <-     glmnet(X.bs, y.bs, alpha = 1, lambda = cv.fit$lambda.min)  
  beta.la.bs[,m]   <-     as.vector(fit$beta)

  # fit bs elastic-net
  cv.fit           <-     cv.glmnet(X.bs, y.bs, alpha = 0.5, nfolds = 10)
  fit              <-     glmnet(X.bs, y.bs, alpha = 0.5, lambda = cv.fit$lambda.min)  
  beta.en.bs[,m]   <-     as.vector(fit$beta)
  
  # fit bs ridge
  cv.fit           <-     cv.glmnet(X.bs, y.bs, alpha = 0, nfolds = 10)
  fit              <-     glmnet(X.bs, y.bs, alpha = 0, lambda = cv.fit$lambda.min)  
  beta.ri.bs[,m]   <-     as.vector(fit$beta)
  
  # fit bs rf
  rf               <-     randomForest(X.bs, y.bs, mtry = sqrt(p), importance = TRUE, ntree = 100, nodesize = 100)
  beta.rf.bs[,m]   <-     as.vector(rf$importance[,1])

  cat(sprintf("Bootstrap Sample %3.f \n", m))
}

# calculate bootstrapped standard errors / alternatively you could use qunatiles to find upper and lower bounds
sd.bs.la    <-   apply(beta.la.bs, 1, "sd")
sd.bs.en    <-   apply(beta.en.bs, 1, "sd")
sd.bs.ri    <-   apply(beta.ri.bs, 1, "sd")
sd.bs.rf    <-   apply(beta.rf.bs, 1, "sd")

# fit lasso to the whole data
cv.lasso    <-     cv.glmnet(X, y, alpha = 1, nfolds = 10)
lasso       <-     glmnet(X, y, alpha = 1, lambda = cv.lasso$lambda.min)

# fit elastic-net to the whole data
cv.elast    <-     cv.glmnet(X, y, alpha = 0.5, nfolds = 10)
elast       <-     glmnet(X, y, alpha = 0.5, lambda = cv.elast$lambda.min)

# fit ridge to the whole data
cv.ridge    <-     cv.glmnet(X, y, alpha = 0, nfolds = 10)
ridge       <-     glmnet(X, y, alpha = 0, lambda = cv.ridge$lambda.min)

# fit rf to the whole data
rf          <-     randomForest(X, y, mtry = sqrt(p), importance = TRUE, ntree = 100, nodesize = 100)


betaS.rf               <-     data.frame(names(X[1,]), as.vector(rf$importance[,1]), 2*sd.bs.rf)
colnames(betaS.rf)     <-     c( "feature", "value", "err")

betaS.en               <-     data.frame(names(X[1,]), as.vector(elast$beta), 2*sd.bs.en)
colnames(betaS.en)     <-     c( "feature", "value", "err")

betaS.la               <-     data.frame(names(X[1,]), as.vector(lasso$beta), 2*sd.bs.la)
colnames(betaS.la)     <-     c( "feature", "value", "err")

betaS.ri               <-     data.frame(names(X[1,]), as.vector(ridge$beta), 2*sd.bs.ri)
colnames(betaS.ri)     <-     c( "feature", "value", "err")

# We need to change the order of factor levels by specifying the order explicitly.
betaS.rf$feature  <-  factor(betaS.rf$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])
betaS.en$feature  <-  factor(betaS.en$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])
betaS.la$feature  <-  factor(betaS.la$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])
betaS.ri$feature  <-  factor(betaS.ri$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])

# Compare random forest and elastic net
rfPlot <-  ggplot(betaS.rf, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) +
  theme(plot.title = element_text(size = 10, face = "bold", hjust = 0.5),
        axis.text.x = element_text(angle = 90, hjust = 1), 
        axis.title.x = element_blank()) +
  ggtitle("Feature Importance of Random Forest")

enPlot <-  ggplot(betaS.en, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) +
  theme(plot.title = element_text(size = 10, face = "bold", hjust = 0.5),
        axis.text.x = element_text(angle = 90, hjust = 1), 
        axis.title.x = element_blank()) + 
  ggtitle("Feature Importance of Elastic Net")
  
grid.arrange(rfPlot, enPlot, nrow = 2)

# Compare elastic net and lasso/ridge
laPlot <-  ggplot(betaS.la, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) +
  theme(plot.title = element_text(size = 10, face = "bold", hjust = 0.5),
        axis.text.x = element_text(angle = 90, hjust = 1), 
        axis.title.x = element_blank()) +
  ggtitle("Feature Importance of Lasso")

riPlot <-  ggplot(betaS.ri, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) +
  theme(plot.title = element_text(size = 10, face = "bold", hjust = 0.5),
        axis.text.x = element_text(angle = 90, hjust = 1), 
        axis.title.x = element_blank()) + 
  ggtitle("Feature Importance of Ridge")

grid.arrange(enPlot, laPlot, nrow = 2)
grid.arrange(enPlot, riPlot, nrow = 2)

## Summarize the performance and the time need to train each model in a table and comment on it.
# Calculate the time needed to train each model
start_lasso  <-  Sys.time()
cv.lasso     <-  cv.glmnet(X, y, alpha = 1, nfolds = 10)
lasso        <-  glmnet(X, y, alpha = 1, lambda = cv.lasso$lambda.min)
end_lasso    <-  Sys.time()
time_lasso   <-  end_lasso - start_lasso

start_elast  <-  Sys.time()
cv.elast     <-  cv.glmnet(X, y, alpha = 0.5, nfolds = 10)
elast        <-  glmnet(X, y, alpha = 0.5, lambda = cv.elast$lambda.min)
end_elast    <-  Sys.time()
time_elast   <-  end_elast - start_elast

start_ridge  <-  Sys.time()
cv.ridge     <-  cv.glmnet(X, y, alpha = 0, nfolds = 10)
ridge        <-  glmnet(X, y, alpha = 0, lambda = cv.ridge$lambda.min)
end_ridge    <-  Sys.time()
time_ridge   <-  end_ridge - start_ridge

start_rf     <-  Sys.time()
rf           <-  randomForest(X, y, mtry = sqrt(p), importance = TRUE, ntree = 100, nodesize = 100)
end_rf       <-  Sys.time()
time_rf      <-  end_rf - start_rf


model <- c('Lasso', 'ElasticNet', 'Ridge', 'RandomForest')
performance <- round(c(mean(Rsq.test.la), mean(Rsq.test.en), mean(Rsq.test.ri), mean(Rsq.test.rf)), 3)
time <- round(c(time_lasso, time_elast, time_ridge, time_rf), 2)

summary_table <- data.frame(model, performance, time)
write.csv(summary_table, 'summary_table.csv')
