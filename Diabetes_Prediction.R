
##################################
# Load data and required packages
##################################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(rpart.plot)) install.packages("rpart.plot", repos = "http://cran.us.r-project.org")
if(!require(gbm)) install.packages("gbm", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(devtools)) install.packages("devtools", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(ggcorrplot)) install.packages("ggcorrplot", repos = "http://cran.us.r-project.org")
library(devtools)

# Load the Diabetes data set from my github account
diabetes_data <- read.table("https://raw.githubusercontent.com/apurba-das/Diabetes-Prediction/master/pima-indians-diabetes.data.txt", 
                          header = FALSE, 
                          sep = ",")

colnames(diabetes_data) <- c("Pregnancies",
                            "Glucose",
                            "BloodPressure",
                            "SkinThickness",
                            "Insulin",
                            "BMI",
                            "DiabetesPedigreeFunction",
                            "Age",
                            "Outcome")

diabetes_data$Outcome <- ifelse(diabetes_data$Outcome == "1", "Diabetes",
                          ifelse(diabetes_data$Outcome == "0", "NoDiabetes", NA))

##################################
# Data Exploration
##################################

diabetes_data[diabetes_data == "?"] <- NA

# how many NAs are in the data
length(which(is.na(diabetes_data)))

# There are no NAs in the dataset, so there is no need to remove or impute any rows.

# See the initial few rows of the data
head(diabetes_data)

# Understand the diabetes data better
str(diabetes_data)

# Check the number of patients affected with diabetes and those without it
diabetes_data$Outcome <- as.factor(diabetes_data$Outcome)

summary(diabetes_data$Outcome)

library(ggplot2)

# Plot which shows the imbalance in the dataset
ggplot(diabetes_data, aes(x = Outcome, fill = Outcome)) +
  geom_bar()

##################################
# Correlation Matrix
##################################

# Compute correlation matrix
data_corr <- round(cor(diabetes_data[1:8]),1)
data_corr

# Plot the correlation matrix
library(ggcorrplot)
ggcorrplot(data_corr)
# There is no strong correlation between any of the diagnosed factors and they are pretty much independent.

##################################
# Feature Importance
##################################

#Understanding the features
library(tidyr)

gather(diabetes_data, x, y, Pregnancies:Age) %>%
  ggplot(aes(x = y, color = Outcome, fill = Outcome)) +
  geom_density(alpha = 0.3) +
  facet_wrap( ~ x, scales = "free", ncol = 3)
# Glucose looks like the most important feature from the graphs


##################################
# Split into train and test set
##################################

library(caret)
set.seed(42) # for reproducibility
test_index <- createDataPartition(diabetes_data$Outcome, p = 0.9, list = FALSE)
train_data <- diabetes_data[test_index, ]
test_data  <- diabetes_data[-test_index, ]

# Check whether feature distribution is similar in the train and test set
library(dplyr)

rbind(data.frame(group = "train", train_data),
      data.frame(group = "test", test_data)) %>%
  gather(x, y, Pregnancies:Age) %>%
  ggplot(aes(x = y, color = group, fill = group)) +
    geom_density(alpha = 0.3) +
    facet_wrap( ~ x, scales = "free", ncol = 3)
# We find that the distribution between the two is similar

##################################
# Modeling
##################################

# Variable ComputationControl is used for controlling the computation of train function in the models that we build

ComputationControl <- trainControl(method="repeatedcv",    
                           number = 15, 
                           repeats = 10,
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary)


##################################
# Logistic Regression
################################## 

model_reg <- caret::train(Outcome ~ .,
                          data = train_data,
                          method = "glm",
                          metric = "ROC", 
                          preProcess = c("scale", "center"),
                          trControl =  ComputationControl)
# Prediction
prediction_reg<- predict(model_reg, test_data)

# Confusion matrix
confusionMat_reg <- confusionMatrix(prediction_reg, test_data$Outcome)

confusionMat_reg

# Accuracy of model
accuracy_reg <- confusionMat_reg$overall['Accuracy']

# Tibble containing the accuracy results
accuracy_results <- tibble(Method = "Logistic Regression", Accuracy = accuracy_reg)

# Viewing the results obtained so far

accuracy_results %>% knitr::kable()

# Plot of top five most important variables
plot(varImp(model_reg), top=5, main="Top variables - Logistic Regression")

##################################
# KNN (K Nearest Neighbor)
################################## 

model_knn <- caret::train(Outcome ~ .,
                             data = train_data,
                             method = "knn",
                             metric = "ROC", 
                             preProcess = NULL,
                             trControl =  ComputationControl)

# Prediction
prediction_knn<- predict(model_knn, test_data)

# Confusion matrix
confusionMat_knn <- confusionMatrix(prediction_knn, test_data$Outcome)

confusionMat_knn

# Accuracy of model
accuracy_knn <- confusionMat_knn$overall['Accuracy']

# Add the current model accuracy to the results
accuracy_results <- add_row(accuracy_results, Method = "KNN", Accuracy = accuracy_knn)

# Viewing the results obtained so far:

accuracy_results %>% knitr::kable()

# Plot of top five most important variables
plot(varImp(model_knn), top=5, main="Top variables - KNN")

##################################
# Decision Tree
##################################

library(rpart)
library(rpart.plot)

model_dt <- rpart(Outcome ~ .,
                  data = train_data,
                  method = "class",
                  control = rpart.control(xval = 10, 
                                          minbucket = 2, 
                                          cp = 0), 
                  parms = list(split = "information"))

# Plot the tree
rpart.plot(model_dt, extra = 100)

# Prediction
prediction_dt<- predict(model_dt, test_data, type = "class")

# Confusion matrix
confusionMat_dt <- confusionMatrix(prediction_dt, test_data$Outcome)

confusionMat_dt

# Accuracy of model
accuracy_dt <- confusionMat_dt$overall['Accuracy']

# Add the current model accuracy to the results
accuracy_results <- add_row(accuracy_results, Method = "Decision Tree", Accuracy = accuracy_dt)

# Viewing the results obtained so far:
accuracy_results %>% knitr::kable()

##################################
# Random Forest
##################################

model_rf <- caret::train(Outcome ~ .,
                         data = train_data,
                         method = "rf",
                         metric = "ROC", 
                         preProcess = c("scale", "center"),
                         trControl = ComputationControl)

# Prediction
prediction_rf<- predict(model_rf, test_data)

# Confusion matrix
confusionMat_rf <- confusionMatrix(prediction_rf, test_data$Outcome)

confusionMat_rf

# Accuracy of model
accuracy_rf <- confusionMat_rf$overall['Accuracy']

# Add the current model accuracy to the results
accuracy_results <- add_row(accuracy_results, Method = "Random Forest", Accuracy = accuracy_rf)

# Viewing the results obtained so far
accuracy_results %>% knitr::kable()

# Plot of top five most important variables
plot(varImp(model_rf), top=5, main="Top variables - Random Forest")

##################################
# Gradient Boost
##################################

model_gb <- caret::train(Outcome ~ .,
                         data = train_data,
                         method = "gbm",
                         metric = "ROC", 
                         trControl = ComputationControl,
                         verbose = 0)

# Prediction
prediction_gb<- predict(model_gb, test_data)

# Confusion matrix
confusionMat_gb <- confusionMatrix(prediction_gb, test_data$Outcome)

confusionMat_gb

# Accuracy of model
accuracy_gb <- confusionMat_gb$overall['Accuracy']

# Add the current model accuracy to the results
accuracy_results <- add_row(accuracy_results, Method = "Gradient Boosting", Accuracy = accuracy_gb)

# Viewing the results obtained so far
accuracy_results %>% knitr::kable()

# Plot of top five most important variables
plot(varImp(model_gb), top=5, main="Top variables - Gradient boost")

##################################
# Final accuracy results
##################################

accuracy_results %>% knitr::kable()
