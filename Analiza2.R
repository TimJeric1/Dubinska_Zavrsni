install.packages("dplyr")
library(dplyr)

# Čitanje dataseta iz csv
current_directory <- getwd()
file_path <- file.path(current_directory, "diabetes_risk_prediction_dataset.csv")
data <- read.csv(file_path)
# 1. Binarizacija
data$Age = as.numeric(data$Age)
# Binarize the data
binary_data <- data
binary_data$Age <- as.factor(binary_data$Age)
binary_data$Gender <- ifelse(binary_data$Gender == "Male", 1, 0)
binary_data$Polyuria <- ifelse(binary_data$Polyuria == "Yes", 1, 0)
binary_data$Polydipsia <- ifelse(binary_data$Polydipsia == "Yes", 1, 0)
binary_data$sudden.weight.loss <- ifelse(binary_data$sudden.weight.loss == "Yes", 1, 0)
binary_data$weakness <- ifelse(binary_data$weakness == "Yes", 1, 0)
binary_data$Polyphagia <- ifelse(binary_data$Polyphagia == "Yes", 1, 0)
binary_data$Genital.thrush <- ifelse(binary_data$Genital.thrush == "Yes", 1, 0)
binary_data$visual.blurring <- ifelse(binary_data$visual.blurring == "Yes", 1, 0)
binary_data$Itching <- ifelse(binary_data$Itching == "Yes", 1, 0)
binary_data$Irritability <- ifelse(binary_data$Irritability == "Yes", 1, 0)
binary_data$delayed.healing <- ifelse(binary_data$delayed.healing == "Yes", 1, 0)
binary_data$partial.paresis <- ifelse(binary_data$partial.paresis == "Yes", 1, 0)
binary_data$muscle.stiffness <- ifelse(binary_data$muscle.stiffness == "Yes", 1, 0)
binary_data$Alopecia <- ifelse(binary_data$Alopecia == "Yes", 1, 0)
binary_data$Obesity <- ifelse(binary_data$Obesity == "Yes", 1, 0)
binary_data$class <- as.factor(binary_data$class)


# 2. k-najbližih susjeda

install.packages("caret")
library(caret)
library(class)
# Dijeljenje podataka na testne i trening
indexes = createDataPartition(data$class, p = 0.8, list = FALSE)
data_train = binary_data[indexes, ]
data_test = binary_data[-indexes, ]


# Define features and target variable for training and test sets
train_features <- data_train[, -ncol(data_train)]
train_target <- data_train$class
test_features <- data_test[, -ncol(data_test)]
test_target <- data_test$class

knn_model <- knn(train_features, test_features, factor(train_target), 3)

# Make predictions on test data
predicted_classes <- as.factor(knn_model)

# Print the confusion matrix
confusionMatrix(predicted_classes, test_target)


# 3. Bagging

# Load the required library
library(caret)
# Define predictors and target
predictors <- data[, -ncol(data)]
target <- data$class

# Define the trainControl
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

# Train the bagging model
bagging_model <- train(class ~ ., data = data, method = "treebag", trControl = train_control)

# Summary of bagging model
print(bagging_model)

# Make predictions on the training data
predicted_classes <- predict(bagging_model, predictors)

# Confusion matrix
confusionMatrix(predicted_classes, as.factor(target))



# 4. Učenje asocijacijskih pravila
install.packages("arules")

# Load the required library
library(arules)

# Create a transactions object
transactions <- as(data, "transactions")


# Mine association rules
rules <- apriori(transactions, parameter = list(support = 0.1, confidence = 0.5))
rules <- sort(rules, by="confidence", decreasing=TRUE)
rules <- head(rules, n=10)

# Print the rules
inspect(rules)


# 5. Hijerarhijsko grupiranje

# Perform hierarchical clustering
distance_matrix <- dist(data[, -ncol(data)])  # Compute distance matrix
hclust_result <- hclust(distance_matrix, method = "complete")  # Perform hierarchical clustering

# Plot the dendrogram
plot(hclust_result, main = "Hierarchical Clustering Dendrogram", xlab = "Samples", sub = NULL,
     ylab = "Distance", labels = data$Quality)
fit <- cutree(hclust_result, k = 4)
table(fit, data$class)

rect.hclust(hclust_result,k=4,border="green")









