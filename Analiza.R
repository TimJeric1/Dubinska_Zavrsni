install.packages("dplyr")
library(dplyr)

# Čitanje dataseta iz csv
current_directory <- getwd()
file_path <- file.path(current_directory, "apple_quality.csv")
data <- read.csv(file_path)


# Struktura podataka
str(data)

# Pretvoren stupac "Acidity" u numeričkog tipa
data$Acidity = as.numeric(data$Acidity)

# Izbačen stupac ID
data = data[,-1]

head(data)
summary(data)

# Provjera za nedostajućim vrijednostima
sapply(data, function(x) sum(is.na(x)))

# Izbacivanje svih redova sa nedostajućim vrijednostima (samo 1)
data <- na.omit(data)




# 1. Binarizacija

binary_data <- mutate(data, across(where(is.numeric), ~ifelse(. > 0, 1, 0)))
binary_data$Quality <- ifelse(binary_data$Quality == "good", 1, 0)
colnames(binary_data) <- c("is_big", "is_heavy", "is_sweet", "is_crunchy", "is_juicy", "is_ripe", "is_acidic", "is_quality")

# 2. k-najbližih susjeda

install.packages("caret")
library(caret)
library(class)

# Dijeljenje podataka na testne i trening
indexes = createDataPartition(data$Quality, p = 0.8, list = FALSE)
data_train = data[indexes, ]
data_test = data[-indexes, ]


# Define features and target variable for training and test sets
train_features <- data_train[, -ncol(data_train)]
train_target <- data_train$Quality
test_features <- data_test[, -ncol(data_test)]
test_target <- data_test$Quality

# Convert target variable to factor
train_target <- as.factor(train_target)
test_target <- as.factor(test_target)

# Perform k-nearest neighbors classification with k = 3
predicted_classes <- knn(train = train_features, test = test_features, cl = train_target, k = 3)

# Create a dataframe combining test data with predicted classes
results <- data.frame(data_test, Predicted_Quality = predicted_classes)

# Visualize predictions
ggplot(results, aes(x = Size, y = Weight, color = Predicted_Quality, shape = Quality)) +
  geom_point(size = 3) +
  scale_color_manual(values = c("red", "blue"), labels = c("Bad", "Good")) +
  scale_shape_manual(values = c(4, 16), labels = c("Bad", "Good")) +
  labs(title = "K-Nearest Neighbors Predictions", x = "Size", y = "Weight", color = "Predicted Quality", shape = "True Quality") +
  theme_minimal()






# 2. Metoda potpornih vektora
install.packages("e1071")
install.packages("ggplot2")
library(e1071)
library(ggplot2)

svm_data = data
svm_data$Quality = ifelse(svm_data$Quality == "good", 1, 0)

# Dijeljenje podataka na testne i trening
indexes = createDataPartition(svm_data$Quality, p = 0.8, list = FALSE)
data_train = svm_data[indexes, ]
data_test = svm_data[-indexes, ]


# Define features and target variable for training and test sets
train_features <- data_train[, -ncol(data_train)]
train_target <- data_train$Quality
test_features <- data_test[, -ncol(data_test)]
test_target <- data_test$Quality

# Convert target variable to factor
train_target <- as.factor(train_target)
test_target <- as.factor(test_target)

# Train SVM model on training data
svm_model <- svm(Quality ~ ., data = data_train, kernel = "radial")

# Make predictions on test data
predicted_classes <- predict(svm_model, newdata = test_features)
predicted_classes = ifelse(predicted_classes > 0.5 , 1, 0)
conf_matrix = confusionMatrix(data = factor(predicted_classes), reference = factor(test_target))
conf_matrix


mosaicplot(conf_matrix$table, main = "Confusion Matrix Mosaic Plot", 
           color = terrain.colors(10), shade = TRUE)



# 3. Bagging

# Load the required library
library(caret)

# Define the training control for bagging
ctrl <- trainControl(method = "repeatedcv",  # 10-fold cross-validation
                     number = 10,            # number of folds
                     repeats = 3,            # repeated 3 times
                     search = "grid")        # use grid search for tuning

# Train the Bagged Decision Tree model
bagged_model <- train(Quality ~ .,               # formula
                      data = data_train,        # training data
                      method = "treebag",       # Bagging with decision trees
                      trControl = ctrl)         # training control

# Print the model
print(bagged_model)

# Make predictions on the test set
predicted_classes <- predict(bagged_model, newdata = test_features)
predicted_classes = ifelse(predicted_classes > 0.5 , 1, 0)
# Evaluate the performance of the model
conf_matrix = confusionMatrix(data = factor(predicted_classes), reference = factor(test_target))
conf_matrix

mosaicplot(conf_matrix$table, main = "Confusion Matrix Mosaic Plot", 
           color = terrain.colors(10), shade = TRUE)



# 4. Učenje asocijacijskih pravila
install.packages("arules")

# Load the required library
library(arules)

# Create a transactions object
transactions <- as(as.matrix(binary_data), "transactions")


# Mine association rules
rules <- apriori(transactions, parameter = list(support = 0.1, confidence = 0.5))
rules <- sort(rules, by="confidence", decreasing=TRUE)
rules <- head(rules, n=10)

# Print the rules
inspect(rules)


# 5. Hijerarhijsko grupianje

# Perform hierarchical clustering
distance_matrix <- dist(data[, -ncol(data)])  # Compute distance matrix
hclust_result <- hclust(distance_matrix, method = "complete")  # Perform hierarchical clustering

# Plot the dendrogram
plot(hclust_result, main = "Hierarchical Clustering Dendrogram", xlab = "Samples", sub = NULL,
     ylab = "Distance", labels = data$Quality)
fit <- cutree(hclust_result, k = 6)
fit
table(fit, data$Quality)

rect.hclust(hclust_result,k=2,border="green")


# 5. DBScan

# Install and load the dbscan package
install.packages("dbscan")
library(dbscan)

kNNdistplot(data[, -ncol(data)], k=5)

dbscan_result <- dbscan(data[, -ncol(data)], eps = 2.6, minPts = 3)


# Plot the clusters
plot(data[, c("Size", "Weight")], col = dbscan_result$cluster + 1, pch = dbscan_result$cluster + 1,
     main = "DBSCAN Clustering", xlab = "Size", ylab = "Weight")
legend("topright", legend = unique(dbscan_result$cluster), pch = unique(dbscan_result$cluster) + 1, 
       col = unique(dbscan_result$cluster) + 1, title = "Cluster")









