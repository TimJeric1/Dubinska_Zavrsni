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

data$Quality <- as.factor(data$Quality)


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


# Definiranje prediktora i ciljne varijable za trening i test skupove
train_features <- data_train[, -ncol(data_train)]
train_target <- data_train$Quality
test_features <- data_test[, -ncol(data_test)]
test_target <- data_test$Quality

train_target <- as.factor(train_target)
test_target <- as.factor(test_target)

# Klasifikacija najblizih susjeda,  k = 3
predicted_classes <- knn(train = train_features, test = test_features, cl = train_target, k = 3)

conf_matrix = confusionMatrix(predicted_classes, test_target)
conf_matrix

mosaicplot(conf_matrix$table, main = "Confusion Matrix Mosaic Plot", 
           color = terrain.colors(10), shade = TRUE)


# 3. Bagging

library(caret)
library(ipred)

# Dijeljenje podataka na testne i trening
indexes = createDataPartition(data$Quality, p = 0.8, list = FALSE)
data_train = data[indexes, ]
data_test = data[-indexes, ]


train_features <- data_train[, -ncol(data_train)]
train_target <- data_train$Quality
test_features <- data_test[, -ncol(data_test)]
test_target <- data_test$Quality

# Treniranje modela
bagged_model <- bagging(Quality ~ ., data = data_train)

# Važnost prediktora
importance <- varImp(bagged_model)
importance


predicted_classes <- predict(bagged_model, newdata = test_features)
#predicted_classes = ifelse(predicted_classes > 0.5 , 1, 0)
# Matrica zabune
conf_matrix = confusionMatrix(data = predicted_classes, reference = test_target)
conf_matrix

mosaicplot(conf_matrix$table, main = "Confusion Matrix Mosaic Plot", 
           color = terrain.colors(10), shade = TRUE)



# 4. Učenje asocijacijskih pravila
install.packages("arules")

library(arules)

# Pretvaranje u tip transactions
transactions <- as(as.matrix(binary_data), "transactions")


# Učenje pravila algoritmom apriori
rules <- apriori(transactions, parameter = list(support = 0.1, confidence = 0.5))
rules <- sort(rules, by="confidence", decreasing=TRUE)
rules <- head(rules, n=10)

# Ispis pravila
inspect(rules)


# 5. Hijerarhijsko grupianje

distance_matrix <- dist(data[, -ncol(data)])
hclust_result <- hclust(distance_matrix, method = "complete")

# Pretvaranje hclust-a u dendogram klasu
hcd <- as.dendrogram(hclust_result)

#Lista parametara za plotanje
nodePar <- list(lab.cex = 0.6, pch = c(NA, 19), 
                cex = 0.8, col = "cyan")

# Plotanje cijelog dendograma
plot(hcd, main = "Hierarchical Clustering Dendrogram", xlab = "Samples", ylab = "Height", 
     nodePar = nodePar, edgePar = list(col = 2:3, lwd = 2:1))

fit <- cutree(hclust_result, k = 6)
#fit
table(fit, data$Quality)
rect.hclust(hclust_result,k = 6, border = "blue")

# Plotanje lijevog dijela dendograma da se lakše vidi
plot(hcd, main = "Hierarchical Clustering Dendrogram", xlab = "Samples", ylab = "Height", 
     nodePar = nodePar, edgePar = list(col = 2:3, lwd = 2:1), xlim = c(1, 19), ylim = c(0,9))