library(dplyr)
library(readxl)
Iris <- read_excel("Iris.xlsx")
View(Iris)

# create a scatter plot of mtcars data 
plot(Iris$SepalLengthCm, Iris$PetalLengthCm, 
     xlab = "SepalLength (Cm)", 
     ylab = "PetalLength (Cm)", 
     main = "Scatter plot")

# create a bar chart of the table 
# of gear counts in the mtcars data 
barplot(table(Iris$SepalWidthCm), 
        xlab = "SepalWidth", 
        ylab = "Frequency", 
        main = "Bar chart")

# create a box plot of the horsepower 
# variable in the mtcars data 
boxplot(Iris$PetalWidthCm, 
        xlab = "PetalWidth", 
        main = "Box plot")

# create a density plot of the 
# displacement variable in the mtcars data 
plot(density(Iris$SepalLengthCm),
     xlab = "SepalLength", 
     main = "density function")

# create a heatmap of the correlation 
# matrix for the mtcars data 
corr_matrix <- cor(Iris[, c('SepalLengthCm','SepalWidthCm','PetalLengthCm', 'PetalWidthCm')])
heatmap(corr_matrix,
        main = "Heatmap of the correlation matrix")

# Create a histogram of the "mtcars" dataset 
data(Iris) 
hist(Iris$PetalLengthCm, breaks = 5, 
     main = "Distribution", 
     xlab = "PetalLength", 
     ylab = "Frequency") 

# Create a pie chart 
x <- c(Iris$Species)
y <- c(sum(x == "Iris-setosa"), sum(x == "Iris-versicolor"), sum(x == "Iris-virginica"))
pie (y, labels = c("Iris-setosa", "Iris-versicolor", 
                     "Iris-virginica"), 
    main = "Pie Chart of Iris")

# Create a stepped line graph 
plot(Iris$SepalLengthCm, type = "s", 
     main = "Stepped Line Graph", 
     xlab = "Index", 
     ylab = "Values", 
     col = "blue", 
     lwd = 2, lend = "round") 

# Create a dataset 
data(Iris) 
x <- Iris[, 2:5] 

# Create a matrix of scatter plots 
pairs(x, main = "Matrix of Scatter Plots of Iris", 
      col = "blue") 

