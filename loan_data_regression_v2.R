
library(dplyr)
library(ggplot2)
library(caret)


#loading data
loan_data <- read.csv("/Users/linus/Documents/BA_Portfolio/Auto_Encode/cleaned_loan_data.csv")

# Set seed for reproducibility
set.seed(123)

# Create train-test split
split <- createDataPartition(y = loan_data$Loan_Status, p = 0.75, list = FALSE)
train_data <- loan_data[split,]
test_data <- loan_data[-split,]


#fitting logit regression
model <- glm(Loan_Status ~ Gender + Married + Dependents + Education + Self_Employed + ApplicantIncome + CoapplicantIncome + LoanAmount + Loan_Amount_Term + Credit_History + Property_Area_Rural + Property_Area_Semiurban + Property_Area_Urban, data = train_data, family = binomial)

# Summary of the model to see results
# Predict on the test data
predictions <- predict(model, test_data, type = "response")
predicted_classes <- ifelse(predictions > 0.5, 1, 0)

# Convert predictions to factor for accuracy calculation
predicted_classes <- as.factor(predicted_classes)

# Calculate accuracy
accuracy <- sum(predicted_classes == test_data$Loan_Status) / nrow(test_data)
print(paste("Accuracy: ", accuracy))



