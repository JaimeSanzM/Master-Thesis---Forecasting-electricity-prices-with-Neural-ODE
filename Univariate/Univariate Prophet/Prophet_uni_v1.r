####################################################################################################
##############                    Univariate Prophet                    ############################
##############  TFM - Forecasting Electricity Prices with Neural ODE    ############################
####################################################################################################

library(MLTools)
library(fpp2)
library(ggplot2)
library(readxl)
library(lmtest)  #contains coeftest function
library(tseries) #contains adf.test function
library(smooth)
library(Rcpp)
library(tfarima)
library(Rcpp)
library(smooth)
library(forecast)
library(Metrics)
library(Rlgt)
library(prophet)
library(dplyr)
library(scales)

## Set working directory ---------------------------------------------------------------------------------------------
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

## Load dataset -------------------------------------------------------------------------------------------------------
df_csv <- read.csv("df.csv", sep = ",")

head(df_csv)
tail(df_csv)

# Slicing
# df consists of the previous 14 months to the test set. 
# It is the same training size as the DL methods: 14 months
start_row = max(1, 4345) # 2018-07-01 00:00:00 62.0
end_row = min(nrow(df_csv), 14592) # 2019-08-31 23:00:00 44.8
# df <- df_csv[start_row:end_row, c(4)]
df <- df_csv[start_row:end_row, ]
head(df)
tail(df)

# Names of the variables of the dataset in Prophet
df$ds <- as.POSIXct(df$datetime, format="%Y-%m-%d %H:%M:%S")
df$y <- as.numeric(df$price)  

df[which(is.na(df$ds)), "ds"] <- as.POSIXct("2019-03-31 02:00:00")

head(df)
# Cross-validation
# Renaming the 'datetime' column to 'ds' and selecting only the 'ds' and 'y' columns
df <- df[, c("ds", "y")]
head(df)
sum(is.na(df$ds))  # This should return 0

windows <- seq(168, 168*11, 168) # Cross-validation, from 1 week to 11 weeks

best_mae <- Inf
best_window <- NA

mean_maes <- list()  # List to store the mean MAE for each window size

for (window in windows) {
  
  maes <- c()
  
  for (i in seq(window, nrow(df) - 24, 24)) {  # Shift the window by 24 hours
    
    # Informative print
    print(paste("Evaluating window:", i - window + 1, "to", i, "for window size", window))
    
    train <- df[(i-window+1):i, ]
    test <- df[(i+1):(i+24), ]
    
    m <- prophet(df = train, daily.seasonality = TRUE, weekly.seasonality = TRUE)  # Enable daily seasonality
    future <- make_future_dataframe(m, periods = 24, freq = "hour", include_history = FALSE)
    forecast <- predict(m, future)
    
    mae <- mean(abs(test$y - forecast$yhat), na.rm = TRUE)  
    maes <- c(maes, mae)
    
    # Print the current MAE
    print(paste("Current MAE =", mae))
  }
  
  avg_mae <- mean(maes, na.rm = TRUE)  # Remove any NA values for average MAE calculation
  print(paste("Window size", window, ": average MAE =", avg_mae))
  
  mean_maes[[as.character(window)]] <- avg_mae  # Store the mean MAE for the current window size
  
  if (avg_mae < best_mae) {
    best_mae <- avg_mae
    best_window <- window
  }
}

print(paste("Best window size is", best_window, "with average MAE", best_mae))
print("Mean MAE for each window size:")
print(mean_maes)

# Best window size: 6 weeks
best_window <- 6*168
################################################################################
# Test

## Set working directory ---------------------------------------------------------------------------------------------
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(lubridate)
library(timeDate)
library(dplyr)
library(ggplot2)

# Load dataset
df_csv <- read.csv("df.csv", sep = ",")

# Slicing
start_row = max(1, 14592 - best_window + 1) # 2019-07-21 00:00:00 50.28
end_row = min(nrow(df_csv), 16777) # 2019-11-30 23:00:00 40.13
df <- df_csv[start_row:end_row, ]
head(df)
tail(df)

# Rename the variables for Prophet
df$ds <- as.POSIXct(df$datetime, format="%Y-%m-%d %H:%M:%S")
df$y <- as.numeric(df$price)

sum(is.na(df$ds))  # This should return 0
# which(is.na(df$ds))
# 
# Drop unnecessary columns
df <- df[, c("ds", "y")]

# Define the train and test datasets
train <- df[df$ds < as.POSIXct("2019-09-01 00:00:00"), ]
test <- df[df$ds >= as.POSIXct("2019-09-01 00:00:00"), ]
head(train)
tail(train)
head(test)
tail(test)

# Initialize lists to store results
maes <- c()
residuals <- c()

# Start the loop from the end of the training set and go to the end of the entire dataset
for (i in seq(nrow(train), nrow(df)-24, 24)) {
  
  # Get the training and testing sets
  train_temp <- df[(i-best_window+1):i, ]
  test_temp <- df[(i+1):(i+24), ]
  
  # Train the Prophet model and make predictions
  m <- prophet(df = train_temp, daily.seasonality = TRUE, weekly.seasonality = TRUE)  # Enable daily seasonality)
  future <- make_future_dataframe(m, periods = 24, freq = "hour", include_history = FALSE)
  forecast <- predict(m, future)
  
  # Calculate the MAE and residuals
  mae <- mean(abs(test_temp$y - forecast$yhat))
  maes <- c(maes, mae)
  residuals <- test_temp$y - forecast$yhat
  
  # Plot the actuals vs forecasts
  plot_df <- data.frame(ds = forecast$ds, yhat = forecast$yhat, y = test_temp$y)
  p <- ggplot(plot_df) +
    geom_line(aes(x = ds, y = y), color = "blue") +
    geom_line(aes(x = ds, y = yhat), color = "red") +
    labs(title = paste("Actuals vs Forecasts for", format(min(forecast$ds), "%Y-%m-%d")), x = "Date", y = "Value") +
    theme_minimal()
  
  # Save the plot as a separate file
  ggsave(filename = paste0("plot_", format(min(forecast$ds), "%Y%m%d"), ".png"), plot = p)
  
  # Print the MAE
  print(paste("The MAE of the forecast for", format(min(forecast$ds), "%Y-%m-%d"), "is", mae))
}

# Print the average MAE
print(paste("The average MAE of the forecast for the test period is", mean(maes)))
