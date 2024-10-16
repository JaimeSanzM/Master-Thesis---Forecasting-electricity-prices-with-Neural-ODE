####################################################################################################
##############                    SARIMAX                               ############################
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
library(readr)
library(dplyr)
library(tidyr)
library(tidyverse)
library(corrplot)
library(forecast)
library(Rlgt)

## Set working directory ---------------------------------------------------------------------------------------------
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

## Load dataset -------------------------------------------------------------------------------------------------------
df_csv <- read.csv("df.csv", sep = ",")

head(df_csv)
tail(df_csv)

# Slicing
# df consists of the previous 14 months to the test set. 
# It is the same period of training as the DL methods
start_row = max(1, 2161) # 2018-04-01 00:00:00 40.43
end_row = min(nrow(df_csv), 12384) # 2019-05-31 23:00:00 55.98
df <- df_csv[start_row:end_row, ]

# Checking the first rows of the dataframe
head(df)

# Find number of rows in df
num_rows <- nrow(df)
print(num_rows) # 26304

# Checking the structure of the dataframe
str(df)
summary(df)

# Correlation of the variables
correlation_matrix <- cor(df[,-1])  # Remove datetime column for correlation analysis
corrplot(correlation_matrix, method = "color", addCoef.col = "black")

y <- ts(df)
ggtsdisplay(y[,"price"],lag.max = 200)

# Transforming data to time series
ts_data <- ts(df[, c("demand", "wind", "price")])

# Checking if there are any NA values in the dataset
anyNA(ts_data)

# Replacing NA with 0
df[is.na(df)] <- 0

# Total observations
num_rows <- nrow(df)
num_rows


################################################################################
# # Training set: first two years
# train <- df[1:(365*24*2),]
# 
# # Window size of 25 weeks
# window_size <- 15 * 7 * 24
# 
# # Initialize a vector to store losses
# losses <- numeric(n - window_size)
# anyNA(train)
# str(train)
# 
# # Loop over each window in the training set
# for (i in 1:(n - window_size)) {
#   # Fit the SARIMA model on the window and forecast the next 24 values
#   model <- msarima(train[i:(i + window_size - 1), 2:4], orders=list(ar=c(2,2,2),i=c(1,1,1),ma=c(4,4,4)), lags=c(1,24,168), h=24, holdout = TRUE, silent = FALSE, initial = 'backcasting')
#   forecasts <- predict(model, n.ahead = 24)
#   
#   # Calculate the MAE loss for the forecast
#   actuals <- train[(i + window_size):(i + window_size + 23), "precio_spot"]
#   loss <- mean(abs(forecasts$mean - actuals))
#   
#   # Store the loss
#   losses[i] <- loss
# }
# 
# # Test set: second year
# test <- df[(365 * 24 + 1):length(df),]
# 
# # Limit the number of iterations to 3 months
# n_test <- 3 * 30 * 24
# test_losses <- numeric(n_test - window_size)
# 
# # Loop over each window in the test set
# for (i in 1:(n_test - window_size)) {
#   # Fit the SARIMA model on the window and forecast the next 24 values
#   model <- msarima(test[i:(i + window_size - 1), 2:4], orders=list(ar=c(2,2,2),i=c(1,1,1),ma=c(4,4,4)), lags=c(1,24,168), h=24, holdout = TRUE, silent = FALSE, initial = 'backcasting')
#   forecasts <- predict(model, n.ahead = 24)
#   
#   # Calculate the MAE loss for the forecast
#   actuals <- test[(i + window_size):(i + window_size + 23), "precio_spot"]
#   loss <- mean(abs(forecasts$mean - actuals))
#   
#   # Store the loss
#   test_losses[i] <- loss
# }
# 
# # Calculate the global MAE
# global_mae <- mean(test_losses, na.rm = TRUE)
# print(paste("Global MAE loss on test set:", global_mAE))
# 



################################################################################
# SARIMAX MODEL

## Load dataset -------------------------------------------------------------------------------------------------------
df_csv <- read.csv("df.csv", sep = ",")

# Slicing
start_row = max(1, 14593-14 * 7 * 24)  # 2019-05-26 00:00:00 44.17. Siendo 14593: 2019-09-01 00:00:00 46.53
end_row = min(nrow(df_csv), 16776) # 2019-11-30 23:00:00 40.13
y <- df_csv[start_row:end_row, c("price")]
head(y)

TT <- 24
train <- ts(y,frequency = TT)

# train is the target time series and 'exogenous' is the dataframe of exogenous variables
exogenous <- df_csv[start_row:end_row, c("demand", "wind")]
str(exogenous)  # To check the structure
exogenous <- data.matrix(exogenous)  # Convert the data frame to a numeric matrix

fit <- auto.arima(train, xreg=exogenous, test="kpss",seasonal.test="ocsb", allowdrift=TRUE)
fit
# ARIMA(0,1,0) is deemed best when demand and wind are taken into account

# Number of observations
n <- length(train)
n # 4536
head(train)

# Window size to 14 weeks
window_size <- 14 * 7 * 24 # 2520

# Vector to store losses
losses <- numeric(n - window_size)

# List to store residuals for model diagnostics
residuals <- list()

# List to store the Ljung-Box test p_value for the residuals
ljung.box <- list()

# Plot for visualizing forecasts
plot(1:n, train, type="l", xlim=c(n-window_size, n), ylim=range(train), xlab="Time", ylab="precio_spot")

# Iterating each window in the training set
for (i in seq(1, (n - window_size), by = 24)) {
  day = ceiling(i / 24)
  print(paste("Training window model: Day", day))
  
  # Exogenous data of the current moving window
  exogenous_window <- as.matrix(exogenous[i:(i + window_size - 1), ])
  
  # Model
  model <- msarima(train[i:(i + window_size - 1)], xreg=exogenous_window, orders=list(ar=c(0,0,0),i=c(1,1,0),ma=c(0,0,0)), lags=c(1,24,168), h=24, holdout = TRUE, silent = FALSE, initial = 'backcasting')
  
  # Generating forecasts
  forecasts <- forecast(model, h = 24)
  
  # Plot and save the diagnostics for the model
  png(filename=paste0("diagnostics_", day, ".png"))
  checkresiduals(model)
  dev.off()
  
  # MAE loss for the forecast
  actuals <- train[(i + window_size):(i + window_size + 23)]
  loss <- MAE(actuals, forecasts$mean)
  
  # Storing the loss
  losses[i] <- loss
  print(paste("Loss:", loss))
  
  # Storing residuals for model diagnostics
  resid <- residuals(model)
  residuals[[i]] <- resid
  
  # Computing Ljung-Box test
  ljung_box <- Box.test(resid, type="Ljung-Box")
  print(paste("P-Value of forecasts is:",ljung_box$p.value))
  ljung.box[i] <- ljung_box$p.value
  
  # Plotting forecasted and actual values
  # png(filename=paste0("actual_vs_forecast_", day, ".png"))
  plot(actuals, type="l", col="black", xlab="Time", ylab="precio_spot", ylim=range(c(actuals, forecasts$mean)))
  lines(1:24, forecasts$mean, col="blue")
  legend("topright", legend=c("Actual", "Forecast"), fill=c("black", "blue"))
  # dev.off()
  
  # Optional: Pause after each plot
  Sys.sleep(1)
}

# Plotting residuals of the last model for diagnostics
png(filename="residuals_last_model.png")
plot(residuals[[length(residuals)]])
dev.off()

# As the windows are spaced by 24h, a new vector is created excluding the zeros 
losses_no_zeros <- losses[losses != 0]

# Computing the mean of the new vector
avg_loss <- mean(losses_no_zeros)
avg_loss # 5.678912

