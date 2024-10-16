####################################################################################################
##############                    SARIMA                                ############################
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
df <- df_csv[start_row:end_row, c(4)]
head(df)
tail(df)

TT <- 24
y <- ts(df,frequency = TT)
length(y) # 26304
ggtsdisplay(y,lag.max = 200)

adf.test(y)
# Time series is not stationary

# Run the KPSS test
test_result <- kpss.test(y)
print(test_result)
# p-value = 0.01. Not stationary

####################################################################################
# Import necessary library
library(forecast)

# Applying Box-Cox transformation
lambda <- BoxCox.lambda(na.omit(y))
z_boxcox <- BoxCox(na.omit(y), lambda)

# # Box-Cox transformation 
BoxCox.lambda(y) # 1.045611 not needed
BoxCox.lambda(z_boxcox) # 1.045611 not needed

z <- z_boxcox
head(z)

# Displaying the time series
ggtsdisplay(z, lag.max = 500)

# Daily seasonality first
z_s24diff <- diff(z, lag = 24, differences = 1)
ggtsdisplay(z_s24diff, lag.max = 100)

# # Weekly seasonality 
# z_s168diff <- diff(z_boxcox, lag = 168, differences = 1)
# ggtsdisplay(z_s168diff, lag.max = 500)

# # Regular differentiation
# z_rdiff <- diff(z_s24diff, differences = 1)
# ggtsdisplay(z_rdiff, lag.max = 500)

# In principle, daily seasonality is enough to provide a stationary time series

####################################################################################
# AutoARIMA
# Fit non-seasonal ARIMA model
daily_fit <- auto.arima(y, seasonal = TRUE)
daily_fit <- auto.arima(df, seasonal = TRUE)

 # Fitting seasonal model with estimated orders
msarima.fit <- msarima(y, orders=list(ar=c(5,2,0),i=c(0,1,0),ma=c(1,0,0)), lags=c(1,24,168), h=24, holdout = TRUE, silent = FALSE, initial = 'backcasting')
summary(msarima.fit)

# Model diagnosis
ggtsdisplay(msarima.fit$residuals, lag.max = 200)
coeftest(msarima.fit)
autoplot(msarima.fit)
# The phi coefficients represent the autoregressive terms in the model, and 
# the theta coefficient represents the moving average term. 

# Checking residuals
CheckResiduals.ICAI(msarima.fit, bins = 100, lag.max=500)
# ggtsdisplay(residuals(msarima.fit),lag.max = 200)

################################################################################
# BEST WINDOW SIZE
# 
# Initialize variables to store the best model and the smallest error
best_model <- NULL
smallest_error <- Inf
best_window <- 0

# Set the test size to 24
test_size <- 24

# Loop over window sizes from 12 to 52 weeks
for (weeks in 12:52) {
  # Convert weeks to hours
  window_size <- weeks * 7 * 24

  # Calculate the start of the test set
  test_start <- length(y) - test_size - window_size + 1

  # Skip if the window size is too large
  if (test_start < 1) {
    next
  }

  # Split the data into training and testing sets
  train <- y[1:(test_start-1)]
  test <- y[test_start:(test_start+test_size-1)]

  # Fit an ARIMA model and forecast the next 24 values
  model <- msarima(y, orders=list(ar=c(5,2,0),i=c(0,1,0),ma=c(1,0,0)), lags=c(1,24,168), h=24, holdout = TRUE, silent = FALSE, initial = 'backcasting')
  forecasts <- forecast(model, h = test_size)

  # Calculate the MAE loss
  loss <- MAE(test, forecasts$mean)

  # Print window size, ARIMA model, and loss
  print(paste0("Window size: ", weeks, " weeks"))

  # # Capture the output of the summary(model) and print it
  # arima_summary <- capture.output(summary(model))
  # cat(paste(arima_summary, collapse = "\n"), "\n")

  print(paste0("MAE loss: ", loss))

  # If this model has the smallest error so far, save it
  if (loss < smallest_error) {
    smallest_error <- loss
    best_model <- model
    best_window <- weeks
  }
}

# Printing the best window size and the corresponding MAE
print(paste("Best window size: ", best_window, " weeks"))
print(paste("MAE: ", smallest_error))

# # Best window size: 14 weeks

################################################################################
# SARIMA MODEL

# Now it's time for the test set. We test between september and november, included: 3 months.

## Load dataset -------------------------------------------------------------------------------------------------------
df_csv <- read.csv("df.csv", sep = ",")

# Slicing
start_row = max(1, 14593-14 * 7 * 24)  # 2019-05-26 00:00:00 44.17. Siendo 14593: 2019-09-01 00:00:00 46.53
end_row = min(nrow(df_csv), 16776) # 2019-11-30 23:00:00 40.13
df <- df_csv[start_row:end_row, c(4)]

TT <- 24
train <- ts(df,frequency = TT)

# Number of observations
n <- length(train)

# Setting the window size to 14 weeks
window_size <- 14 * 7 * 24 # 

# Vector to store losses
losses <- numeric(n - window_size)

# List to store residuals for model diagnostics
residuals <- list()

# Initialize an empty vector to hold all residuals
global_residuals <- c()

# List to store the Ljung-Box test p_value for the residuals
ljung.box <- list()

# Plotting for visualizing forecasts
plot(1:n, train, type="l", xlim=c(n-window_size, n), ylim=range(train), xlab="Time", ylab="precio_spot")

# Iterating each window in the training set
for (i in seq(1, (n - window_size), by = 24)) {
  day_num <- round(i / 24) + 1
  print(paste("Training window model: Day", day_num))
  
  # Model
  model <- msarima(train[i:(i + window_size - 1)], orders=list(ar=c(5,2,0),i=c(0,1,0),ma=c(1,0,0)), lags=c(1,24,168), h=24, holdout = TRUE, silent = FALSE, initial = 'backcasting')
  
  # Generating forecasts
  forecasts <- forecast(model, h = 24)
  
  # Extracting training residuals for diagnostics
  train_residuals <- model$fit$model$residuals
  
  # Plot and save the diagnostics for the model
  png(filename=paste0("diagnostics_day_", day_num, ".png"))
  plot(train_residuals, type="l", main="Training Residuals", xlab="Time", ylab="")
  abline(h=0, lty=2)
  dev.off()
  
  # MAE loss for the forecast
  actuals <- train[(i + window_size):(i + window_size + 23)]
  loss <- MAE(actuals, forecasts$mean)
  
  # Storing the loss
  losses[i] <- loss
  print(paste("Loss:", loss))
  
  # Storing residuals for model diagnostics
  residuals[[i]] <- train_residuals
  
  # Add the residuals to the global_residuals vector
  global_residuals <- c(global_residuals, train_residuals)
  
  # Computing Ljung-Box test
  ljung_box <- Box.test(train_residuals, type="Ljung-Box")
  print(paste("P-Value of forecasts is:",ljung_box$p.value))
  ljung.box[i] <- ljung_box$p.value
  
  # Plotting forecasted and actual values
  plot(actuals, type="l", col="black", xlab="Time", ylab="precio_spot", ylim=range(c(actuals, forecasts$mean)))
  lines(1:24, forecasts$mean, col="blue")
  legend("topright", legend=c("Actual", "Forecast"), fill=c("black", "blue"))
  
  Sys.sleep(1)
}


# Now, you can make a plot of the global residuals and conduct your diagnostics on this single series
png(filename="global_residuals_plot.png")
plot(global_residuals, type="l", main="Global Residuals", xlab="Time", ylab="")
abline(h=0, lty=2)
dev.off()

png(filename="global_acf_plot.png")
acf(global_residuals, main="Autocorrelation Function of Global Residuals")
dev.off()

ljung_box_global <- Box.test(global_residuals, type="Ljung-Box")
print(paste("Global Ljung-Box test p-value:", ljung_box_global$p.value))


