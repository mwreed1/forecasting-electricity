# load libraries

library(tidyverse)
library(lubridate)
library(prophet)
library(MLmetrics)

# read pre-processed data

data <- read_csv("../raw_data/weekly_usage.csv")

# clean data

data <- data %>%
  mutate(
    date = lubridate::parse_date_time(paste(year, week, 1, sep="/"),'Y/W/w')
  ) %>%
  select(date, account, value, t)

get_account <- function(account_name){
  data %>%
    filter(account == account_name) %>%
    select(date, value) %>%
    rename(ds = date, y = value)
}


split_data <- function(data){
  n <- nrow(data)
  N_train <- round(0.7*n, 0)
  N_validation <- round(0.2*n, 0)
  N_test <- n - N_train - N_validation
  
  train <- data %>%
    slice(1:N_train)
  
  validation <- data %>%
    anti_join(train) %>%
    slice(1: N_validation)
  
  test <- data %>%
    anti_join(train) %>%
    anti_join(validation)
  return(list(
    "train" = train,
    "validation" = validation,
    "test" = test
  ))
}


find_mape <- function(forecast, test){
  n = nrow(test)
  s <- round(n/3, 0)
  
  pred_test <- forecast %>%
    slice_tail(n=n) %>%
    pull(yhat)
  
  true_test <- test %>%
    pull(y)
  
  return(list(
    "total" = MAPE(pred_test, true_test),
    "first" = MAPE(pred_test[1:s], true_test[1:s]),
    "second" = MAPE(pred_test[(s+1):(2*s)], true_test[(s+1):(2*s)]),
    "third" = MAPE(pred_test[(2*s+1):nrow(test)], true_test[(2*s+1):nrow(test)])
  ))
}

split_mape <- function(forecast, test){
  n = nrow(test)
  pred_test <- forecast %>%
    slice_tail(n=n) %>%
    pull(yhat)
  
  true_test <- test %>%
    pull(y)
  
  s <- round(n/3, 0)
  
  tibble(
    first = map2_dbl(pred_test[1:s], true_test[1:s], MAPE),
    second = map2_dbl(pred_test[(s+1):(2*s)], true_test[(s+1):(2*s)], MAPE),
    third = map2_dbl(pred_test[(2*s+1):(3*s)], true_test[(2*s+1):(3*s)], MAPE)
  )
}
