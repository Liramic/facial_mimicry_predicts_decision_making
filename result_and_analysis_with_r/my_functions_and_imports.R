library(lme4)
library(dplyr)
library(effects)
library("ordinal")
library("MASS")

create_minus_column <- function(data, col1, col2)
{
  return((data[,col1] - data[,col2])/(data[,col1]+data[,col2]))
}

create_minus_column_single_name <- function(data, col_name)
{
  col1 <- paste0(col_name, "_1")
  col2 <- paste0(col_name, "_2")
  return(create_minus_column(data, col1, col2))
}


convert_to_factors <- function(data, colnames) {
  for (colname in colnames) {
    data[[colname]] <- as.factor(data[[colname]])
  }
  return(data)
}

normalize_and_centralize <- function(data, colname) {
  ## usage example:
  #  data_normalized <- normalize_and_centralize(data, "scores")
  
  
  # Check if the specified column exists in the dataframe
  if (!colname %in% names(data)) {
    stop("Column not found in the dataframe")
  }
  
  # Extract the column from the dataframe
  column_data <- data[[colname]]
  
  # Calculate mean and standard deviation of the column
  mean_data <- mean(column_data, na.rm = TRUE)
  sd_data <- sd(column_data, na.rm = TRUE)
  
  # Centralize and normalize the column
  normalized_data <- (column_data - mean_data) / sd_data
  
  # Replace the original column with the normalized data
  data[[colname]] <- normalized_data
  
  # Return the modified dataframe
  return(data)
}
