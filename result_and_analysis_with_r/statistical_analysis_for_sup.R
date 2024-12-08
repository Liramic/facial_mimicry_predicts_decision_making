source("my_functions_and_imports.R")
current_path <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(current_path)


############# statistical analysis for the reading part ##################

reading_data = read.csv("reading.csv")
reading_data$ListenerChoice = as.factor(reading_data$ListenerChoice)

reading_data = normalize_and_centralize(reading_data, "minus_positive_sync")
reading_data = normalize_and_centralize(reading_data, "minus_negative_sync")
reading_data = normalize_and_centralize(reading_data, "minus_positive_listener_mean")
reading_data = normalize_and_centralize(reading_data, "minus_negative_listener_mean")

reading_model <- glm(ListenerChoice ~ 1 + minus_positive_sync + minus_negative_sync,
                            family = binomial(link = "logit"), data = reading_data)
summary_model = summary(reading_model)
print(summary_model)

reading_model_with_activations <- glm(ListenerChoice ~ 1 + minus_positive_sync + minus_negative_sync + minus_positive_listener_mean + minus_negative_listener_mean,
                            family = binomial(link = "logit"), data = reading_data)
summary_model_with_activations = summary(reading_model_with_activations)
print(summary_model_with_activations)

# compute chi:

# Fit a null model (intercept only)
null_model <- glm(ListenerChoice ~ 1, family = binomial(link = "logit"), data = reading_data)

# Calculate the chi-squared statistic for the model fit
chi_squared <- null_model$deviance - reading_model$deviance
df <- null_model$df.residual - reading_model$df.residual

# P-value for the chi-squared test
p_value <- pchisq(chi_squared, df, lower.tail = FALSE)

# Print the chi-squared test result
cat("Chi-squared:", chi_squared, "\n")
cat("Degrees of freedom:", df, "\n")
cat("P-value:", p_value, "\n")

#odds ratio and 95% CI:
exp(cbind(OR = coef(reading_model), confint(reading_model)))

# plots:
a = allEffects(reading_model)
plot(a$minus_positive_sync)
plot(a$minus_negative_sync, colors = "red")


############### statistical analysis for listening part ####################

data_listening = read.csv("listening.csv")
data_listening$choice = as.factor(data_listening$choice)

data_listening = normalize_and_centralize(data_listening, "minus_positive_sync")
data_listening = normalize_and_centralize(data_listening, "minus_negative_sync")
data_listening = normalize_and_centralize(data_listening, "minus_positive_listener_mean")
data_listening = normalize_and_centralize(data_listening, "minus_negative_listener_mean")

data_listening = normalize_and_centralize(data_listening, "minus_nasalis")
data_listening = normalize_and_centralize(data_listening, "minus_zygomaticus_major")


###############         participants' self choice:       ###################
data_listening_self = data_listening[data_listening$is_other == 0,]

listening_model_self <- glm(choice ~ 1 + minus_positive_sync + minus_negative_sync,
                            family = binomial(link = "logit"), data = data_listening_self)
summary_model_self = summary(listening_model_self)
print(summary_model_self)

listening_model_with_activations_self <- glm(choice ~ 1 + minus_positive_sync + minus_negative_sync + minus_positive_listener_mean + minus_negative_listener_mean,
                            family = binomial(link = "logit"), data = data_listening_self)
summary_model_with_activations_self = summary(listening_model_with_activations_self)
print(summary_model_with_activations_self)

# compute chi:
null_model_self <- glm(choice ~ 1, family = binomial(link = "logit"), data = data_listening_self)

chi_squared_self <- null_model_self$deviance - listening_model_self$deviance
df_self <- null_model_self$df.residual - listening_model_self$df.residual

p_value_self <- pchisq(chi_squared_self, df_self, lower.tail = FALSE)

cat("Chi-squared:", chi_squared_self, "\n")
cat("Degrees of freedom:", df_self, "\n")
cat("P-value:", p_value_self, "\n")

#OR:
exp(cbind(OR = coef(listening_model_self), confint(listening_model_self)))

#plots:
a_self = allEffects(listening_model_self)
plot(a_self$minus_positive_sync)
plot(a_self$minus_negative_sync, colors = "red")


###############         participants' other choice:       ###################

listening_data_other = data_listening[data_listening$is_other == 1,]

model <- glm(choice ~  minus_nasalis + minus_zygomaticus_major,
             family = binomial(link = "logit"), data = listening_data_other)
summary(model)
plot(allEffects(model))

# Fit a null model (intercept only)
null_model <- glm(choice ~ 1, family = binomial(link = "logit"), data = listening_data_other)

# Calculate the chi-squared statistic for the model fit
chi_squared_other <- null_model$deviance - model$deviance
df_other <- null_model$df.residual - model$df.residual

# P-value for the chi-squared test
p_value_other <- pchisq(chi_squared_other, df_other, lower.tail = FALSE)

cat("Chi-squared:", chi_squared_other, "\n")
cat("Degrees of freedom:", df_other, "\n")
cat("P-value:", p_value_other, "\n")

#calculate odds ratio and 95% CI
exp(cbind(OR = coef(model), confint(model)))


library(ggplot2)

# Create a data frame with a range of values for nose and zygomaticus
plot_data <- expand.grid(
  minus_nasalis = seq(min(listening_data_other$minus_nasalis), max(listening_data_other$minus_nasalis), length.out = 100),
  minus_zygomaticus_major = seq(min(listening_data_other$minus_zygomaticus_major), max(listening_data_other$minus_zygomaticus_major), length.out = 100)
)

# Calculate predicted probabilities
plot_data$pred_nose <- predict(model, newdata = data.frame(minus_nasalis = plot_data$minus_nasalis, minus_zygomaticus_major = 0), type = "response")
plot_data$pred_zyg <- predict(model, newdata = data.frame(minus_nasalis = 0, minus_zygomaticus_major = plot_data$minus_zygomaticus_major), type = "response")

# Create the plot
ggplot(plot_data, aes(x = minus_nasalis)) +
  geom_line(aes(y = pred_nose, color = "minus_nasalis"), size = 1) +
  geom_line(aes(x = minus_zygomaticus_major, y = pred_zyg, color = "minus_zygomaticus_major"), size = 1) +
  scale_color_manual(values = c("minus_nasalis" = "red", "minus_zygomaticus_major" = "blue")) +
  labs(x = "Difference in activation (Story 1 - Story 2)", 
       y = "Probability of choosing Story 2",
       title = "Effect of muscle activation on story choice",
       color = "Muscle") +
  theme_minimal()
