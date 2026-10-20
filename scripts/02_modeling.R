#------------------------------------------------------------------------------#
#          Intro to Statistical Learning — Final Project
#          Predicting Critical Import Concentration in French Trade
#------------------------------------------------------------------------------#
# Author : Jules Tastet
# Date   : March–April 2026
# School : École Polytechnique, M1 Data & Economics for Public Policy
#
# This script trains and evaluates three classification models to predict
# whether a French import market (HS6 product × year) will become critically
# concentrated (HHI > 0.25) in the following year.
#
# Models compared:
#   1. Logistic Regression (baseline)
#   2. Elastic Net (glmnet, alpha = 0.5)
#   3. Random Forest
#
# Validation: time series split (4 expanding-window folds, 2011–2022)
# Final model: retrained on 2000–2021, evaluated on 2022–2023
#
# OUTPUT FILES (all saved to OUTPUT_DIR):
#   - fig_model_comparison.png
#   - fig_auc_by_fold.png
#   - fig_feature_importance.png
#   - fig_roc_curve.png
#   - fig_confusion_matrix.png
#   - model_comparison.csv
#   - rf_final_model.rds
#   - all_model_results.rds
#------------------------------------------------------------------------------#


# =============================================================================
# 0. SETUP
# =============================================================================

# install.packages(c("tidyverse", "randomForest", "caret", "glmnet", "pROC"))

library(tidyverse)
library(randomForest)
library(caret)
library(glmnet)
library(pROC)
library(ggplot2)

rm(list = ls())
set.seed(42)

# -----------------------------------------------------------------------------
# !! IMPORTANT — set this path to your local folder before running !!
# The folder must contain: data_ml_ready_v2.csv
# All figures and model outputs will also be saved here.
# -----------------------------------------------------------------------------
OUTPUT_DIR <- "/Users/ju/Desktop/FINAL_PROJECT_ML/"


# =============================================================================
# 1. DATA LOADING & PREPARATION
# =============================================================================

data_ml <- read.csv(paste0(OUTPUT_DIR, "data_ml_ready_v2.csv"))

cat("=== Raw dataset dimensions ===\n")
print(dim(data_ml))

cat("\n=== Target distribution (critical_t1) ===\n")
print(prop.table(table(data_ml$critical_t1)))

# -----------------------------------------------------------------------------
# Feature selection
#
# Note: log_commodity_price is excluded despite being available in the raw data.
# It exhibits ~75% missing values because it only covers commodity-intensive
# HS2 sectors (metals, energy, agricultural commodities). Keeping it and
# applying drop_na() would silently restrict the sample to those sectors,
# introducing severe selection bias. We therefore exclude it entirely.
# -----------------------------------------------------------------------------
features <- c(
  # Concentration — core predictors (HHI autocorrelation ≈ 0.83 at lag 1)
  "hhi_lag1", "hhi_lag2", "hhi_lag3",
  "delta_hhi",       # year-on-year change in HHI
  "hhi_trend_5y",    # 5-year rolling average — structural trend
  
  # Market structure
  "share_top_exporter",    # value share of the dominant supplier
  "share_top3_exporters",  # combined share of top 3 suppliers
  "share_top_lag1",        # lagged dominant share
  "Nb_exporters",          # number of distinct source countries
  
  # Import volume (log-transformed to handle extreme outliers)
  "log_total_import",
  "log_import_lag1",
  
  # Unit cost proxy (log-transformed — diamonds, pharma etc. create extreme values)
  "log_avg_unit_cost",
  
  # Macro controls
  "crisis_year",      # = 1 for 2008, 2009, 2020, 2021, 2022
  "log_world_trade",  # log of WTO world merchandise trade value
  
  # Sector fixed effect (HS2 = first 2 digits of HS6 code)
  "hs2"
)

target <- "critical_t1"

# Keep only relevant columns and drop rows with any missing value
df <- data_ml %>%
  select(all_of(c("product", "year", features, target))) %>%
  drop_na()

# -----------------------------------------------------------------------------
# Key descriptive stats cited in the report — computed here for traceability
# -----------------------------------------------------------------------------
hhi_autocor <- cor(
  data_ml$hhi[!is.na(data_ml$hhi) & !is.na(data_ml$hhi_lag1)],
  data_ml$hhi_lag1[!is.na(data_ml$hhi) & !is.na(data_ml$hhi_lag1)]
)
class_ratio_full <- sum(data_ml$critical_t1 == 0, na.rm = TRUE) /
  sum(data_ml$critical_t1 == 1, na.rm = TRUE)
hhi_pct_threshold <- mean(data_ml$hhi <= 0.25, na.rm = TRUE) * 100

cat("\n=== Key figures cited in the report ===\n")
cat(sprintf("HHI autocorrelation (lag 1)         = %.3f\n", hhi_autocor))
cat(sprintf("Class ratio (majority / minority)   = %.1f:1\n", class_ratio_full))
cat(sprintf("Percentile of HHI = 0.25 threshold  = %.1f%%\n", hhi_pct_threshold))

cat("\n=== Dataset after cleaning ===\n")
print(dim(df))

cat("\n=== Target distribution after cleaning ===\n")
print(table(df[[target]]))
print(prop.table(table(df[[target]])))

# Convert target to factor — required by caret and randomForest
df[[target]] <- factor(df[[target]], levels = c(0, 1),
                       labels = c("Non_Critique", "Critique"))


# =============================================================================
# 2. TIME SERIES SPLIT
# =============================================================================
# Standard k-fold cross-validation would leak future information into training
# because our features include lagged HHI values. We use an expanding-window
# time series split instead: always train on the past, test on the future.

years <- sort(unique(df$year))
cat("\n=== Available years ===\n")
print(years)

folds <- list(
  list(train_end = 2010, test_start = 2011, test_end = 2013),
  list(train_end = 2013, test_start = 2014, test_end = 2016),
  list(train_end = 2016, test_start = 2017, test_end = 2019),
  list(train_end = 2019, test_start = 2020, test_end = 2022)
)

cat("\n=== Fold structure ===\n")
for (i in seq_along(folds)) {
  f <- folds[[i]]
  n_train <- nrow(df %>% filter(year <= f$train_end))
  n_test  <- nrow(df %>% filter(year >= f$test_start & year <= f$test_end))
  cat(sprintf("Fold %d : Train 2000-%d (%d obs) | Test %d-%d (%d obs)\n",
              i, f$train_end, n_train, f$test_start, f$test_end, n_test))
}

# Containers for results
results_logit <- list()
results_enet  <- list()
results_rf    <- list()


# =============================================================================
# 3. MODEL 1 — LOGISTIC REGRESSION (BASELINE)
# =============================================================================
# No regularization. Serves as the lower-bound reference.
# Class imbalance (~4% positives) is handled via inverse frequency weights.
# Note: the "non-integer #successes" warning is expected when using
# non-integer weights in glm() — it has no impact on results.

cat("\n", strrep("=", 60), "\n")
cat("MODEL 1 — Logistic Regression (Baseline)\n")
cat(strrep("=", 60), "\n")

for (i in seq_along(folds)) {
  f <- folds[[i]]
  
  train <- df %>% filter(year <= f$train_end)
  test  <- df %>% filter(year >= f$test_start & year <= f$test_end)
  
  # Inverse frequency weights to upweight the minority class
  ratio      <- sum(train[[target]] == "Non_Critique") / sum(train[[target]] == "Critique")
  sample_wts <- ifelse(train[[target]] == "Critique", ratio, 1)
  
  set.seed(42)
  logit_model <- glm(
    formula = as.formula(paste(target, "~", paste(features, collapse = " + "))),
    data    = train,
    family  = binomial(link = "logit"),
    weights = sample_wts
  )
  
  y_pred_proba <- predict(logit_model, test, type = "response")
  y_pred       <- factor(ifelse(y_pred_proba > 0.5, "Critique", "Non_Critique"),
                         levels = c("Non_Critique", "Critique"))
  y_test       <- test[[target]]
  
  auc_val <- roc(y_test, y_pred_proba, quiet = TRUE)$auc
  cm      <- confusionMatrix(y_pred, y_test, positive = "Critique")
  
  cat(sprintf("Fold %d — AUC: %.3f | Recall: %.3f | F1: %.3f\n",
              i, as.numeric(auc_val),
              cm$byClass["Recall"],
              cm$byClass["F1"]))
  
  results_logit[[i]] <- list(
    fold      = i,
    auc       = as.numeric(auc_val),
    recall    = as.numeric(cm$byClass["Recall"]),
    precision = as.numeric(cm$byClass["Precision"]),
    f1        = as.numeric(cm$byClass["F1"])
  )
}

auc_logit <- sapply(results_logit, function(r) r$auc)
cat(sprintf("\nLogit — Mean AUC: %.3f (sd = %.3f)\n", mean(auc_logit), sd(auc_logit)))


# =============================================================================
# 4. MODEL 2 — ELASTIC NET (alpha = 0.5)
# =============================================================================
# Combines L1 (Lasso) and L2 (Ridge) penalties.
# Alpha = 0.5 gives equal weight to both. Lambda is chosen by 5-fold CV
# within each training fold, maximising AUC.
# The near-zero lambda values obtained (~7e-5) confirm that the logistic
# solution is already well-specified — regularization adds little here.

cat("\n", strrep("=", 60), "\n")
cat("MODEL 2 — Elastic Net (alpha = 0.5)\n")
cat(strrep("=", 60), "\n")

for (i in seq_along(folds)) {
  f <- folds[[i]]
  
  train <- df %>% filter(year <= f$train_end)
  test  <- df %>% filter(year >= f$test_start & year <= f$test_end)
  
  # glmnet requires a design matrix (no formula interface)
  X_train <- model.matrix(~ . - 1, data = train[, features])
  X_test  <- model.matrix(~ . - 1, data = test[, features])
  y_train <- as.numeric(train[[target]] == "Critique")
  y_test  <- test[[target]]
  
  # Inverse frequency weights
  ratio_enet <- sum(y_train == 0) / sum(y_train == 1)
  wts        <- ifelse(y_train == 1, ratio_enet, 1)
  
  # Lambda selection by internal CV
  set.seed(42)
  cv_enet <- cv.glmnet(
    x            = X_train,
    y            = y_train,
    alpha        = 0.5,
    family       = "binomial",
    weights      = wts,
    nfolds       = 5,
    type.measure = "auc"
  )
  
  lambda_opt   <- cv_enet$lambda.min
  y_pred_proba <- as.vector(predict(cv_enet, X_test, s = lambda_opt, type = "response"))
  y_pred       <- factor(ifelse(y_pred_proba > 0.5, "Critique", "Non_Critique"),
                         levels = c("Non_Critique", "Critique"))
  
  auc_val <- roc(y_test, y_pred_proba, quiet = TRUE)$auc
  cm      <- confusionMatrix(y_pred, y_test, positive = "Critique")
  
  cat(sprintf("Fold %d — AUC: %.3f | Recall: %.3f | F1: %.3f | Lambda: %.5f\n",
              i, as.numeric(auc_val),
              cm$byClass["Recall"],
              cm$byClass["F1"],
              lambda_opt))
  
  results_enet[[i]] <- list(
    fold       = i,
    auc        = as.numeric(auc_val),
    recall     = as.numeric(cm$byClass["Recall"]),
    precision  = as.numeric(cm$byClass["Precision"]),
    f1         = as.numeric(cm$byClass["F1"]),
    lambda_opt = lambda_opt
  )
}

auc_enet <- sapply(results_enet, function(r) r$auc)
cat(sprintf("\nElastic Net — Mean AUC: %.3f (sd = %.3f)\n", mean(auc_enet), sd(auc_enet)))


# =============================================================================
# 5. MODEL 3 — RANDOM FOREST
# =============================================================================
# Non-linear ensemble method. Can capture interactions between features
# without explicit specification (e.g. high HHI + few exporters).
# Key hyperparameters:
#   ntree    = 300 (folds) / 500 (final model)
#   mtry     = floor(sqrt(15)) = 3  — standard for classification
#   nodesize = 10  — minimum leaf size, avoids overfitting on minority class
#   classwt  — inverse frequency weights (same logic as models 1 and 2)

cat("\n", strrep("=", 60), "\n")
cat("MODEL 3 — Random Forest\n")
cat(strrep("=", 60), "\n")

for (i in seq_along(folds)) {
  f <- folds[[i]]
  
  train <- df %>% filter(year <= f$train_end)
  test  <- df %>% filter(year >= f$test_start & year <= f$test_end)
  
  X_train <- train[, features]
  y_train <- train[[target]]
  X_test  <- test[, features]
  y_test  <- test[[target]]
  
  cat(sprintf("Fold %d — Train: %d obs | Test: %d obs\n", i, nrow(train), nrow(test)))
  
  class_counts  <- table(y_train)
  class_weights <- max(class_counts) / class_counts
  
  set.seed(42)
  rf_model <- randomForest(
    x          = X_train,
    y          = y_train,
    ntree      = 300,
    mtry       = floor(sqrt(length(features))),
    nodesize   = 10,
    classwt    = class_weights,
    importance = TRUE
  )
  
  y_pred       <- predict(rf_model, X_test)
  y_pred_proba <- predict(rf_model, X_test, type = "prob")[, "Critique"]
  
  auc_val <- roc(y_test, y_pred_proba, quiet = TRUE)$auc
  cm      <- confusionMatrix(y_pred, y_test, positive = "Critique")
  
  cat(sprintf("         AUC: %.3f | Recall: %.3f | F1: %.3f\n",
              as.numeric(auc_val),
              cm$byClass["Recall"],
              cm$byClass["F1"]))
  
  results_rf[[i]] <- list(
    fold      = i,
    auc       = as.numeric(auc_val),
    recall    = as.numeric(cm$byClass["Recall"]),
    precision = as.numeric(cm$byClass["Precision"]),
    f1        = as.numeric(cm$byClass["F1"]),
    model     = rf_model,
    cm        = cm
  )
}

auc_rf <- sapply(results_rf, function(r) r$auc)
cat(sprintf("\nRandom Forest — Mean AUC: %.3f (sd = %.3f)\n", mean(auc_rf), sd(auc_rf)))


# =============================================================================
# 6. MODEL COMPARISON
# =============================================================================

cat("\n", strrep("=", 60), "\n")
cat("MODEL COMPARISON (average across 4 folds)\n")
cat(strrep("=", 60), "\n")

comparison_df <- data.frame(
  Model       = c("Logit (Baseline)", "Elastic Net", "Random Forest"),
  AUC_mean    = c(mean(sapply(results_logit, function(r) r$auc)),
                  mean(sapply(results_enet,  function(r) r$auc)),
                  mean(sapply(results_rf,    function(r) r$auc))),
  AUC_sd      = c(sd(sapply(results_logit, function(r) r$auc)),
                  sd(sapply(results_enet,  function(r) r$auc)),
                  sd(sapply(results_rf,    function(r) r$auc))),
  Recall_mean = c(mean(sapply(results_logit, function(r) r$recall), na.rm = TRUE),
                  mean(sapply(results_enet,  function(r) r$recall), na.rm = TRUE),
                  mean(sapply(results_rf,    function(r) r$recall), na.rm = TRUE)),
  F1_mean     = c(mean(sapply(results_logit, function(r) r$f1), na.rm = TRUE),
                  mean(sapply(results_enet,  function(r) r$f1), na.rm = TRUE),
                  mean(sapply(results_rf,    function(r) r$f1), na.rm = TRUE))
)

print(comparison_df)
write.csv(comparison_df, paste0(OUTPUT_DIR, "model_comparison.csv"), row.names = FALSE)
cat("\nSaved: model_comparison.csv\n")


# =============================================================================
# 7. FINAL MODEL — Random Forest trained on 2000–2021, tested on 2022–2023
# =============================================================================
# We retrain the Random Forest on the maximum available data and evaluate
# on a held-out period that was never seen during cross-validation.

cat("\n", strrep("=", 60), "\n")
cat("FINAL MODEL (Random Forest) — Holdout test set 2022–2023\n")
cat(strrep("=", 60), "\n")

train_final <- df %>% filter(year <= 2021)
test_final  <- df %>% filter(year > 2021)

X_train_final <- train_final[, features]
y_train_final <- train_final[[target]]
X_test_final  <- test_final[, features]
y_test_final  <- test_final[[target]]

cat(sprintf("Training set : %d obs | Test set : %d obs\n",
            nrow(train_final), nrow(test_final)))

class_counts_final  <- table(y_train_final)
class_weights_final <- max(class_counts_final) / class_counts_final

set.seed(42)
rf_final <- randomForest(
  x          = X_train_final,
  y          = y_train_final,
  ntree      = 500,
  mtry       = floor(sqrt(length(features))),
  nodesize   = 10,
  classwt    = class_weights_final,
  importance = TRUE
)

y_pred_final       <- predict(rf_final, X_test_final)
y_pred_proba_final <- predict(rf_final, X_test_final, type = "prob")[, "Critique"]

auc_final <- roc(y_test_final, y_pred_proba_final, quiet = TRUE)$auc
cm_final  <- confusionMatrix(y_pred_final, y_test_final, positive = "Critique")

cat(sprintf("AUC-ROC (final model) : %.3f\n", as.numeric(auc_final)))
print(cm_final)


# =============================================================================
# 8. FIGURES — exported for the report
# =============================================================================

# --- Figure 1: AUC comparison across models ---
p_compare <- ggplot(comparison_df,
                    aes(x = reorder(Model, AUC_mean), y = AUC_mean, fill = Model)) +
  geom_col(alpha = 0.85, width = 0.6) +
  geom_errorbar(aes(ymin = AUC_mean - AUC_sd, ymax = AUC_mean + AUC_sd),
                width = 0.2, linewidth = 0.8) +
  geom_text(aes(label = sprintf("%.3f", AUC_mean)),
            vjust = -0.5, size = 4.5, fontface = "bold") +
  scale_fill_manual(values = c("Logit (Baseline)" = "#A8C8E8",
                               "Elastic Net"       = "#5B9BD5",
                               "Random Forest"     = "#1F4E79")) +
  ylim(0, 1) +
  labs(title    = "Model comparison — Mean AUC-ROC (4 temporal folds)",
       subtitle = "Error bars = ± 1 standard deviation across folds",
       x = NULL, y = "AUC-ROC") +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none")

ggsave(paste0(OUTPUT_DIR, "fig_model_comparison.png"),
       p_compare, width = 8, height = 5, dpi = 150)

# --- Figure 2: AUC by fold for each model ---
fold_df <- data.frame(
  Fold  = rep(paste0("Fold ", 1:4), 3),
  Model = rep(c("Logit (Baseline)", "Elastic Net", "Random Forest"), each = 4),
  AUC   = c(sapply(results_logit, function(r) r$auc),
            sapply(results_enet,  function(r) r$auc),
            sapply(results_rf,    function(r) r$auc))
)

p_folds <- ggplot(fold_df, aes(x = Fold, y = AUC, color = Model, group = Model)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3) +
  scale_color_manual(values = c("Logit (Baseline)" = "#A8C8E8",
                                "Elastic Net"       = "#5B9BD5",
                                "Random Forest"     = "#1F4E79")) +
  ylim(0.5, 1) +
  labs(title = "AUC-ROC by temporal fold — all three models",
       x = NULL, y = "AUC-ROC", color = "Model") +
  theme_minimal(base_size = 13)

ggsave(paste0(OUTPUT_DIR, "fig_auc_by_fold.png"),
       p_folds, width = 9, height = 5, dpi = 150)

# --- Figure 3: Feature importance (final RF) ---
importance_df <- as.data.frame(importance(rf_final)) %>%
  rownames_to_column("feature") %>%
  arrange(desc(MeanDecreaseGini))

p_importance <- ggplot(importance_df[1:15, ],
                       aes(x = reorder(feature, MeanDecreaseGini),
                           y = MeanDecreaseGini)) +
  geom_col(fill = "#1F4E79") +
  coord_flip() +
  labs(title    = "Feature Importance — Random Forest (final model)",
       subtitle = "Mean Decrease Gini — top 15 features",
       x = NULL, y = "Mean Decrease Gini") +
  theme_minimal(base_size = 13)

ggsave(paste0(OUTPUT_DIR, "fig_feature_importance.png"),
       p_importance, width = 8, height = 6, dpi = 150)

# --- Figure 4: ROC curve (final model, holdout test set) ---
roc_obj <- roc(y_test_final, y_pred_proba_final, quiet = TRUE)

png(paste0(OUTPUT_DIR, "fig_roc_curve.png"), width = 700, height = 600)
plot(roc_obj,
     main = sprintf("ROC Curve — Random Forest final model (AUC = %.3f)",
                    as.numeric(auc_final)),
     col = "#1F4E79", lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "gray60")
dev.off()

# --- Figure 5: Confusion matrix (final model, holdout test set) ---
cm_table <- as.data.frame(cm_final$table)
names(cm_table) <- c("Prediction", "Reference", "Freq")

p_cm <- ggplot(cm_table, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 8, fontface = "bold") +
  scale_fill_gradient(low = "#D6E4F0", high = "#1F4E79") +
  labs(title = "Confusion Matrix — Random Forest (test set 2022–2023)",
       x = "Actual", y = "Predicted") +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none")

ggsave(paste0(OUTPUT_DIR, "fig_confusion_matrix.png"),
       p_cm, width = 6, height = 5, dpi = 150)

cat("\nAll figures saved to:", OUTPUT_DIR, "\n")


# =============================================================================
# 9. SAVING RESULTS
# =============================================================================

saveRDS(rf_final, paste0(OUTPUT_DIR, "rf_final_model.rds"))

all_results <- list(
  logit      = results_logit,
  enet       = results_enet,
  rf         = results_rf,
  comparison = comparison_df
)
saveRDS(all_results, paste0(OUTPUT_DIR, "all_model_results.rds"))

cat("\n=== FINAL SUMMARY ===\n")
print(comparison_df)
cat(sprintf("\nFinal model AUC (holdout 2022-2023): %.3f\n", as.numeric(auc_final)))
cat("\nDone.\n")