#------------------------------------------------------------------------------#
#          Intro to Statistical Learning — Final Project
#          Data Processing Script
#------------------------------------------------------------------------------#
# Author : Jules Tastet
# Date   : February–March 2026
# School : École Polytechnique, M1 Data & Economics for Public Policy
#
# This script builds the ML-ready dataset from raw BACI trade data and two
# supplementary sources (WTO world trade values, World Bank Pink Sheet).
#
# INPUT FILES (all expected in RAW_DIR):
#   - BACI_HS96_Y[year]_V202601.csv  for years 1996 to 2024
#   - world_trade_values.xlsx                      WTO world trade values
#   - CMO-Historical-Data-Monthly.xlsx  World Bank Pink Sheet
#------------------------------------------------------------------------------#

# NOTE : YOU CAN DOWNLOAD THE DATA HERE 
# BACI Data : https://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=37
# WTO Data : https://www.wto.org/english/res_e/statis_e/trade_evolution_e/world_trade_values.xlsx
# Pink Sheet (NOT USED IN THE END) : https://thedocs.worldbank.org/en/doc/18675f1d1639c7a34d463f59263ba0a2-0050012025/

# =============================================================================
# 0. SETUP
# =============================================================================

# Uncomment to install on first run:
# install.packages(c("tidyverse", "arrow", "readxl", "zoo"))

library(tidyverse)
library(arrow)
library(readxl)
library(zoo)

rm(list = ls())

# -----------------------------------------------------------------------------
# !! IMPORTANT — adapt these two paths before running !!
# RAW_DIR       must contain the raw BACI files and supplementary Excel files
# PROCESSED_DIR will receive all intermediate and final outputs
# -----------------------------------------------------------------------------
RAW_DIR       <- "/Users/ju/Desktop/FINAL_PROJECT_ML/data_raw/"
PROCESSED_DIR <- "/Users/ju/Desktop/FINAL_PROJECT_ML/data_processed/"

# France's BACI country code
FRANCE_CODE <- 251


# =============================================================================
# 1. BACI — LOAD, FILTER TO FRANCE, STACK
# =============================================================================
# Each annual BACI file is loaded, columns are renamed to human-readable names,
# and the dataset is filtered to observations where France is the importer.
# Processed annual files are saved as parquet for efficiency, then stacked.

years <- 1996:2024

cat("=== Processing BACI annual files ===\n")

for (yr in years) {
  file_in  <- paste0(RAW_DIR, "BACI_HS96_Y", yr, "_V202601.csv")
  file_out <- paste0(PROCESSED_DIR, "baci_fr_", yr, ".parquet")
  
  data_yr <- read_csv(file_in, show_col_types = FALSE) %>%
    rename(
      importer      = j,
      exporter      = i,
      transac_value = v,
      quantity      = q,
      year          = t,
      product       = k
    ) %>%
    filter(importer == FRANCE_CODE)
  
  write_parquet(data_yr, file_out)
  cat(sprintf("  %d — %d rows saved\n", yr, nrow(data_yr)))
}

# Stack all annual files into a single panel
cat("\n=== Stacking annual files ===\n")
ds         <- open_dataset(PROCESSED_DIR, format = "parquet")
data_final <- collect(ds)

cat(sprintf("Raw stacked panel: %d rows, %d columns\n", nrow(data_final), ncol(data_final)))
write_parquet(data_final, paste0(PROCESSED_DIR, "data_final.parquet"))


# =============================================================================
# 2. CLEANING & FEATURE ENGINEERING
# =============================================================================

# --- 2.1 Missing values ---
# Rows with missing quantity are dropped. Inspection confirmed these are
# missing at random with no systematic pattern across products or years.
n_before <- nrow(data_final)
data_final <- drop_na(data_final)
cat(sprintf("Dropped %d rows with missing values (%d remaining)\n",
            n_before - nrow(data_final), nrow(data_final)))

# --- 2.2 Aggregate to product × year ---
# For each product-year cell, we compute HHI, market shares, and other
# market structure indicators from the raw bilateral trade flows.
cat("\n=== Aggregating to product × year ===\n")

data_agg <- data_final %>%
  group_by(product, year) %>%
  summarise(
    # Volume
    total_import     = sum(transac_value, na.rm = TRUE),
    total_qty        = sum(quantity, na.rm = TRUE),
    Nb_exporters     = n_distinct(exporter),
    log_total_import = log1p(sum(transac_value, na.rm = TRUE)),
    
    # Top exporter
    top_exporter     = exporter[which.max(transac_value)],
    max_export_value = max(transac_value, na.rm = TRUE),
    
    # Market shares (value-based)
    share_top_exporter   = max(transac_value, na.rm = TRUE) /
      sum(transac_value, na.rm = TRUE),
    
    share_top3_exporters = {
      s <- transac_value / sum(transac_value, na.rm = TRUE)
      sum(sort(s, decreasing = TRUE)[1:min(3, length(s))], na.rm = TRUE)
    },
    
    # HHI — Herfindahl-Hirschman Index
    hhi = {
      s <- transac_value / sum(transac_value, na.rm = TRUE)
      sum(s^2, na.rm = TRUE)
    },
    
    # Average unit cost (log-transformed later to handle extreme outliers)
    avg_unit_cost = sum(transac_value, na.rm = TRUE) /
      sum(quantity, na.rm = TRUE),
    
    .groups = "drop"
  ) %>%
  mutate(
    # HS2 sector code — first 2 digits of HS6 product code
    hs2 = floor(as.numeric(product) / 10000),
    
    # Crisis dummy: 2008-09 financial crisis + COVID + Ukraine war shocks
    crisis_year = ifelse(year %in% c(2008, 2009, 2020, 2021, 2022), 1, 0),
    
    # Log unit cost — handles extreme products (rhodium, diamonds, hormones)
    log_avg_unit_cost = log1p(avg_unit_cost)
  )

# --- 2.3 Lags, rolling averages, and target variables ---
# Sort by product and year before computing lags to avoid incorrect ordering.
# Losing the first 3 years per product is expected and not treated as data loss.
cat("=== Computing lags and target variables ===\n")

data_agg <- data_agg %>%
  arrange(product, year) %>%
  group_by(product) %>%
  mutate(
    # HHI lags — core predictors (autocorrelation ≈ 0.83 at lag 1)
    hhi_lag1 = lag(hhi, 1),
    hhi_lag2 = lag(hhi, 2),
    hhi_lag3 = lag(hhi, 3),
    
    # Annual change in HHI — short-run momentum
    delta_hhi = hhi - lag(hhi, 1),
    
    # 5-year rolling average — structural trend vs. short-term noise
    hhi_trend_5y = zoo::rollmean(hhi, k = 5, fill = NA, align = "right"),
    
    # Lagged market structure variables
    share_top_lag1  = lag(share_top_exporter, 1),
    log_import_lag1 = lag(log_total_import, 1),
    
    # Target: will the market become critically concentrated next year?
    # Threshold: HHI > 0.25 (EU/DoJ standard definition of high concentration)
    critical_t1 = ifelse(lead(hhi, 1) > 0.25, 1, 0),  # 1-year horizon
    critical_t3 = ifelse(lead(hhi, 3) > 0.25, 1, 0)   # 3-year horizon
  ) %>%
  ungroup()

# ML-ready subset: drop rows where lags or 1-year target are missing
data_ml <- data_agg %>%
  filter(!is.na(hhi_lag3), !is.na(critical_t1))

cat(sprintf("ML-ready dataset: %d rows\n", nrow(data_ml)))
cat(sprintf("Critical cases (critical_t1 = 1): %d (%.1f%%)\n",
            sum(data_ml$critical_t1 == 1),
            mean(data_ml$critical_t1 == 1) * 100))


# =============================================================================
# 3. WTO WORLD TRADE VALUES
# =============================================================================
# Annual world merchandise trade values from the WTO are merged by year
# to serve as a macroeconomic control for global trade conditions.

cat("\n=== Merging WTO world trade data ===\n")

wto <- read_excel(paste0(RAW_DIR, "world_trade_values.xlsx"),
                  col_names = c("year", "world_trade_value")) %>%
  filter(year >= 1996) %>%
  mutate(
    year            = as.integer(year),
    log_world_trade = log(world_trade_value)
  )

data_ml <- data_ml %>%
  left_join(wto, by = "year")

cat(sprintf("Missing world_trade_value after merge: %d\n",
            sum(is.na(data_ml$world_trade_value))))


# =============================================================================
# 4. WORLD BANK PINK SHEET — COMMODITY PRICES BY HS2
# =============================================================================
# Monthly commodity prices are aggregated to annual averages, then matched
# to HS2 sectors via a hand-coded concordance table.
# Note: only 29 out of 97 HS2 sectors have a commodity price match.
# This variable is ultimately excluded from the ML models due to ~75% missing
# values — documented here for transparency and potential future use.

cat("\n=== Merging World Bank Pink Sheet commodity prices ===\n")

# Load and clean monthly price data
pink_raw <- read_excel(
  paste0(RAW_DIR, "CMO-Historical-Data-Monthly.xlsx"),
  sheet = "Monthly Prices",
  skip  = 4
) %>%
  rename(date = `...1`) %>%
  filter(!is.na(date), grepl("M", as.character(date))) %>%
  mutate(
    year = as.integer(substr(date, 1, 4)),
    across(-c(date, year), ~ as.numeric(ifelse(. %in% c("…", "..."), NA, .)))
  ) %>%
  filter(year >= 1996)

# Annual averages
pink_annual <- pink_raw %>%
  group_by(year) %>%
  summarise(across(-date, ~ mean(., na.rm = TRUE)))

# HS2 → commodity concordance (hand-coded)
hs2_map <- tribble(
  ~hs2, ~commodity_col,
  1,  "Beef **",
  2,  "Beef **",
  3,  "Fish meal",
  8,  "Orange",
  9,  "Coffee, Arabica",
  10, "Wheat, US HRW",
  11, "Wheat, US HRW",
  12, "Soybeans",
  15, "Palm oil",
  17, "Sugar, world",
  18, "Cocoa",
  23, "Soybean meal",
  24, "Tobacco, US import u.v.",
  25, "Phosphate rock",
  26, "Iron ore, cfr spot",
  27, "Crude oil, average",
  31, "Urea ",
  40, "Rubber, TSR20 **",
  44, "Logs, Malaysian",
  47, "Logs, Malaysian",
  52, "Cotton, A Index",
  71, "Gold",
  72, "Iron ore, cfr spot",
  74, "Copper",
  75, "Nickel",
  76, "Aluminum",
  78, "Lead",
  79, "Zinc",
  80, "Tin"
)

# Check for missing columns in Pink Sheet
missing_cols <- setdiff(unique(hs2_map$commodity_col), names(pink_annual))
if (length(missing_cols) > 0) {
  cat("WARNING — commodity columns not found in Pink Sheet:\n")
  print(missing_cols)
}

# Build hs2 × year price table
pink_long <- pink_annual %>%
  select(year, any_of(unique(hs2_map$commodity_col))) %>%
  pivot_longer(-year, names_to = "commodity_col", values_to = "commodity_price")

hs2_prices <- hs2_map %>%
  filter(commodity_col %in% names(pink_annual)) %>%
  left_join(pink_long, by = "commodity_col") %>%
  mutate(hs2 = as.integer(hs2), year = as.integer(year)) %>%
  select(hs2, year, commodity_price)

# Merge into data_ml
data_ml <- data_ml %>%
  mutate(hs2 = as.integer(hs2), year = as.integer(year)) %>%
  left_join(hs2_prices, by = c("hs2", "year")) %>%
  mutate(log_commodity_price = log1p(commodity_price))

coverage <- round(mean(!is.na(data_ml$commodity_price)) * 100, 1)
cat(sprintf("Commodity price coverage: %.1f%% of rows\n", coverage))
cat("(Remaining %.1f%% have no commodity match — variable excluded from ML models)\n",
    100 - coverage)


# =============================================================================
# 5. SAVE FINAL ML-READY DATASET
# =============================================================================

write.csv(data_ml,
          paste0(PROCESSED_DIR, "data_ml_ready_v2.csv"),
          row.names = FALSE)

cat("\n=== Done ===\n")
cat(sprintf("Final dataset: %d rows × %d columns\n", nrow(data_ml), ncol(data_ml)))
cat(sprintf("Saved to: %sdata_ml_ready_v2.csv\n", PROCESSED_DIR))
cat("\nThis file is the direct input to scripts/02_modeling.R\n")