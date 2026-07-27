args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("Usage: Rscript run.R DATA_PATH OUTPUT_PATH SAMPLE_PATH")
}

data_path <- args[[1]]
output_path <- args[[2]]
sample_path <- args[[3]]
data <- read.csv(data_path, check.names = FALSE)
required_columns <- c("y", "x1", "x2")
if (!all(required_columns %in% names(data))) {
  stop("Validation data is missing one or more required columns")
}

normalize_model <- function(case_id, formula, include_constant) {
  model <- lm(formula, data = data)
  model_summary <- summary(model)
  coefficients <- model_summary$coefficients
  if (include_constant) {
    coefficient_order <- c("x1", "x2", "(Intercept)")
  } else {
    coefficient_order <- c("x1", "x2")
  }
  coefficients <- coefficients[coefficient_order, , drop = FALSE]
  omitted <- is.na(coef(model)[coefficient_order])
  term_names <- rownames(coefficients)
  term_names[term_names == "(Intercept)"] <- "_cons"

  residual_ss <- sum(residuals(model)^2)
  if (include_constant) {
    total_ss <- sum((data$y - mean(data$y))^2)
  } else {
    total_ss <- sum(data$y^2)
  }
  model_ss <- total_ss - residual_ss
  f_stat <- unname(model_summary$fstatistic[["value"]])
  df_model <- as.integer(model_summary$fstatistic[["numdf"]])
  df_resid <- as.integer(model_summary$fstatistic[["dendf"]])
  f_p_value <- pf(f_stat, df_model, df_resid, lower.tail = FALSE)

  result <- data.frame(
    case_id = case_id,
    engine_version = R.version.string,
    n_obs = nobs(model),
    df_model = df_model,
    df_resid = df_resid,
    term = term_names,
    estimate = coefficients[, "Estimate"],
    std_error = coefficients[, "Std. Error"],
    statistic = coefficients[, "t value"],
    p_value = coefficients[, "Pr(>|t|)"],
    r_squared = model_summary$r.squared,
    r_squared_adj = model_summary$adj.r.squared,
    f_stat = f_stat,
    f_p_value = f_p_value,
    ss_residual = residual_ss,
    ss_model = model_ss,
    ss_total = total_ss,
    root_mse = model_summary$sigma,
    omitted = as.integer(omitted),
    check.names = FALSE
  )
  samples <- data.frame(
    case_id = case_id,
    original_row_number = as.integer(rownames(model.frame(model))),
    check.names = FALSE
  )
  list(result = result, samples = samples)
}

basic <- normalize_model("basic", y ~ x1 + x2, TRUE)
no_constant <- normalize_model("no_constant", y ~ x1 + x2 - 1, FALSE)
results <- rbind(basic$result, no_constant$result)
samples <- rbind(basic$samples, no_constant$samples)
write.table(
  results,
  file = output_path,
  sep = "\t",
  row.names = FALSE,
  col.names = TRUE,
  quote = FALSE,
  na = ""
)
write.table(
  samples,
  file = sample_path,
  sep = "\t",
  row.names = FALSE,
  col.names = TRUE,
  quote = FALSE,
  na = ""
)
