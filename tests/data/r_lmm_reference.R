#!/usr/bin/env Rscript
# lmerTest Satterthwaite + Kenward-Roger reference values for the
# validation datasets. Output: JSON per dataset.
args <- commandArgs(trailingOnly = TRUE)
dir <- args[1]; out <- args[2]
suppressPackageStartupMessages({library(lmerTest); library(pbkrtest)})
files <- list.files(dir, pattern = "\\.csv$", full.names = TRUE)
res <- list()
for (f in files) {
  dat <- read.csv(f)
  fit <- lmerTest::lmer(y ~ treatment + (1 | cluster), data = dat)
  vc <- as.data.frame(lme4::VarCorr(fit))
  sat <- coef(summary(fit, ddf = "Satterthwaite"))["treatment", ]
  kr <- coef(summary(fit, ddf = "Kenward-Roger"))["treatment", ]
  res[[tools::file_path_sans_ext(basename(f))]] <- list(
    tau2 = vc$vcov[1], sigma2 = vc$vcov[2],
    estimate = unname(sat["Estimate"]),
    sat = list(se = unname(sat["Std. Error"]), df = unname(sat["df"]),
               p = unname(sat["Pr(>|t|)"])),
    kr = list(se = unname(kr["Std. Error"]), df = unname(kr["df"]),
              p = unname(kr["Pr(>|t|)"])),
    singular = lme4::isSingular(fit)
  )
}
writeLines(jsonlite::toJSON(res, auto_unbox = TRUE, digits = 12), out)
cat("wrote", out, "\n")
