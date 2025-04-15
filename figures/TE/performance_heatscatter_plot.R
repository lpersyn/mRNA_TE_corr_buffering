library(tidyverse)
library(LSD)
library(cowplot)

predictions <- read.csv("./results/human/TE/lgbm-LL_P5_P3_CF_AAF_3mer_freq_5/predictions.csv", 
  header = TRUE, sep = ","
)


predictions <- predictions %>% 
  rename(ObservedTE = mean_te_true, PredictedTE = mean_te_pred)


pdf("./figures/TE/predicted_vs_observed_TE_human.pdf", 5, 5) 
# 'predictions' below is a dataframe with at least two columns named PredictedTE and ObservedTE
predictions %>% 
  mutate(
    r2 = cor(PredictedTE, ObservedTE)^2,
    r = cor(PredictedTE, ObservedTE),
    rho = cor(PredictedTE, ObservedTE, method = "spearman"),
    ) %>%
  {
    heatscatter(
      .$PredictedTE, .$ObservedTE,
      xlab = "CV fitted model prediction (human)",
      xlim = c(-3, 3),
      ylim = c(-3,3),
      ylab = "TE (human)", bty='n', cex=0.3, las=1,
      main = str_glue("r2 = {.$r2[1] %>% round(3)}, r = {.$r[1] %>% round(3)}, rho = {.$rho[1] %>% round(3)}")
    )
  }
dev.off()

predictions <- read.csv("./results/mouse/TE/lgbm-LL_P5_P3_CF_AAF_3mer_freq_5/predictions.csv", 
  header = TRUE, sep = ","
)


predictions <- predictions %>% 
  rename(ObservedTE = mean_te_true, PredictedTE = mean_te_pred)


pdf("./figures/TE/predicted_vs_observed_TE_mouse.pdf", 5, 5) 
# 'predictions' below is a dataframe with at least two columns named PredictedTE and ObservedTE
predictions %>% 
  mutate(
    r2 = cor(PredictedTE, ObservedTE)^2,
    r = cor(PredictedTE, ObservedTE),
    rho = cor(PredictedTE, ObservedTE, method = "spearman"),
    ) %>%
  {
    heatscatter(
      .$PredictedTE, .$ObservedTE,
      xlab = "CV fitted model prediction (mouse)",
      xlim = c(-3, 3),
      ylim = c(-3,3),
      ylab = "TE (mouse)", bty='n', cex=0.3, las=1,
      main = str_glue("r2 = {.$r2[1] %>% round(3)}, r = {.$r[1] %>% round(3)}, rho = {.$rho[1] %>% round(3)}")
    )
  }
dev.off()