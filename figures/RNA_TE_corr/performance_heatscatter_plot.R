library(tidyverse)
library(LSD)
library(cowplot)

predictions <- read.csv("./results/human/RNA_TE_corr/lgbm-LL_P5_P3_CF_AAF_3mer_freq_5/predictions.csv", 
  header = TRUE, sep = ","
)


predictions <- predictions %>% 
  rename(ObservedCorr = bio_source_TE_RNA_cor_value_nond_true, PredictedCorr = bio_source_TE_RNA_cor_value_nond_pred)


pdf("./figures/RNA_TE_corr/predicted_vs_observed_RNA_TE_corr_human.pdf", 5, 5) 
# 'predictions' below is a dataframe with at least two columns named PredictedCorr and ObservedCorr
predictions %>% 
  mutate(
    r2 = cor(PredictedCorr, ObservedCorr)^2,
    r = cor(PredictedCorr, ObservedCorr),
    rho = cor(PredictedCorr, ObservedCorr, method = "spearman"),
    ) %>%
  {
    heatscatter(
      .$PredictedCorr, .$ObservedCorr,
      xlab = "CV fitted model prediction (human)",
      xlim = c(-0.8, 0.6),
      ylim = c(-0.8, 0.6),
      ylab = "RNA_TE_corr (human)", bty='n', cex=0.3, las=1,
      main = str_glue("r2 = {.$r2[1] %>% round(3)}, r = {.$r[1] %>% round(3)}, rho = {.$rho[1] %>% round(3)}")
    )
  }
dev.off()

predictions <- read.csv("./results/mouse/RNA_TE_corr/lgbm-LL_P5_P3_CF_AAF_3mer_freq_5/predictions.csv", 
  header = TRUE, sep = ","
)


predictions <- predictions %>% 
  rename(ObservedCorr = bio_source_TE_RNA_cor_value_nond_true, PredictedCorr = bio_source_TE_RNA_cor_value_nond_pred)


pdf("./figures/RNA_TE_corr/predicted_vs_observed_RNA_TE_corr_mouse.pdf", 5, 5) 
# 'predictions' below is a dataframe with at least two columns named PredictedCorr and ObservedCorr
predictions %>% 
  mutate(
    r2 = cor(PredictedCorr, ObservedCorr)^2,
    r = cor(PredictedCorr, ObservedCorr),
    rho = cor(PredictedCorr, ObservedCorr, method = "spearman"),
    ) %>%
  {
    heatscatter(
      .$PredictedCorr, .$ObservedCorr,
      xlab = "CV fitted model prediction (mouse)",
      xlim = c(-0.8, 0.6),
      ylim = c(-0.8, 0.6),
      ylab = "RNA_TE_corr (mouse)", bty='n', cex=0.3, las=1,
      main = str_glue("r2 = {.$r2[1] %>% round(3)}, r = {.$r[1] %>% round(3)}, rho = {.$rho[1] %>% round(3)}")
    )
  }
dev.off()