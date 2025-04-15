library(tidyverse)
library(LSD)
library(cowplot)

predictions <- read.csv("./results/cross_species/human_model_on_mouse_RNA_TE_corr.csv", 
  header = TRUE, sep = "\t"
)


predictions <- predictions %>% 
  rename(ObservedRNA_TE_corr = bio_source_TE_RNA_cor_value_nond_true, PredictedRNA_TE_corr = bio_source_TE_RNA_cor_value_nond_pred)


pdf("./figures/RNA_TE_corr/human_model_on_mouse_RNA_TE_corr.pdf", 5, 5) 
# 'predictions' below is a dataframe with at least two columns named PredictedRNA_TE_corr and ObservedRNA_TE_corr
predictions %>% 
  mutate(
    r2 = cor(PredictedRNA_TE_corr, ObservedRNA_TE_corr)^2,
    r = cor(PredictedRNA_TE_corr, ObservedRNA_TE_corr),
    rho = cor(PredictedRNA_TE_corr, ObservedRNA_TE_corr, method = "spearman"),
    ) %>%
  {
    heatscatter(
      .$PredictedRNA_TE_corr, .$ObservedRNA_TE_corr,
      xlab = "CV fitted model prediction (human)",
      # xlim = c(-3, 3),
      # ylim = c(-3, 3),
      ylab = "RNA_TE_corr (mouse)", bty='n', cex=0.3, las=1,
      main = str_glue("r2 = {.$r2[1] %>% round(3)}, r = {.$r[1] %>% round(3)}, rho = {.$rho[1] %>% round(3)}")
    )
  }
dev.off()

predictions <- read.csv("./results/cross_species/mouse_model_on_human_RNA_TE_corr.csv", 
  header = TRUE, sep = "\t"
)


predictions <- predictions %>% 
  rename(ObservedRNA_TE_corr = bio_source_TE_RNA_cor_value_nond_true, PredictedRNA_TE_corr = bio_source_TE_RNA_cor_value_nond_pred)


pdf("./figures/RNA_TE_corr/mouse_model_on_human_RNA_TE_corr.pdf", 5, 5) 
# 'predictions' below is a dataframe with at least two columns named PredictedRNA_TE_corr and ObservedRNA_TE_corr
predictions %>% 
  mutate(
    r2 = cor(PredictedRNA_TE_corr, ObservedRNA_TE_corr)^2,
    r = cor(PredictedRNA_TE_corr, ObservedRNA_TE_corr),
    rho = cor(PredictedRNA_TE_corr, ObservedRNA_TE_corr, method = "spearman"),
    ) %>%
  {
    heatscatter(
      .$PredictedRNA_TE_corr, .$ObservedRNA_TE_corr,
      xlab = "CV fitted model prediction (mouse)",
      # xlim = c(-3, 3),
      # ylim = c(-3, 3),
      ylab = "RNA_TE_corr (human)", bty='n', cex=0.3, las=1,
      main = str_glue("r2 = {.$r2[1] %>% round(3)}, r = {.$r[1] %>% round(3)}, rho = {.$rho[1] %>% round(3)}")
    )
  }
dev.off()