library(tidyverse)
library(LSD)
library(cowplot)

human_features <- read.csv("./results/human/RNA_TE_corr/lgbm-LL_P5_P3_CF_AAF_3mer_freq_5/feature_importance.csv", 
  header = TRUE, sep = ","
)

mouse_features <- read.csv("./results/mouse/RNA_TE_corr/lgbm-LL_P5_P3_CF_AAF_3mer_freq_5/feature_importance.csv", 
  header = TRUE, sep = ","
)
features <- data.frame(row.names = row.names(human_features))
features['Human'] <- human_features$mean_importance
features['Mouse'] <- mouse_features$mean_importance
features['Human'] <- log10(features['Human'])
features['Mouse'] <- log10(features['Mouse'])
features$Human[features$Human == -Inf] <- 0
features$Mouse[features$Mouse == -Inf] <- 0

pdf("./figures/RNA_TE_corr/human_mouse_feature_imp_corr.pdf", 5, 5) 
features %>% 
  mutate(
    r = cor(Human, Mouse, use = "pairwise.complete.obs", method = "pearson"),
    rho = cor(Human, Mouse, use= "pairwise.complete.obs", method = "spearman"),
    ) %>%
  {
    heatscatter(
      .$Human, .$Mouse,
      # xlim = c(0, 4), ylim = c(0, 4),
      xlab = "Human", 
      ylab = "Mouse", bty='n', las=1,
      main = str_glue("r = {.$r[1] %>% round(3)}, rho = {.$rho[1] %>% round(3)}"),
      cexplot = 1.5,
    )
  }
dev.off()