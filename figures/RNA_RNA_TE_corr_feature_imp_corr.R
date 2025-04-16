library(tidyverse)
library(LSD)
library(cowplot)

te_features <- read.csv("./results/human/RNA_TE_corr/lgbm-LL_P5_P3_CF_AAF_3mer_freq_5/feature_importance.csv", 
  header = TRUE, sep = ","
)

rna_features <- read.csv("./results/human/RNA/lgbm-LL_P5_P3_CF_AAF_3mer_freq_5/feature_importance.csv", 
  header = TRUE, sep = ","
)
features <- data.frame(row.names = row.names(te_features))
features['RNA_TE_corr'] <- te_features$mean_importance
features['RNA'] <- rna_features$mean_importance
features['RNA_TE_corr'] <- log10(features['RNA_TE_corr'])
features['RNA'] <- log10(features['RNA'])
features$RNA_TE_corr[features$RNA_TE_corr == -Inf] <- 0
features$RNA[features$RNA == -Inf] <- 0

pdf("./figures/human_RNA_RNA_TE_corr_feature_imp_corr.pdf", 5, 5) 
features %>% 
  mutate(
    r = cor(RNA_TE_corr, RNA, use = "pairwise.complete.obs", method = "pearson"),
    rho = cor(RNA_TE_corr, RNA, use= "pairwise.complete.obs", method = "spearman"),
    ) %>%
  {
    heatscatter(
      .$RNA_TE_corr, .$RNA,
      # xlim = c(0, 4), ylim = c(0, 4),
      xlab = "RNA_TE_corr", 
      ylab = "RNA", bty='n', las=1,
      main = str_glue("r = {.$r[1] %>% round(3)}, rho = {.$rho[1] %>% round(3)}"),
      cexplot = 1.5,
    )
  }
dev.off()




te_features <- read.csv("./results/mouse/RNA_TE_corr/lgbm-LL_P5_P3_CF_AAF_3mer_freq_5/feature_importance.csv", 
  header = TRUE, sep = ","
)

rna_features <- read.csv("./results/mouse/RNA/lgbm-LL_P5_P3_CF_AAF_3mer_freq_5/feature_importance.csv", 
  header = TRUE, sep = ","
)
features <- data.frame(row.names = row.names(te_features))
features['RNA_TE_corr'] <- te_features$mean_importance
features['RNA'] <- rna_features$mean_importance
features['RNA_TE_corr'] <- log10(features['RNA_TE_corr'])
features['RNA'] <- log10(features['RNA'])
features$RNA_TE_corr[features$RNA_TE_corr == -Inf] <- 0
features$RNA[features$RNA == -Inf] <- 0

pdf("./figures/mouse_RNA_RNA_TE_corr_feature_imp_corr.pdf", 5, 5) 
features %>% 
  mutate(
    r = cor(RNA_TE_corr, RNA, use = "pairwise.complete.obs", method = "pearson"),
    rho = cor(RNA_TE_corr, RNA, use= "pairwise.complete.obs", method = "spearman"),
    ) %>%
  {
    heatscatter(
      .$RNA_TE_corr, .$RNA,
      # xlim = c(0, 4), ylim = c(0, 4),
      xlab = "RNA_TE_corr", 
      ylab = "RNA", bty='n', las=1,
      main = str_glue("r = {.$r[1] %>% round(3)}, rho = {.$rho[1] %>% round(3)}"),
      cexplot = 1.5,
    )
  }
dev.off()