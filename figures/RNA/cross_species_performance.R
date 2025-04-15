library(tidyverse)
library(LSD)
library(cowplot)

predictions <- read.csv("./results/cross_species/human_model_on_mouse_RNA.csv", 
  header = TRUE, sep = "\t"
)


predictions <- predictions %>% 
  rename(ObservedRNA = mean_te_true, PredictedRNA = mean_te_pred)


pdf("./figures/RNA/human_model_on_mouse_RNA.pdf", 5, 5) 
# 'predictions' below is a dataframe with at least two columns named PredictedRNA and ObservedRNA
predictions %>% 
  mutate(
    r2 = cor(PredictedRNA, ObservedRNA)^2,
    r = cor(PredictedRNA, ObservedRNA),
    rho = cor(PredictedRNA, ObservedRNA, method = "spearman"),
    ) %>%
  {
    heatscatter(
      .$PredictedRNA, .$ObservedRNA,
      xlab = "CV fitted model prediction (human)",
      xlim = c(-3, 3),
      ylim = c(-3, 3),
      ylab = "RNA (mouse)", bty='n', cex=0.3, las=1,
      main = str_glue("r2 = {.$r2[1] %>% round(3)}, r = {.$r[1] %>% round(3)}, rho = {.$rho[1] %>% round(3)}")
    )
  }
dev.off()

predictions <- read.csv("./results/cross_species/mouse_model_on_human_RNA.csv", 
  header = TRUE, sep = "\t"
)


predictions <- predictions %>% 
  rename(ObservedRNA = mean_te_true, PredictedRNA = mean_te_pred)


pdf("./figures/RNA/mouse_model_on_human_RNA.pdf", 5, 5) 
# 'predictions' below is a dataframe with at least two columns named PredictedRNA and ObservedRNA
predictions %>% 
  mutate(
    r2 = cor(PredictedRNA, ObservedRNA)^2,
    r = cor(PredictedRNA, ObservedRNA),
    rho = cor(PredictedRNA, ObservedRNA, method = "spearman"),
    ) %>%
  {
    heatscatter(
      .$PredictedRNA, .$ObservedRNA,
      xlab = "CV fitted model prediction (mouse)",
      xlim = c(-3, 3),
      ylim = c(-3, 3),
      ylab = "RNA (human)", bty='n', cex=0.3, las=1,
      main = str_glue("r2 = {.$r2[1] %>% round(3)}, r = {.$r[1] %>% round(3)}, rho = {.$rho[1] %>% round(3)}")
    )
  }
dev.off()