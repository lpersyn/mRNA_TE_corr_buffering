library(tidyverse)
library(LSD)
library(cowplot)

predictions <- read.csv("./results/cross_species/human_model_on_mouse_TE.csv", 
  header = TRUE, sep = "\t"
)


predictions <- predictions %>% 
  rename(ObservedTE = mean_te_true, PredictedTE = mean_te_pred)


pdf("./figures/TE/human_model_on_mouse_TE.pdf", 5, 5) 
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
      ylim = c(-3, 3),
      ylab = "TE (mouse)", bty='n', cex=0.3, las=1,
      main = str_glue("r2 = {.$r2[1] %>% round(3)}, r = {.$r[1] %>% round(3)}, rho = {.$rho[1] %>% round(3)}")
    )
  }
dev.off()

predictions <- read.csv("./results/cross_species/mouse_model_on_human_TE.csv", 
  header = TRUE, sep = "\t"
)


predictions <- predictions %>% 
  rename(ObservedTE = mean_te_true, PredictedTE = mean_te_pred)


pdf("./figures/TE/mouse_model_on_human_TE.pdf", 5, 5) 
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
      ylim = c(-3, 3),
      ylab = "TE (human)", bty='n', cex=0.3, las=1,
      main = str_glue("r2 = {.$r2[1] %>% round(3)}, r = {.$r[1] %>% round(3)}, rho = {.$rho[1] %>% round(3)}")
    )
  }
dev.off()