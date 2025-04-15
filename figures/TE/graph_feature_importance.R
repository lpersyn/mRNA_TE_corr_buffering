library(tidyverse)
library(cowplot)

generate_feature_importance_plot <- function(feature_importance_path, features_path, predictions_path, top_k, output_path) {
    codon_to_aa_dict <- c(
        "TCA" = "S",    # Serine
        "TCC" = "S",    # Serine
        "TCG" = "S",    # Serine
        "TCT" = "S",    # Serine
        "TTC" = "F",    # Phenylalanine
        "TTT" = "F",    # Phenylalanine
        "TTA" = "L",    # Leucine
        "TTG" = "L",    # Leucine
        "TAC" = "Y",    # Tyrosine
        "TAT" = "Y",    # Tyrosine
        "TAA" = "X",    # Stop
        "TAG" = "X",    # Stop
        "TGC" = "C",    # Cysteine
        "TGT" = "C",    # Cysteine
        "TGA" = "X",    # Stop
        "TGG" = "W",    # Tryptophan
        "CTA" = "L",    # Leucine
        "CTC" = "L",    # Leucine
        "CTG" = "L",    # Leucine
        "CTT" = "L",    # Leucine
        "CCA" = "P",    # Proline
        "CCC" = "P",    # Proline
        "CCG" = "P",    # Proline
        "CCT" = "P",    # Proline
        "CAC" = "H",    # Histidine
        "CAT" = "H",    # Histidine
        "CAA" = "Q",    # Glutamine
        "CAG" = "Q",    # Glutamine
        "CGA" = "R",    # Arginine
        "CGC" = "R",    # Arginine
        "CGG" = "R",    # Arginine
        "CGT" = "R",    # Arginine
        "ATA" = "I",    # Isoleucine
        "ATC" = "I",    # Isoleucine
        "ATT" = "I",    # Isoleucine
        "ATG" = "M",    # Methionine (start)
        "ACA" = "T",    # Threonine
        "ACC" = "T",    # Threonine
        "ACG" = "T",    # Threonine
        "ACT" = "T",    # Threonine
        "AAC" = "N",    # Asparagine
        "AAT" = "N",    # Asparagine
        "AAA" = "K",    # Lysine
        "AAG" = "K",    # Lysine
        "AGC" = "S",    # Serine
        "AGT" = "S",    # Serine
        "AGA" = "R",    # Arginine
        "AGG" = "R",    # Arginine
        "GTA" = "V",    # Valine
        "GTC" = "V",    # Valine
        "GTG" = "V",    # Valine
        "GTT" = "V",    # Valine
        "GCA" = "A",    # Alanine
        "GCC" = "A",    # Alanine
        "GCG" = "A",    # Alanine
        "GCT" = "A",    # Alanine
        "GAC" = "D",    # Aspartic Acid
        "GAT" = "D",    # Aspartic Acid
        "GAA" = "E",    # Glutamic Acid
        "GAG" = "E",    # Glutamic Acid
        "GGA" = "G",    # Glycine
        "GGC" = "G",    # Glycine
        "GGG" = "G",    # Glycine
        "GGT" = "G"     # Glycine
    )
    amino_acid_to_color <- c(
        "S" = "#FF0000",    # Serine
        "F" = "#ffbb00",    # Phenylalanine
        "L" = "#ffd500",    # Leucine
        "Y" = "#c3ff00",    # Tyrosine
        "X" = "#6abc00",    # Stop
        "C" = "#00ffa2",    # Cysteine
        "W" = "#00b3ff",    # Tryptophan
        "P" = "#005eff",    # Proline
        "H" = "#a86eff",    # Histidine
        "Q" = "#9900ff",    # Glutamine
        "R" = "#fa93ff",    # Arginine
        "I" = "#ff00e6",    # Isoleucine
        "M" = "#9d004f",    # Methionine (start)
        "T" = "#950000",    # Threonine
        "N" = "#006f82",    # Asparagine
        "K" = "#004123",    # Lysine
        "V" = "#0b005c",    # Valine
        "A" = "#5f2800",    # Alanine
        "D" = "#ce800a",    # Aspartic Acid
        "E" = "#00ffe5",    # Glutamic Acid
        "G" = "#744c4c",     # Glycine
        "none" = "gray75"
    )

    # Load data
    data <- read.csv(feature_importance_path, header = TRUE, row.names = 1)
    features <- read.csv(features_path, header = TRUE)

    predictions <- read.csv(predictions_path, header = TRUE)

    features <- merge(features, predictions, by = "SYMBOL")
    # print(head(features))

    # features <- features[, -grep("fold", names(features))]
    # features <- features[, -grep("SYMBOL", names(features))]
    # features <- features[, -grep("transcript_id", names(features))]
    # features <- features[, -grep("gene_id", names(features))]

    bio_source_cols <- grep("_true", names(features), value = TRUE)
    # Replace Inf values with NA
    # features$struct_min_dG_UTR5[features$struct_min_dG_UTR5 == Inf] <- NA
    # features$struct_min_dG_CDS[features$struct_min_dG_CDS == Inf] <- NA

    # Compute mean correlation of biosources with features and add to data
    for (feature in row.names(data)) {
        if (feature == "aa_freq_X") {
            next
        }
        temp <- grep(feature, names(features), value = TRUE)
        corrs <- cor(features[, temp], features[, bio_source_cols], use = "complete.obs", method = "spearman")
        stopifnot(length(corrs) == length(bio_source_cols))
        stopifnot(!any(is.na(corrs)))
        mean_corr <- mean(corrs)
        data[feature, "mean_corr"] <- mean_corr
    }

    data$amino_acid <- "none"
    for (i in 1:length(row.names(data))) {
        if (grepl("_[ATGC]{3}", row.names(data)[i])) {
            codon <- regmatches(row.names(data)[i], regexpr("[ATGC]{3}", row.names(data)[i]))
            aa <- codon_to_aa_dict[codon]
            row.names(data)[i] <- paste(row.names(data)[i], aa, sep = "_")
            data$amino_acid[i] <- aa
        }
    }

    # Prepare data for plotting
    data <- data[order(data$mean_importance, decreasing = TRUE), ]
    data$log_mean_importance <- log10(data$mean_importance)
    first <- data[1:top_k, ]

    # Create plot
    p <- ggplot(first, aes(x = log_mean_importance, y = reorder(row.names(first), log_mean_importance))) +
        geom_bar(stat = "identity", aes(fill = mean_corr), color = "black", size = 0.2) +  # Add black border
        scale_fill_gradient2(low = "red", high = "blue", midpoint = 0, name = "Mean spearman rho") +
        labs(x = expression("log"[10]*"(mean importance)"), y = str_glue("Top {top_k} features")) +
        theme_minimal() +
        theme(
            panel.grid.major.x = element_blank(),
            panel.grid.major.y = element_blank(),
            panel.grid.minor.x = element_blank(),
            panel.grid.minor.y = element_blank(),
            axis.line.x = element_line(color = "black"),
            axis.line.y = element_blank(),
            axis.ticks.x = element_line(color = "black"),
            axis.ticks.y = element_blank(),
            axis.text.y = element_text(size = 5),
            axis.title.x = element_text(size = 7),
            axis.title.y = element_text(size = 7),
            legend.position = "none",
        )

    legend <- get_legend(p + theme(
        legend.position = "bottom",
        legend.text = element_text(size = 4),  # Reduce text size
        legend.title = element_text(size = 5, hjust = 0.5),  # Center title on legend bar
        legend.key.height = unit(0.3, "cm")  # Reduce key height
    ))
    p <- plot_grid(p, legend, ncol = 1, nrow = 2, rel_heights = c(1, 0.1))  # Adjust relative heights

    # Save plot
    ggsave(
        filename = output_path,
        plot = p,
        width = 3,
        height = 3,
        units = "in"
    )
}


# Example usage
generate_feature_importance_plot(
    feature_importance_path = "./results/human/TE/lgbm-LL_P5_P3_CF_AAF_3mer_freq_5/feature_importance.csv",
    features_path = "./figures/human_features.csv",
    predictions_path = "./results/human/TE/lgbm-LL_P5_P3_CF_AAF_3mer_freq_5/predictions.csv",
    top_k = 20,
    output_path = "./figures/TE/human_feature_importance.pdf"
)

generate_feature_importance_plot(
    feature_importance_path = "./results/mouse/TE/lgbm-LL_P5_P3_CF_AAF_3mer_freq_5/feature_importance.csv",
    features_path = "./figures/mouse_features.csv",
    predictions_path = "./results/mouse/TE/lgbm-LL_P5_P3_CF_AAF_3mer_freq_5/predictions.csv",
    top_k = 20,
    output_path = "./figures/TE/mouse_feature_importance.pdf"
)

