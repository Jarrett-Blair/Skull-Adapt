library(ggplot2)
library(reshape2)
library(dplyr)

setwd("C:/Users/blair/OneDrive - UBC/Skull-Adapt")

# Defining functions

plotDF = function(df){
  df$Score = as.character(df$Score)
  colnames(df)[5] = "Fine-tuned"
  colnames(df)[6] = "Photo Full"
  colnames(df)[7] = "Photo Subset"
  
  
  m_df = melt(df)
  m_df$Score <- factor(m_df$Score, levels = rev(levels(as.factor(m_df$Score))))
  colnames(m_df)[2] = "Model"
  m_df <- m_df %>%
    group_by(Model) %>%
    mutate(Total = sum(value),                # Calculate the total value per group
           Proportion = value / Total) %>%    # Calculate the proportion of each value
    select(-c(Total, value)) 
  colnames(m_df)[3] = "value"
  return(m_df)
}

mult_format = function(mult = 100) {
  function(x) format(mult*x,digits = 2) 
}

# Total plot

total_scores = read.csv("Grad-CAM_Total.csv")
m_t_scores = plotDF(total_scores)

p = ggplot(m_t_scores, aes(fill=Score, y=value, x=Model)) + 
  geom_bar(position="fill", stat="identity", alpha = 0.8) +
  # geom_text(aes(label = "n = 100", y = 0.97), position = position_stack(vjust = 1.05), size = 3.5) +
  scale_y_continuous(limits = c(0, 1.05), expand = c(0, 0),
                     labels = mult_format()) +
  scale_fill_manual(values = c("#B20000", "#DAA520", "#006400", "#00008B")) +
  ylab("Number\nof images") +
  labs(title = "Total scores") +
  theme_classic() +
  theme(axis.title = element_text(size = 16, face = "bold"),
        axis.title.y = element_text(angle = 0, hjust = 0.5, vjust = 0.5),
        axis.ticks = element_blank(),
        axis.text = element_text(size = 12, colour = "black"),
        legend.text = element_text(size = 11, colour = "black"),
        legend.title = element_text(size = 12, face = "bold"),
        plot.title = element_text(size = 18, face = "bold", hjust = 0.5, vjust = 0.5))
ggdraw(p) +
  draw_label("(c)", x = 0, y = 1, hjust = 0, vjust = 1, color = "black", size = 18)

ggsave("plots/tot_scores.png", plot = p, width = 8, height = 6, dpi = 300, device = 'png')



