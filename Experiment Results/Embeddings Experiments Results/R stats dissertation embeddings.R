library (ggplot2)

F1_TK<- read_excel("OneDrive - University of Edinburgh/Biology/4th Year/Ecology Honours/Dissertation/Code/Experiments/Embedding Experiments/Experiment Results/All F1 and TK data.xlsx")


regression<-lm(mean_f1~chunk_size, data=F1_TK)
summary.lm(regression)

ggplot(F1_TK, aes(x=chunk_size,y=mean_f1))+geom_line()+ylim (0,1)



regression2<-lm(top_k_accuracy~chunk_size, data=F1_TK)
summary.lm(regression2)

ggplot(F1_TK, aes(x=chunk_size,y=top_k_accuracy))+geom_line() +ylim (0,1)
