###data checking and preparation

library(tidyverse)
library(missForest)
library(caret)
library(tree)
library(caret)
library(tidyverse)
library(xgboost)
library(MLeval)
library(readr)
library(stringr)
library(car)
library(corrplot)
library(permimp)
library(visdat)

final_data$V_SUM_INSURED <- ifelse(final_data$Q_MC_PACKAGE == "COMFORT" & is.na(final_data$V_SUM_INSURED)==T, 0, final_data$V_SUM_INSURED)
view(filter(final_data, is.na(V_CARAGE) == T))
view(filter(final_data, V_CARBRAND == "FORD" & (V_SUM_INSURED > 33000  & V_SUM_INSURED <36000) & (V_ENGPOWER==110) & PH_N_AGE ==55))
view(filter(final_data, V_CARBRAND == "SEAT" &  PH_N_AGE ==29 & V_ENGPOWER==85))

final_data[which(is.na(final_data$V_CARAGE) == T & final_data$V_CARBRAND == "FORD"),]$V_CARAGE<-4  
final_data[which(is.na(final_data$V_CARAGE) == T & final_data$V_CARBRAND == "SEAT"),]$V_CARAGE<-18  

#categorical data as factors

final_data$Q_QUOTATION_SOURCE <- as.factor(final_data$Q_QUOTATION_SOURCE)
final_data$Q_MC_PACKAGE <- as.factor(final_data$Q_MC_PACKAGE)
final_data$AG_SALESCHANNEL <- as.factor(final_data$AG_SALESCHANNEL)
final_data$Q_CREATE_QUARTER <- as.factor(final_data$Q_CREATE_QUARTER)
final_data$V_CARBRAND <- as.factor(final_data$V_CARBRAND)
final_data$V_FUELTYPE <- as.factor(final_data$V_FUELTYPE)
final_data$PH_PARTNER_TYPE    <- as.factor(final_data$PH_PARTNER_TYPE)
final_data$Q_IS_CONVERTED    <- as.factor(final_data$Q_IS_CONVERTED)

brands<-c(unique(as.character(final_data$V_CARBRAND)))

view(final_data%>%group_by(V_CARBRAND)%>%count()%>%arrange(desc(n)))

final_data_21<-final_data%>%filter(Q_CREATE_QUARTER == "1.1.2021" | Q_CREATE_QUARTER == "1.2.2021") #only first half of 2021
brands_count<-final_data_21%>%group_by(V_CARBRAND)%>%count()%>%arrange(desc(n))
brands_freq<-as.character(brands_count$V_CARBRAND[1:20])
brands_frequent <- final_data_21%>%filter(V_CARBRAND %in% c(brands_freq)) #only 20 most frequent car makes
brands_frequent<-brands_frequent[,c(2,4:17)]
summary(brands_frequent)
str(brands_frequent)

#making names in proepr format for R
brands_frequent$AG_SALESCHANNEL <- make.names(brands_frequent$AG_SALESCHANNEL)
brands_frequent$V_FUELTYPE <- make.names(brands_frequent$V_FUELTYPE)
brands_frequent$Q_QUOTATION_SOURCE <- make.names(brands_frequent$Q_QUOTATION_SOURCE)
brands_frequent$Q_MC_PACKAGE <- make.names(brands_frequent$Q_MC_PACKAGE)
brands_frequent$V_CARBRAND <- make.names(brands_frequent$V_CARBRAND)
brands_frequent$PH_PARTNER_TYPE <- make.names(brands_frequent$PH_PARTNER_TYPE)

levels(brands_frequent$Q_IS_CONVERTED) <- c("No", "Yes")
brands_frequent$Q_IS_CONVERTED<-relevel(brands_frequent$Q_IS_CONVERTED, ref="No")


unique(brands_frequent$PH_N_AGE)
boxplot(brands_frequent$V_SUM_INSURED)
view(brands_frequent%>%filter(V_SUM_INSURED > 200000))
brands_frequent[which(brands_frequent$V_SUM_INSURED > 1000000),]$V_SUM_INSURED<-0
brands_frequent[which(brands_frequent$V_SUM_INSURED > 250000 & (brands_frequent$V_CARBRAND == "FIAT" | brands_frequent$V_CARBRAND == "HYUNDAI")),]$V_SUM_INSURED<-0 #rather 0 than NA as it is comfort

boxplot(brands_frequent$Q_PREMIUM)
boxplot(brands_frequent$Q_DISC_TOTAL)
unique(brands_frequent$C_N_CI )
boxplot(brands_frequent$V_CARAGE)
view(brands_frequent%>%filter(V_CARAGE > 60)) #correct, compared with horse power = really old cars
boxplot(brands_frequent$V_ENGPOWER)
view(brands_frequent%>%filter(V_ENGPOWER > 600)) 

brands_frequent[which(brands_frequent$V_ENGPOWER > 600),]$V_ENGPOWER<-NA #with threshold 600 all unrealistic cars to have that power 

boxplot(brands_frequent$PH_N_AGE)
view(brands_frequent%>%filter(PH_N_AGE > 99 | PH_N_AGE < 18)) 
brands_frequent[which(brands_frequent$PH_N_AGE > 99 | brands_frequent$PH_N_AGE < 18),]$PH_N_AGE<-NA #impossible to have driver younger 18 and older 100 unlikely/big outliers

boxplot(brands_frequent$Q_PREMIUM)
view(brands_frequent%>%filter(Q_PREMIUM > 5000)) #legit

unique(brands_frequent$V_FUELTYPE)

brands_frequent$V_FUELTYPE<-as.factor(as.character(ifelse(brands_frequent$V_FUELTYPE == "2" | brands_frequent$V_FUELTYPE == "02-motorova-nafta", "diesel", 
                                                          ifelse(brands_frequent$V_FUELTYPE == "07-kombinacia-benzin-a-lpg" | brands_frequent$V_FUELTYPE == "06-stlaceny-zemny-plyn"| brands_frequent$V_FUELTYPE == "03-propan-butan","lpg",
                                                                 ifelse(brands_frequent$V_FUELTYPE == "08-hybrid" | brands_frequent$V_FUELTYPE == "04-elektricky-pohon" ,"hybrid-electro",
                                                                        ifelse(brands_frequent$V_FUELTYPE == "01-benzin", "gasoline",
                                                                               ifelse(brands_frequent$V_FUELTYPE == "05-iny", "other",  brands_frequent$V_FUELTYPE)))))))

view(brands_frequent%>%filter(is.na(AG_SALESCHANNEL) == T)) 
view(brands_frequent%>%filter(Q_QUOTATION_SOURCE == "02_AZ_Online"))

unique(brands_frequent$AG_SALESCHANNEL)
brands_frequent$AG_SALESCHANNEL <- as.factor(as.character(ifelse(brands_frequent$Q_QUOTATION_SOURCE == "02_AZ_Online" & is.na(brands_frequent$AG_SALESCHANNEL) == T, "online", as.character(brands_frequent$AG_SALESCHANNEL))))
brands_frequent$AG_SALESCHANNEL <- as.factor(ifelse(brands_frequent$AG_SALESCHANNEL == "4" | brands_frequent$AG_SALESCHANNEL == "4-ma-makleri", "insurance-broker",
                                                    ifelse(brands_frequent$AG_SALESCHANNEL == "2" | brands_frequent$AG_SALESCHANNEL == "2-us-univerzalna-siet", "general-network",
                                                           ifelse(brands_frequent$AG_SALESCHANNEL == "7-cc-call-centrum", "call-centrum",
                                                                  ifelse(brands_frequent$AG_SALESCHANNEL == "6-fo-front-office", "front-office", as.character(brands_frequent$AG_SALESCHANNEL))))))

view(brands_frequent%>%filter(is.na(V_ENGPOWER) == T)) 

table(brands_frequent$Q_IS_CONVERTED)
data_impute<-brands_frequent[,c(1:4,6:15)]

#missing data graph
vis_miss(data_impute[,-1], warn_large_data=F)

#correlation plot
corrplot(cor(na.omit(brands_frequent[,c(6:9,12,13,15)])))