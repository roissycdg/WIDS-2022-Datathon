library(readr)
# install.packages("skimr")
library(skimr)
library(dataPreparation)
library(tidyverse)
train <- read_csv("train.csv", col_types = cols(building_class = col_factor(levels = c("Commercial","Residential"))))
test <- read_csv("test.csv")                                                                                     
# View(train)



# Variables with 40,000+ missing
# direction_max_wind_speed
# direction_peak_wind_speed
# max_wind_speed
# days_with_fog

newdata <- na.omit(train$direction_max_wind_speed, train$direction_peak_wind_speed)
# Large but not as high number of NAs
# energy_star_rating  # has 26,709 NAs
# year_built  # 1837 NAs

###############################################
#          Exploratory Data Analysis          #
###############################################

# Check dimension of data 
dim(train)     # 75757    64

# Summary of data
skim(train)
skim(test)

# Check missing data : Which columns have NAs
TrainNA <- colSums(is.na(train))

TrainNA1 <- as.vector(TrainNA)
total <- merge(train, TrainNA1, by=c("ID","Country"))
newdata <- Train[order(Train$TrainNA),]

# energy_star_rating 26709
# year_built 1837
# direction_max_wind_speed 41082
colSums(is.na(test))    # 


# Replace missing values with median
train <- train %>% 
  mutate(children=ifelse(is.na(children), 0, children))


# TEXT OUTPUT
# FIRST only select id, (predicted) site_eui from test
# Then output using sink() function
# The sink("filename") function redirects output to the file filename. 
# Including the option split=TRUE will send output to both the screen and the output file.
# Issuing the command sink() without options will return output to the screen alone.
sink("submission", split = TRUE)
