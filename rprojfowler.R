#set your working directory, uncomment the following line and change the filepath appropriately
setwd("/Users/jennifershelton/Desktop/wids2022")

library(readr)

#import data
trainset = read.csv("train.csv", header = TRUE, stringsAsFactors = TRUE)
testset = read.csv("test.csv", header = TRUE, stringsAsFactors = TRUE)
samplesol = read.csv("sample_solution.csv", header = TRUE, stringsAsFactors = TRUE)
entrain = read.csv("encodedtrain.csv", header = TRUE, stringsAsFactors = TRUE)

#trainset = na.omit(trainset)

x=model.matrix(site_eui~.,trainset)[,-1]
y=trainset$site_eui

library(glmnet)
grid = 10^seq(10,-2,length=100)
ridge.mod=glmnet(x,y,alpha=0, lambda=grid)
dim(coef(ridge.mod ))

set.seed (1)
train=sample(1: nrow(x), nrow(x)/2)
test=(-train)
y.test=y[test]

ridge.mod =glmnet(x[train,], y[train], alpha=0, lambda=grid, thresh=1e-12)
ridge.pred=predict (ridge.mod, s=4, newx=x[test,])
mean((ridge.pred-y.test)^2)
#2024.42

set.seed (1)
cv.out = cv.glmnet(x[train,], y[train], alpha =0)
plot(cv.out)
bestlam =cv.out$lambda.min
bestlam
# 4.441674
#Therefore, we see that the value of λ that results in the smallest crossvalidation
#error is 4.44. What is the test MSE associated with this value of λ?

ridge.pred=predict(ridge.mod, s=bestlam, newx=x[test,])
mean((ridge.pred-y.test)^2)
#2025.4

lasso.mod = glmnet(x[train,], y[train], alpha =1, lambda=grid)
plot(lasso.mod)

set.seed(1)
cv.out = cv.glmnet(x[train,],y[train], alpha =1)
plot(cv.out)
bestlam = cv.out$lambda.min
lasso.pred=predict(lasso.mod, s=bestlam, newx=x[test,])
mean((lasso.pred-y.test)^2)
# MSE with this code = 2038.225


# new data set with encoding - 0 for commercial, 1 for residential. 
#entrain = na.omit(entrain)
#which(apply(entrain, c(1, 2), var)==0)
#entrain[ , which(apply(entrain, , var) != 0)]
#trainpca = prcomp(entrain, scale = TRUE)
#screeplot(trainpca, type = "l")
#df <- cbind.data.frame(trainpca$x[,1], trainpca$x[,2])
#head(df)
#df$class = entrain$site_eui
#colnames(df)[1:2] <- c("PC1","PC2")
#autoplot(trainpca, colour=df$class, loadings=TRUE, loadings.label=TRUE, data=entrain)

#library(pls)
#pcr.fit=pcr(site_eui~., data=trainset, scale=F, validation ="CV")
#summary(pcr.fit)
#validationplot(pcr.fit,val.type="MSEP")



