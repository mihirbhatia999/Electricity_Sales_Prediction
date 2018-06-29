# Cleaning data ----------------------------
#libraries 
library(corrplot)
library(glmnet)
library(mgcv)
library(gam)
library(randomForest)
library(ModelMetrics)
library(earth)
library(MASS)
#extracting a preprocessing dataset 

df <- read.csv('Midterm_FittingData.csv',sep = '\t',stringsAsFactors = FALSE,)
df <- df[-1,]
df[,3:10] <- as.numeric(unlist(df[,3:10]))

df$DX32   <-  as.numeric(df$DX32)
df$DT00   <-  as.numeric(df$DT00)
df$DT32   <-  as.numeric(df$DT32)
df$DP01   <-  as.numeric(df$DP01)
df$DP05   <-  as.numeric(df$DP05)
df$DP10   <-  as.numeric(df$DP10)

df$VISIB <- as.numeric(df$VISIB)
df$WDSP <- as.numeric(df$WDSP)
df$MWSPD <- as.numeric(df$MWSPD)        
df$GUST<- as.numeric(df$GUST)  

df$DT90 <- as.numeric(df$DT90)
df$UNEMPRATE <- as.numeric(df$UNEMPRATE)
df$PCINCOME <- as.numeric(df$PCINCOME)
df$GSP <- as.numeric(df$GSP)

colnames(df)[1] <- "year"
print(head(df))

summary(df)

#removing NA value rows (#4 rows will be removed)
df <- df[complete.cases(df), ]

# Residential Electricity Sales is the response variable.

#apply pairs()
pairs(df[,c(4,25:33)])
#from the pairs function we see that MXSD, TSNW have no contribution to res.sales.adj 
#so we remove these rows 
df <- df[,c(-8,-10)]


#Linear Model------------------------------------------------------------------------------------------
set.seed(101)
model1 <- lm(res.sales.adj ~ . , data=df)


df<-df[sample(nrow(df)),] #shuffle 
k=10
folds <- cut(seq(1,nrow(df)),breaks=k,labels=FALSE)

r2.lm <- c(1:k)
rmse.lm <- c(1:k)

r2.model2 <- c(1:k)
rmse.model2<- c(1:k)

r2.lasso <- c(1:k)
rmse.lasso <- c(1:k)

r2.ridge <- c(1:k)
rmse.ridge <- c(1:k)

r2.gam <- c(1:k)
rmse.gam <- c(1:k)

r2.mars <- c(1:k)
rmse.mars <- c(1:k)

r2.rf <- c(1:k)
rmse.rf <- c(1:k)


for(i in 1:k){
  #LINEAR MODEL------------------------------------------------------------------------------------------------
  #Segement your data by fold using the which() function 
  testIndexes <- which(folds==i,arr.ind=TRUE)
  testData <- df[testIndexes, ]
  trainData <- df[-testIndexes, ]
  #print(head(trainData))
  #model1.lm <- lm(y1~x1+x2+x3+x4+x5+x6,data = trainData)
  model1.lm <- lm(res.sales.adj ~ month + res.price + com.price + com.sales.adj + DT90 + HTDD + CLDD, data=trainData)
  model1.predicted.lm <- predict(model1.lm,newdata=testData)
  model1.prediction.table <- as.data.frame(cbind(model1.predicted.lm,testData$res.sales.adj))
  colnames(model1.prediction.table) <- c('predicted','actual')
  error.lm <- sum((model1.prediction.table$predicted - model1.prediction.table$actual)^2)
  SSE = error.lm
  SST = sum((model1.prediction.table$actual - mean(model1.prediction.table$predicted))^2)
  r2.lm[i] <- 1 - (SSE/SST)
  rmse.lm[i] <- sqrt(SSE/(nrow(model1.prediction.table)))
  
  #BOXCOX----------------------------------------------------------------------------------------------------
  b = boxcox(res.sales.adj ~ month + res.price + com.price + com.sales.adj + DT90 + HTDD + CLDD,data=trainData)
  
  lambda <-  b$x
  lik <- b$y
  bc <- cbind(lambda,lik)
  bc <- bc[order(-lik),]
  lambda.max <- bc[1,1]
  
  #now we can make linearmodel2
  model2.lm <- lm(((((res.sales.adj)^(lambda.max)-1))/(lambda.max))~ 
                    month + res.price + com.price + com.sales.adj + DT90 + HTDD + CLDD,data=trainData)
  model2.lm.pred <- predict(model2.lm,newdata = testData)
  model2.lm.actual <- (testData[,4]^(lambda.max) - 1)/(lambda.max)
  
  sse <- sum((model2.lm.pred-model2.lm.actual)^2)
  sst <- sum((model2.lm.actual - mean(model2.lm.actual))^2)
  rmse(model2.lm.actual,model2.lm.pred)
  r2.model2[i] <- 1 - (sse/sst)
  
  #LASSO---------------------------------------------------------------------------------------------------
  x <- as.matrix(trainData[,-4])
  y <- as.double(as.matrix(trainData[,4]))
  
  #fitting the rdge regression model 
  cv.lasso <- cv.glmnet(x, y, family='gaussian', alpha=1, parallel=TRUE, standardize=TRUE, type.measure='auc')
  
  # Results
  #plot(cv.lasso)
  #plot(cv.lasso$glmnet.fit, xvar="lambda", label=TRUE)
  cv.lasso$lambda.min
  cv.lasso$lambda.1se
  lasso.coef <- coef(cv.lasso, s=cv.lasso$lambda.min)
  
  #cross validation using lasso regression using the test data 
  x.matrix <- as.matrix(testData[,-4])
  x.matrix <- cbind(matrix(1,nrow=nrow(testData)),x.matrix)
  lasso.pred <- x.matrix %*% lasso.coef
  lasso.actual <- testData[,4]
  
  
  sse <- sum((lasso.pred - lasso.actual)^2) 
  sst <- sum((lasso.actual - mean(lasso.actual))^2)
  r2.lasso[i] <- 1 - (sse/sst)
  rmse.lasso[i] <- sqrt(sse/nrow(lasso.pred))
  
  #RIDGE------------------------------------------------------------------------------------------------------------
  x <- as.matrix(trainData[,-4])
  y <- as.double(as.matrix(trainData[,4]))
  
  #fitting the rdge regression model 
  cv.ridge <- cv.glmnet(x, y, family='gaussian', alpha=0, parallel=TRUE, standardize=TRUE, type.measure='auc')
  
  # Results
  #plot(cv.ridge)
  #plot(cv.ridge$glmnet.fit, xvar="lambda", label=TRUE)
  cv.ridge$lambda.min
  cv.ridge$lambda.1se
  ridge.coef <- coef(cv.ridge, s=cv.ridge$lambda.min)
  
  #cross validation using ridge regression using the test data 
  x.matrix <- as.matrix(testData[,-4])
  x.matrix <- cbind(matrix(1,nrow=nrow(testData)),x.matrix)
  ridge.pred <- x.matrix %*% ridge.coef
  ridge.actual <- testData[,4]
  sse <- sum((ridge.pred - ridge.actual)^2) 
  sst <- sum((ridge.actual - mean(ridge.actual))^2)
  r2.ridge[i] <- 1 - (sse/sst)
  rmse.ridge[i] <- sqrt(sse/length(ridge.actual))
  
  #GAM------------------------------------------------------------------------------------------------------
  gam.object <- gam(res.sales.adj ~ month + res.price + com.price + com.sales.adj + DT90 + HTDD + CLDD, family = gaussian(), data=trainData)
  
  gam.step <- step.Gam(gam.object,scope = list("month"=~1+month+s(month,df=2),"res.price"=~1+
                                                 res.price+s(res.price,df=2)+s(res.price, df=3)+s(res.price,df=4),
                                               "com.price"=~1+com.price,"com.sales.adj"=~1+com.sales.adj,"DT90"=~1+DT90+s(DT90,df=2),
                                               "HTDD"=~1+HTDD, "CLDD"=~1+CLDD),direction = "both",trace=FALSE)

  gam.object.final <- gam.step
  prediction.gam.out <- predict.Gam(gam.object.final,newdata = testData)
  actual.gam.out <- testData[,4]
  sse.out <- sum((prediction.gam.out-actual.gam.out)^2)
  sst.out <- sum((actual.gam.out - mean(actual.gam.out))^2)
  
  r2.gam[i] <- 1-(sse.out/sst.out)
  rmse.gam[i] <- sqrt(sse.out/(length(actual.gam.out)))
  
  #MARS------------------------------------------------------------------------------------------------------------------------
  model.mars <- earth(res.sales.adj ~ month + res.price + com.price + com.sales.adj + DT90 + HTDD + CLDD
                      , data = trainData, degree=3,penalty=-1)
  pred.mars <- predict(model.mars,newdata = testData)
  sse.mars <- sum((pred.mars - testData[,4])^2)
  r2.mars[i] <- 1 - (sse.mars/sst)
  rmse.mars[i] <- sqrt(sse.mars/(nrow(testData)))
}



sample_size <- floor(0.85 * nrow(df))
## set the seed to make your partition reproductible
set.seed(114)
train.indexes <- sample(seq_len(nrow(df)), size = sample_size)
df.train <- df[train.indexes, ]
df.test <- df[-train.indexes, ]
#Random Forests-------------------------------------------------------------------------------------------- 
model.rf <- randomForest(res.sales.adj ~ . , data=df.train)
#importance of the variables in RF
importance(model.rf)
predict.rf <- predict(model.rf,newdata = df.test)
predict.rf.in <- predict(model.rf,newdata = df.train)
actual.rf.in <- df.train[,4]
actual.rf <- df.test[,4]
sse.in <- sum((predict.rf.in-actual.rf.in)^2)
sse.out <- sum((predict.rf - actual.rf)^2)
sst <- sum((actual.rf-mean(actual.rf))^2)
r2.rf.in <- 1-(sse.in/sst)
r2.rf.out <- 1-(sse.out/sst)

print(r2.rf.out)
rmse.rf <- sqrt(sse.out/(nrow(df.test)))
#random forests performs very well in sample but peforms worse than linear model Out sample 



#RESULTS FROM LINEAR MODEL 1 --------------------------------------------------------------------------------------------------

print(r2.lm)
print(rmse.lm)
#Linear model performs pretty well 
model1.lm <- lm(res.sales.adj ~ month + res.price + com.price + com.sales.adj + DT90 + HTDD + CLDD, data=df)
#plot(model1.lm)
#to improve performance further we check if the inputs x1.....x32 have high correlations 
cor.df <- cor(df[,c("month","res.price","com.price","com.sales.adj","DT90","HTDD","CLDD")])
corrplot(cor.df)
print("r^2 value using simple linear regression 10 fold CV")
print(mean(r2.lm))
print("RMSE value using simple linear regresssion 10 fold CV")
print(mean(rmse.lm))

#RESULTS FROM LINEAR MODEL 2 ----------------------------------------------------------------------------------

print("r2 of model 2 10 fold CV")
print(mean(r2.model2))
rmse.model2 <- NA

#RESULTS FROM LASSO REGRESSION----------------------------------------------------------------------------------

print("r2 of lasso regression 10 fold CV")
print(mean(r2.lasso))
print("rmse of lasso regression 10 fold CV")
print(mean(rmse.lasso))

#RESULTS FROM RIDGE REGRESSION----------------------------------------------------------------------------------

print("r2 of ridge regression 10 fold CV")
print(mean(r2.ridge))
print("rmse of ridge regression 10 fold CV ")
print(mean(rmse.ridge))

#RESULTS FROM GAM-----------------------------------------------------------------------------------------------
print("r2 of GAM 10 fold CV")
print(mean(r2.gam))
print("rmse of GAM 10 fold CV ")
print(mean(rmse.gam))

#RESULT FROM MARS-----------------------------------------------------------------------------------------------
print("r2 of MARS 10 fold CV")
print(mean(r2.mars))
print("rmse of MARS 10 fold CV ")
print(mean(rmse.mars))

#RESULT FROM RANDOM FOREST--------------------------------------------------------------------------------------
print("Using Random Forest r2 value is:")
print(r2.rf.out)
print("Using Random Forest RMSE value is:")
rmse.rf <- rmse(actual = actual.rf,predicted = predict.rf)
print(rmse.rf)


#All RESULTS--------------------------------------------------------------------------------------------------

models <- c("linear model", "boxcox linear model", "lasso", "ridge","GAM","MARS","Random Forest")
R2 <- c(mean(r2.lm),mean(r2.model2),mean(r2.lasso),mean(r2.ridge),mean(r2.gam),mean(r2.mars),r2.rf.out)
RMSE <- c(mean(rmse.lm),mean(rmse.model2),mean(rmse.lasso),mean(rmse.ridge),mean(rmse.gam),mean(rmse.mars),rmse.rf)
AllModelsTable <- cbind(models,R2,RMSE)
AllModelsTable <- as.data.frame(AllModelsTable)

AllModelsTable2 <- AllModelsTable[order(AllModelsTable$R2),]
print(AllModelsTable2)
#From the resits of the table we can say that Random Forests is the best performing among all the models 
#we hence save the object random forests using all the data available in df to make an object that would be very good at predictions 

#CREATING FINAL MODEL 
#We use this because the final model should be trained using the maximum possible data 
#The cross validation was only used to find the best possible model 
finalmodel <- randomForest(res.sales.adj ~ . , data=df)
varImpPlot(finalmodel)
