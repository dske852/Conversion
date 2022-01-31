##models


#dataset split
set.seed(25892)
trainIndex <- createDataPartition(data_impute$Q_IS_CONVERTED, p = .8, 
                                  list = FALSE, 
                                  times = 1)


train <- tree_data[ trainIndex,]
test  <- tree_data[-trainIndex,]

###DECISION TREE
tree_complex <- tree(Class~., data=down_train[,-1], split=c("deviance"),control=tree.control(186912, mincut = 5, minsize = 10, mindev = 0.01)) #control, split? ###trees must have 0-1 nie faktory 
plot(tree_complex)
text(tree_complex, cex=0.8)
#CV - misclassifications 
set.seed(2500)
cv=cv.tree(tree_complex, FUN=prune.misclass, K=5) 
cv
plot(cv$size, cv$dev, type="b")


#CV - deviance
set.seed(2500)
cv2=cv.tree(tree_complex, K=5) 
cv2
plot(cv2$size, cv2$dev, type="b")

#Prune - deviance
prune_tree=prune.tree(tree_complex, best=3)
plot(prune_tree2)
text(prune_tree2, pretty=0, cex=0.6)

prune_tree2=prune.misclass(tree_complex, best=3)

pred.train <- predict(prune_tree, newdata=down_train, type = 'class')
pred.test <- predict(prune_tree, newdata=test, type='class')


#confusion matrix
confusionMatrix(as.factor(pred.test), as.factor(test$Q_IS_CONVERTED), positive = 'Yes')


predi <- predict(tree_complex, newdata=test, type="class")

#ROC
ROCRpred0 <- prediction(as.numeric(predi),as.numeric(test_na_omit$Q_IS_CONVERTED))

ROCRperf0<- performance(ROCRpred0, 'tpr', 'fpr')

plot(ROCRperf0, colorize=TRUE, text.adj=c(-0.2,1.7))

####XGB

#Data Preparation
set.seed(9560)
up_train <- downSample(x = train_xgb[, -3],
                       y = train_xgb$Q_IS_CONVERTED)

train_xgb_up <- up_train
test_xgb_up <- test
test_xgb_up<-rename(test_xgb_up,  Class = Q_IS_CONVERTED)
test_xgb_up$Class<-as.factor(test_xgb_up$Class)

levels(test_xgb_up$Class)<-c(0,1)

train_labels_up <- as.numeric(train_xgb_up$Class)-1
test_labels_up <- as.numeric(test_xgb_up$Class)-1

table(train_xgb_up$Class)

Quatation_Source_up <- model.matrix(~Q_QUOTATION_SOURCE-1,train_xgb_up)
Package_up  <- model.matrix(~Q_MC_PACKAGE-1,train_xgb_up)
Sales_Channel_up  <- model.matrix(~AG_SALESCHANNEL-1,train_xgb_up)
Brand_up  <- model.matrix(~V_CARBRAND-1,train_xgb_up)
Fuel_up  <- model.matrix(~V_FUELTYPE-1,train_xgb_up)
Partner_Type_up  <- model.matrix(~PH_PARTNER_TYPE-1,train_xgb_up)

xgb_train_encoded_up <- cbind(Quatation_Source_up, Package_up, Sales_Channel_up, Brand_up, Fuel_up, Partner_Type_up, Q_DISC_TOTAL=train_xgb_up$Q_DISC_TOTAL, C_N_CI=train_xgb_up$C_N_CI, V_CARAGE=train_xgb_up$V_CARAGE,V_ENGPOWER= train_xgb_up$V_ENGPOWER,V_SUM_INSURED= train_xgb_up$V_SUM_INSURED)
xgb_train_encoded_matrix_up <- data.matrix(xgb_train_encoded_up)

Quatation_Source2_up <- model.matrix(~Q_QUOTATION_SOURCE-1,test_xgb_up)
Package2_up <- model.matrix(~Q_MC_PACKAGE-1,test_xgb_up)
Sales_Channel2_up <- model.matrix(~AG_SALESCHANNEL-1,test_xgb_up)
Brand2_up <- model.matrix(~V_CARBRAND-1,test_xgb_up)
Fuel2_up <- model.matrix(~V_FUELTYPE-1,test_xgb_up)
Partner_Type2_up <- model.matrix(~PH_PARTNER_TYPE-1,test_xgb_up)

xgb_test_encoded_up <- cbind(Quatation_Source2_up, Package2_up, Sales_Channel2_up, Brand2_up, Fuel2_up, Partner_Type2_up,  test_xgb_up$Q_DISC_TOTAL, test_xgb_up$C_N_CI, test_xgb_up$V_CARAGE, test_xgb_up$V_ENGPOWER, test_xgb_up$V_SUM_INSURED)
xgb_test_encoded_matrix_up <- data.matrix(xgb_test_encoded_up)

dtrain_up <- xgb.DMatrix(data = xgb_train_encoded_matrix_up, label= train_labels_up)
dtest_up <- xgb.DMatrix(data = xgb_test_encoded_matrix_up, label= test_labels_up)

#parameters
params <- list(booster = "gbtree", objective = "binary:logistic", eval_metric = "auc",eta=0.3, num_parallel_tree=3,gamma=0, max_depth=6, min_child_weight=1, subsample=1, colsample_bytree=1)

#cross validation
set.seed(1245)
xgbcv2 <- xgb.cv( params = params, data = dtrain_up, nrounds = 100, nfold = 5, showsd = T, stratified = T, print_every_n = 10, maximize = F)
max(xgbcv2[["evaluation_log"]][["test_auc_mean"]])
plot(xgbcv2[["evaluation_log"]][["test_auc_mean"]], type="b", xlab="Rounds", ylab="Mean AUC")

#model 1
xgb2 <- xgb.train (params = params, data = dtrain_up, nrounds = 20, watchlist = list(val=dtest_up,train=dtrain_up), print_every_n = 10, maximize = F)


#Confusion Mat
g<-ifelse(predict(xgb2, dtrain_up)>0.5,1,0)
confusionMatrix (as.factor(g), as.factor(train_labels_up), positive = "1")


traintask <- makeClassifTask (data = train_xgb_up[,-1],target = "Class", positive = 1)
testtask <- makeClassifTask (data = test_xgb_up[,-1],target = "Class", positive = 1)

#do one hot encoding`<br/> 
traintask1 <- createDummyFeatures(obj = traintask) 
testtask1 <- createDummyFeatures (obj = testtask)

#create learner
lrn <- makeLearner("classif.xgboost",predict.type = "prob")
lrn$par.vals <- list( objective="binary:logistic", eval_metric="auc", nrounds=200L)

#set parameter space
params <- makeParamSet( makeDiscreteParam("booster",values = c("gbtree")), makeIntegerParam("max_depth",lower = 3L,upper = 10L), makeNumericParam("min_child_weight",lower = 1L,upper = 10L), makeNumericParam("subsample", lower = 0.7, upper = 1), makeNumericParam("colsample_bytree", lower = 0.5, upper = 1), makeNumericParam("eta", lower = 0.01, upper = 0.15))

#set resampling strategy
rdesc <- makeResampleDesc("CV",stratify = T,iters=5L)

#search strategy
ctrl <- makeTuneControlRandom(maxit = 10)

#set parallel backend
library(parallel)
library(parallelMap) 
parallelStartSocket(cpus = detectCores())

#parameter tuning
mytune <- tuneParams(learner = lrn, task = traintask1, resampling = rdesc, measures = auc, par.set = params, control = ctrl, show.info = T)

mytune$y 

#Op. pars: booster=gbtree; max_depth=10; min_child_weight=5.84; subsample=0.963; colsample_bytree=0.697; eta=0.0951, auc.test.mean=0.8545072

#set hyperparameters
lrn_tune <- setHyperPars(lrn,par.vals = mytune$x)

param_tuned <- list(booster = "gbtree", objective = "binary:logistic", eval_metric = "auc",eta=0.0951, max_depth=10, min_child_weight=5.84, subsample=0.963, colsample_bytree=0.697)

xgbcv3 <- xgb.cv( params = param_tuned, data = dtrain_up, nrounds = 200, nfold = 5, showsd = T, stratified = T, print_every_n = 10, maximize = F)

plot(xgbcv3[["evaluation_log"]][["test_auc_mean"]], type="b", ylab="Mean AUC Test", xlab="Rounds")


#train model
xgmodel <- train(learner = lrn_tune,task = traintask1)

lrn2<-makeLearner("classif.xgboost",predict.type = "prob")
lrn2$par.vals <- list( objective="binary:logistic", eval_metric="auc", nrounds=30)
lrn_tune2<-setHyperPars(lrn2,par.vals = mytune$x)
xgmodel2<- train(learner = lrn_tune2,task = traintask1)
#predict model
xgpred <- predict(xgmodel2,testtask1)
xgpred <- predict(xgmodel2,traintask1)

confusionMatrix(xgpred$data$response,xgpred$data$truth, positive = "1")

predi <- predict(xgmodel2, testtask1, type="raw")

ROCRpred0 <- prediction(as.numeric(predi$data$response),as.numeric(predi$data$truth))

ROCRperf0<- ROCR::performance(ROCRpred0, 'tpr', 'fpr')



plot(ROCRperf0, colorize=TRUE, text.adj=c(-0.2,1.7)) #ROC
