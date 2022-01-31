#Interpretation graphs

#PDP
library(pdp)
p1xv2 <- pdp::partial(xgb_interpretations, pred.var = "C_N_CI", ice = TRUE, center = TRUE, plot = TRUE, rug = TRUE, alpha = 0.1, plot.engine = "ggplot2", type = "classification",train=xgb_train_encoded_matrix_up1)

library(DALEX)
pdp_glm  <- model_profile(xgb_interpretations, variables =  "C_N_CI", type = "accumulated")

#ALE
exp_model<-explain(xgb_interpretations,type="classification",data=xgb_train_encoded_matrix_up1, y=(as.numeric(naomitxgb$Class)-1))
pdp_glm  <- variable_effect(exp_model, variables =  "C_N_CI", type = "accumulated_dependency")
plot(variable_effect(exp_model, variables =  "C_N_CI", type = "accumulated_dependency"))
plot(variable_effect(exp_model, variables =  "Q_DISC_TOTAL", type = "accumulated_dependency"))
plot(variable_effect(exp_model, variables =  "AG_SALESCHANNELonline", type = "accumulated_dependency"))

plot(variable_effect(exp_model, variables =  "C_N_CI", type = "partial_dependency"), variable_effect(exp_model, variables =  "Q_DISC_TOTAL", type = "partial_dependency"), variable_effect(exp_model, variables =  "V_SUM_INSURED", type = "partial_dependency"), variable_effect(exp_model, variables =  "V_CARAGE", type = "partial_dependency"))

plot(variable_effect(exp_model, variables =  "AG_SALESCHANNELonline", type = "accumulated_dependency"), variable_effect(exp_model, variables =  "AG_SALESCHANNELgeneral.network", type = "accumulated_dependency"),variable_effect(exp_model, variables =  "PH_PARTNER_TYPEP", type = "accumulated_dependency"))

#Importance
xgb.plot.tree(feature_names = colnames(xgb_train_encoded_matrix_up1), model=xgb_interpretations)

imp<-variable_importance(exp_model)

plot(imp)

