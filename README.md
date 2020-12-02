# Multivariate_Regression

The analysis is made with the classical data set - Boston house-prices. It contains four parts 1) ETL_feature_selection_model_selection, 2) model_tuning, 3)ensemble_methods and 4) regr_model.
1) The data are loaded directly from Github. Then they are cleaned and scaled. The colinearity is removed too.  SelectKBest is used for feature selection. f_regression and mutual_info_regression are used as score functions. The data are evaluated using KNeighborsRegressor, HuberRegressor, LinearRegression, and DecisionTreeRegressor. The models were evaluated with the mean squared error. 
KNeighborsRegressor with f_regression and 6 features showed the best performance.
2) The KNeighborsRegressor was tuned to increase its performance. The influence of parameter n_neighbors was examined on the model performance. A model with n_neighbors = 5 showed the best results.
3) Ensemble methods - RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, and AdaBoostRegressor were used to examine their performance. They performed similarly to the regression models from point 1).
4) the model was finalized in regr_model
