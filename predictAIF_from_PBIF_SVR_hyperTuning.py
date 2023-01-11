# https://www.machinecurve.com/index.php/2020/11/17/how-to-perform-multioutput-regression-with-svms-in-python/

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, LeaveOneOut, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats
from scipy import stats
import itertools
from pprint import pprint

input_file = "/mnt/shared_data2/snp2003/ATLaS_PD_Processed/Statistics/ATLAS_PD_MS_ML_Compiled_Features_excelForFormat.csv"

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    # print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Average Error: {:0.4f} degrees.'.format(errors.mean()))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy

# https://stackoverflow.com/questions/49371931/unicodeerror-utf-16-stream-does-not-start-with-bom
df = pd.read_csv(input_file, encoding='utf-8')
# df = pd.read_csv(input_file, encoding='utf-16')
# Drop right-hemisphere labels in middle (11th)
# df = df.drop(11)  # Dropping row with right hemisphere labels
# df = df[df["ctx-lh-bankssts-PBIF"].str.contains("ctx-rh-bankssts-PBIF") == False]  # Dropping row with right hemisphere labels
df = df.dropna(subset=['ATLaS_ID'])  # Dropping row with right hemisphere labels
df['ATLaS_ID'] = df['ATLaS_ID'].astype(str)
df['H&Y Stage'] = df['H&Y Stage'].astype('Int32')
df.loc[:, ["Sex", "Genotype"]] = df.loc[:, ["Sex", "Genotype"]].astype("int")  # df.iloc[:, [0, 2, 3, 7]] = df.iloc[:, [0, 2, 3, 7]].astype("int")
# df.loc[:, np.r_["Age", "NET dose (mCi)":"AUC", "ctx-lh-bankssts-PBIF":len(df.columns)]] = df.loc[:, np.r_["Age", "NET dose (mCi)":"AUC", "ctx-lh-bankssts-PBIF":len(df.columns)]].astype('float')   # This does not work as np._ workswith .iloc but not .loc
df.iloc[:, np.r_[1, 4:6, 8:len(df.columns)]] = df.iloc[:, np.r_[1, 4:6, 8:len(df.columns)]].astype('float')
# Scale age, weight, and AUC by a constant. Using this method as we want to leave AIF and PBIF untouched
df.loc[:, 'Age'] = df.loc[:, 'Age']/10
df.loc[:, 'Weight (Kgs)'] = df.loc[:, 'Weight (Kgs)']/10
df.loc[:, 'AUC'] = df.loc[:, 'AUC']/1000
df = df.drop("H&Y Stage", 1)  # Dropping column "H&Y Stage" from the dataframe.
df = df.drop("AUC", 1)  # Dropping column "AUC" from the dataframe
X_end_index = df.columns.get_loc("Left-VentralDC-PBIF") + 1
X = df.loc[:, "Age":"Left-VentralDC-PBIF"]  # X = df.iloc[:, 1:X_end_index]  # X = df.iloc[:, 1:51]
y = df.loc[:, "ctx-lh-bankssts-AIF":]  # y = df.iloc[:, X_end_index:]  # y = df.iloc[:, 51:]

# Plot original VT from PBIF and VT from AIF
PBIF_start_index = df.columns.get_loc("ctx-lh-bankssts-PBIF")
# df.iloc[:, np.r_[1:3, 6:len(df.columns)]]
# df_plot = df.iloc[:, 8:51].copy()
# df_plot = df.loc[:, np.r_[0, "ctx-lh-bankssts-PBIF":"Left-VentralDC-PBIF"]].copy()  # This does not work, as np.r_ works with .iloc but not with .loc
df_plot = df.iloc[:, np.r_[0, PBIF_start_index:X_end_index]].copy()  # df_plot = df.iloc[:, np.r_[0, 8:51]].copy()
# df_plot_long = pd.melt(df_plot.astype(float), id_vars='ATLaS_ID')
df_plot_long = pd.melt(df_plot, id_vars='ATLaS_ID')
df_plot_long.columns = ['ATLaS_ID', 'regions_PBIF', 'VT_PBIF']
df_AIF = y.copy()
df_AIF_long = pd.melt(df_AIF.astype(float))
df_AIF_long.columns = ['regions_AIF', 'VT_AIF']
df_plot_long["regions_AIF"] = df_AIF_long["regions_AIF"]
df_plot_long["VT_AIF"] = df_AIF_long["VT_AIF"].values
df_plot_long["ATLaS_ID"] = df_plot_long["ATLaS_ID"].astype(str)
ATLaS_ID = df_plot_long['ATLaS_ID'].to_numpy()

# plot original data
sns.lmplot(data=df_plot_long, x='VT_PBIF', y='VT_AIF', hue="ATLaS_ID", palette="bright", ci=None)
plt.title('Observed Logan VT from PBIF and AIF - individual regression line')
plt.xlabel('Observed Logan VT from PBIF')
plt.ylabel('Observed Logan VT from AIF')

fig = plt.figure()
sns.regplot(data=df_plot_long, x='VT_PBIF', y='VT_AIF', line_kws={"color": "black"}, scatter=False)
sns.scatterplot(data=df_plot_long, x="VT_PBIF", y="VT_AIF", hue="ATLaS_ID", palette="bright", legend="full")
plt.title('Observed Logan VT from PBIF and AIF - common regression line')
plt.xlabel('Observed Logan VT from PBIF')
plt.ylabel('Observed Logan VT from AIF')
slope_o, intercept_o, r_value_o, p_value_o, std_err_o = stats.linregress(df_plot_long['VT_PBIF'], df_plot_long['VT_AIF'])
print('Pearson correlation R and p-val are: ', r_value_o.round(5), ',', p_value_o)

# View the outliers
fig = plt.figure()
plt.subplot(121)
# X_PBIF_boxplot = sns.boxplot(x="Region names", y="Observed PBIF", data=pd.melt(X.iloc[:, 7:51].astype(float), var_name='Region names', value_name='Observed PBIF'))
X_PBIF_boxplot = sns.boxplot(x="Region names", y="Observed PBIF", data=pd.melt(X.loc[:, "ctx-lh-bankssts-PBIF":"Left-VentralDC-PBIF"].astype(float), var_name='Region names', value_name='Observed PBIF'))
X_PBIF_boxplot.set_xticklabels(X_PBIF_boxplot.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.title('Boxplot of observed Logan VT from PBIF - with outlier')
plt.subplot(122)
# X_PBIF_boxplot = sns.boxplot(x="Region names", y="Observed PBIF", data=pd.melt(X.iloc[:, 7:51].astype(float), var_name='Region names', value_name='Observed PBIF'), showfliers = False)
X_PBIF_boxplot = sns.boxplot(x="Region names", y="Observed PBIF", data=pd.melt(X.loc[:, "ctx-lh-bankssts-PBIF":"Left-VentralDC-PBIF"].astype(float), var_name='Region names', value_name='Observed PBIF'), showfliers = False)
X_PBIF_boxplot.set_xticklabels(X_PBIF_boxplot.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.title('Boxplot of observed Logan VT from PBIF - hidden outlier')

fig = plt.figure()
plt.subplot(121)
y_boxplot = sns.boxplot(x="Region names", y="Observed AIF", data=pd.melt(y.astype(float), var_name='Region names', value_name='Observed AIF'))
y_boxplot.set_xticklabels(y_boxplot.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.title('Boxplot of observed Logan VT from AIF - with outlier')
plt.subplot(122)
y_boxplot = sns.boxplot(x="Region names", y="Observed AIF", data=pd.melt(y.astype(float), var_name='Region names', value_name='Observed AIF'), showfliers = False)
y_boxplot.set_xticklabels(y_boxplot.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.title('Boxplot of observed Logan VT from AIF - hidden outlier')
# plt.show()

Q1_X = X.loc[:, "ctx-lh-bankssts-PBIF":].quantile(0.25)  # Q1_X = X.iloc[:, 7:].quantile(0.25)
Q3_X = X.loc[:, "ctx-lh-bankssts-PBIF":].quantile(0.75)  # Q3_X = X.iloc[:, 7:].quantile(0.75)
IQR_X = Q3_X - Q1_X
# if filtering both upper and lower bounds
# filter = (X >= Q1 - 1.5 * IQR) & (X <= Q3 + 1.5 *IQR)
# if filtering only upper bound
filter_X = (X.loc[:, "ctx-lh-bankssts-PBIF":] <= Q3_X + 1.5 * IQR_X)  # filter_X = (X.iloc[:, 7:] <= Q3_X + 1.5 * IQR_X)
X.loc[:, "ctx-lh-bankssts-PBIF":] = X.loc[:, "ctx-lh-bankssts-PBIF":].where(filter_X, other=np.nan)  # X.iloc[:, 7:] = X.iloc[:, 7:].where(filter_X, other=np.nan)
# Replace the outliers with NAN. In this case we replace any VT values above upper bound with NaN. Once we replace outliers with NaN, we replace those with median
X_noOutlier = X.fillna(X.median())

Q1_y = y.quantile(0.25)
Q3_y = y.quantile(0.75)
IQR_y = Q3_y - Q1_y
filter_y = (y <= Q3_y + 1.5 * IQR_y)
y = y.where(filter_y, other=np.nan)
y_noOutlier = y.fillna(y.median())

# define cross-validation method to use
cv = LeaveOneOut()

y_true, y_pred = list(), list()
mae_errs = list()

fig = plt.figure()
# set palette
palette = itertools.cycle(sns.color_palette())

for train_index, test_index in cv.split(X_noOutlier):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X_noOutlier.iloc[train_index, :], X_noOutlier.iloc[test_index, :]
    y_train, y_test = y_noOutlier.iloc[train_index], y_noOutlier.iloc[test_index]

    svr = SVR()

    # Building the SVR single-output regressor.
    model_svr = MultiOutputRegressor(svr)

    # Look at parameters used by our current forest
    print('Parameters currently in use:\n')
    pprint(model_svr.estimator.get_params())

    # Base model evaluation: Fit the regressor with x and y data
    model_svr.fit(X_train, y_train)  # use this if age, weight, and auc are scaled by a constant
    base_model_accuracy = evaluate(model_svr, X_test.to_numpy(), y_test.to_numpy())

    # For other metrics, we need the predictions of the model
    # Predicting a new result
    yhat = model_svr.predict(X_test)

    # store
    y_true.append(y_test.to_numpy())
    y_pred.append(yhat)

    print('Mean Absolute Error of base model:', mean_absolute_error(y_test, yhat))
    mae_errs.append(mean_absolute_error(y_test, yhat))

    # Random search hyper-parameter tuning
    hyperparameters_random_grid = {
        'estimator__kernel': ("rbf", "linear"),
        'estimator__C': [0.01,0.1,1,10,20,30,40,50,60,70,80,90,100],
        'estimator__epsilon': [float(x) for x in np.linspace(start=0.001, stop=1, num=5)]
    }
    pprint(hyperparameters_random_grid)

    randomized_search_svr = RandomizedSearchCV(estimator=model_svr, param_distributions=hyperparameters_random_grid,
                                   n_iter=100, cv=cv,
                                   scoring='neg_mean_absolute_error', verbose=2,
                                   random_state=0, n_jobs=-1, return_train_score=True)

    # Fit the random search model
    hyperparameters_tuning = randomized_search_svr.fit(X_train, y_train)
    print('Best Parameters = {}'.format(hyperparameters_tuning.best_params_))

    tuned_model = hyperparameters_tuning.best_estimator_
    print(tuned_model.predict(X_test))

    tuned_model_accuracy = evaluate(tuned_model, X_test.to_numpy(), y_test.to_numpy())
    print('Improvement of the model with random search cv from the base model {:0.2f}%.'.format(100 * (tuned_model_accuracy - base_model_accuracy) / base_model_accuracy))

    # color
    c = next(palette)
    # sns.regplot(x=y_test, y=yhat, scatter_kws={"color": "red"}, line_kws={"color": "black"})
    sns.regplot(x=y_test, y=yhat, color=c, ci=None)

    fig.legend(df['ATLaS_ID'].unique())
    plt.title('Multi output Random Forest Regressiong - individual regression lines')
    plt.xlabel('Observed Logan VT (from AIF)')
    plt.ylabel('Predicted Logan VT')

fig = plt.figure()
sns.regplot(x=np.vstack(y_true).flatten(), y=np.vstack(y_pred).flatten(), scatter=False)
sns.scatterplot(x=np.vstack(y_true).flatten(), y=np.vstack(y_pred).flatten(), hue=ATLaS_ID, palette="bright",
                legend="full")
slope_p, intercept_p, r_value_p, p_value_p, std_err_p = stats.linregress(np.vstack(y_true).flatten(),
                                                                         np.vstack(y_pred).flatten())
plt.title('Multi output Random Forest Regression - common regression lines')
plt.xlabel('Observed Logan VT (from AIF)')
plt.ylabel('Predicted Logan VT')
plt.show()
print('Pearson correlation R and p-val are: ', r_value_p.round(5), ',', p_value_p)

# # Generate predictions for testing data
# y_pred = mor.predict(X_test)
#
# # Evaluate the regressor
# mse_one = mean_squared_error(y_test[:,0], y_pred[:,0])
# mse_two = mean_squared_error(y_test[:,1], y_pred[:,1])
# print(f'MSE for first regressor: {mse_one} - second regressor: {mse_two}')
# mae_one = mean_absolute_error(y_test[:,0], y_pred[:,0])
# mae_two = mean_absolute_error(y_test[:,1], y_pred[:,1])
# print(f'MAE for first regressor: {mae_one} - second regressor: {mae_two}')

print('debugging')

