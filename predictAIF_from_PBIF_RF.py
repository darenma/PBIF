import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats
from scipy import stats
import itertools

# Some helpful links:
# https://datascience.stackexchange.com/questions/72764/can-random-forest-regressor-or-decision-trees-handle-missing-values-and-outliers
# https://www.geeksforgeeks.org/detect-and-remove-the-outliers-using-python/
# https://stackoverflow.com/questions/27147300/matplotlib-tcl-asyncdelete-async-handler-deleted-by-the-wrong-thread
# https://www.elderresearch.com/blog/jump-start-your-modeling-with-random-forests/
# https://stackoverflow.com/questions/46257726/do-i-need-to-scale-test-data-and-dependent-variable-in-the-train-data
# https://stackoverflow.com/questions/56739932/when-doing-a-random-forest-regression-how-can-i-invert-a-standardscaler
# https://towardsdatascience.com/machine-learning-step-by-step-6fbde95c455a - Hyperparamter tuning
# https://stats.stackexchange.com/questions/419830/does-it-make-sense-to-do-cross-validation-with-a-small-sample
# https://theexpose.uk/2021/10/01/harvard-business-school-shuts-down-after-massive-covid-19-outbreak-despite-almost-all-students-being-fully-vaccinated/
# https://www.statology.org/leave-one-out-cross-validation-in-python/
# https://machinelearningmastery.com/loocv-for-evaluating-machine-learning-algorithms/
# https://medium.com/analytics-vidhya/step-by-step-guide-to-leave-one-person-out-cross-validation-with-random-forests-in-python-34b2eaefb628

input_file = "/mnt/shared_data2/snp2003/ATLaS_PD_Processed/Statistics/ATLAS_PD_MS_ML_Compiled_Features_excelForFormat.csv"
scale_all_X = 1
model_validation_method = 2  # 1: train_test_split; 2: LOOCV

def r2(x, y):
    return stats.pearsonr(x, y)[0] ** 2

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
## place the legend outside the figure/plot
# plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
# plt.show()
# sns.jointplot(df_plot_long['VT_PBIF'], df_plot_long['VT_AIF'], kind="reg", stat_func=r2)
# fig4 = plt.figure()
# sns.jointplot(df_plot_long['VT_PBIF'], df_plot_long['VT_AIF'], kind="reg")
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

# # Use following if detecting all outliers, i.ie outside of both lower and upper bound
# outliers_X = [out for stat in boxplot_stats(df.iloc[:, 8:51].astype(float)) for out in stat['fliers']]
# # Use following if detecting all outliers, i.ie outside of both lower and upper bound
# print(f"Outliers in observed PBIF are: {outliers_X}")
# outlier_X_thr = np.min(outliers_X).round(1)

# y_num = y.apply(pd.to_numeric)
# y_nan = y_num.mask(y_num > outlier_y_thr, np.nan)
# y_noOutlier = y_nan.fillna(y_num.median())

# outliers_y = [out for stat in boxplot_stats(df.iloc[:, 51:].astype(float)) for out in stat['fliers']]
# print(f"Outliers in observed AIF are: {outliers_y}")
# outlier_y_thr = np.min(outliers_y).round(1)

# Replace the outliers with NAN. In this case we replace any VT values above 10 with NaN. Once we replace outliers with NaN, we replace those with median
# X_num = X.iloc[:, 7:].apply(pd.to_numeric)
# X_nan = X_num.mask(X_num > outlier_X_thr, np.nan)
# X_noOutlier = X_nan.fillna(X_num.median())

# # Position of the Outlier
# print(np.where(X_num > outlier_X_thr))
# print(np.where(y_num > outlier_y_thr))

# Remove outliers
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

# X_noOutlier_arr = X_noOutlier.to_numpy()
# y_noOutlier_arr = y_noOutlier.to_numpy()
#
# X_noOutlier_arr = X_noOutlier_arr.astype(float)
# y_noOutlier_arr = y_noOutlier_arr.astype(float)

## If plotting orginal VT after removing outliers from y

# df_plot = df.iloc[:, np.r_[0, 8:51]].copy()
# df_plot_long = pd.melt(df_plot.astype(float), id_vars='ATLaS_ID')
# df_plot_long.columns = ['ATLaS_ID', 'regions_PBIF', 'VT_PBIF']
# df_AIF = y_noOutlier.copy()
# df_AIF_long = pd.melt(df_AIF.astype(float))
# df_AIF_long.columns = ['regions_AIF', 'VT_AIF']
# df_plot_long["regions_AIF"] = df_AIF_long["regions_AIF"]
# df_plot_long["VT_AIF"] = df_AIF_long["VT_AIF"].values
#
# # plot original data
# sns.lmplot(data=df_plot_long, x='VT_PBIF', y='VT_AIF', hue="ATLaS_ID", ci=None)
# plt.title('Observed Logan VT from PBIF and AIF - individual regression line')
# plt.xlabel('Observed Logan VT from PBIF')
# plt.ylabel('Observed Logan VT from AIF')
#
# fig = plt.figure()
# sns.regplot(data=df_plot_long, x='VT_PBIF', y='VT_AIF', line_kws={"color": "black"}, scatter=False)
# sns.scatterplot(data=df_plot_long, x="VT_PBIF", y="VT_AIF", hue="ATLaS_ID", palette="bright", legend="full")
# plt.title('Observed Logan VT from PBIF and AIF - common regression line')
# plt.xlabel('Observed Logan VT from PBIF')
# plt.ylabel('Observed Logan VT from AIF')
# ## place the legend outside the figure/plot
# # plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
# # plt.show()
# # sns.jointplot(df_plot_long['VT_PBIF'], df_plot_long['VT_AIF'], kind="reg", stat_func=r2)
# # fig4 = plt.figure()
# # sns.jointplot(df_plot_long['VT_PBIF'], df_plot_long['VT_AIF'], kind="reg")
# slope_o, intercept_o, r_value_o, p_value_o, std_err_o = stats.linregress(df_plot_long['VT_PBIF'], df_plot_long['VT_AIF'])
# print('Pearson correlation R and p-val are: ', r_value_o.round(5), ',', p_value_o)

# enumerate splits, for LOOCV method

if model_validation_method == 1:  # train_test_split
    # Train/test split. For now not using split data coz of smaller dataset
    # X_train, X_test, y_train, y_test = train_test_split(X_noOutlier_arr, y_noOutlier_arr, test_size=0.01, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X_noOutlier, y_noOutlier, test_size=0.01, random_state=0)

    # sc = StandardScaler()
    # # Feature Scaling before the split
    # if scale_all_X == 1:
    #     X_train_scaled = sc.fit_transform(X_noOutlier_arr)
    #     X_test_scaled = sc.fit_transform(X_noOutlier_arr)
    #     y_train = y_noOutlier_arr
    #     y_test = y_noOutlier_arr
    # else:
    #     # Feature Scaling after the split
    #     X_train_scaled = sc.fit_transform(X_train)
    #     X_test_scaled = sc.transform(X_test)

    # create random forest regressor object
    regressor = RandomForestRegressor(n_estimators=200, random_state=0)

    # fit the regressor with x and y data
    regressor.fit(X_train_scaled, y_train)  # use this if i/p features are standard scaled after test_train_split

    # Default metric is R2 for regression, which can be accessed by score()
    regressor.score(X_test, y_test)

    # For other metrics, we need the predictions of the model
    # Predicting a new result
    if scale_all_X == 1:
        # y_pred = regressor.predict(X_test_scaled)
        y_pred = regressor.predict(X_test)
    else:
        # y_pred = regressor.predict(X_test_scaled.reshape(1, -1))
        y_pred = regressor.predict(X_test.reshape(1, -1))

    # Print observed and predicted output
    print(np.c_[np.transpose(y_test), np.transpose(y_pred)])

    # Calculate the absolute errors
    errors = abs(y_pred - y_test)
    # Print out the mean absolute error (mae)
    if scale_all_X == 1:
        print('Mean Absolute Error:', np.mean(errors, 1).round(2), 'degrees.')
    else:
        print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / y_test)
    # Calculate and display accuracy
    if scale_all_X == 1:
        accuracy = 100 - np.mean(mape, 1)
    else:
        accuracy = 100 - np.mean(mape)
    if scale_all_X == 1:
        print('Accuracy:', accuracy.round(2), '%.')
    else:
        print('Accuracy:', round(accuracy, 2), '%.')

    if scale_all_X == 1:
        print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred, multioutput='raw_values'))
        print('Mean Squared Error:', mean_squared_error(y_test, y_pred, multioutput='raw_values'))
        print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred, multioutput='raw_values')))
    else:
        print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
        print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
        print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))

    # plot predicted data
    fig = plt.figure()
    if scale_all_X == 1:
        # ATLaS_ID = df_plot_long['ATLaS_ID'].to_numpy()
        sns.regplot(y_test.flatten(), y_pred.flatten(), scatter_kws={"color": "red"}, line_kws={"color": "black"},
                    scatter=False)
        sns.scatterplot(y_test.flatten(), y_pred.flatten(), hue=ATLaS_ID, palette="bright", legend="full")
        slope_p, intercept_p, r_value_p, p_value_p, std_err_p = stats.linregress(y_test.flatten(),
                                                                                 y_pred.flatten())
        print('Pearson correlation R and p-val are: ', r_value_p.round(5), ',', p_value_p)
    else:
        sns.regplot(y_test, y_pred, scatter_kws={"color": "red"}, line_kws={"color": "black"})
    plt.title('Multi output Random Forest Regression')
    plt.xlabel('Observed Logan VT (from AIF)')
    plt.ylabel('Predicted Logan VT')
    plt.show()

    # # Tune hyperparameters
    # n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]max_features = ['log2', 'sqrt']max_depth = [int(x) for x in np.linspace(start = 1, stop = 15, num = 15)]min_samples_split = [int(x) for x in np.linspace(start = 2, stop = 50, num = 10)]min_samples_leaf = [int(x) for x in np.linspace(start = 2, stop = 50, num = 10)]bootstrap = [True, False]param_dist = {'n_estimators': n_estimators,
    #                'max_features': max_features,
    #                'max_depth': max_depth,
    #                'min_samples_split': min_samples_split,
    #                'min_samples_leaf': min_samples_leaf,
    #                'bootstrap': bootstrap}
    #
    # rs = RandomizedSearchCV(rfc_2,
    #                         param_dist,
    #                         n_iter = 100,
    #                         cv = 3,
    #                         verbose = 1,
    #                         n_jobs=-1,
    #                         random_state=0)

elif model_validation_method == 2:  # LOOCV
    # define cross-validation method to use
    cv = LeaveOneOut()

    # # create random forest regressor object
    # regressor = RandomForestRegressor(n_estimators=200, random_state=0)
    #
    # # evaluate model
    # scores = cross_val_score(regressor, X_noOutlier, y_noOutlier, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    #
    # # force positive
    # scores = np.absolute(scores)
    #
    # # report performance
    # print('MAE: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

    y_true, y_pred = list(), list()
    mae_errs = list()

    fig = plt.figure()
    # set palette
    palette = itertools.cycle(sns.color_palette())

    for train_index, test_index in cv.split(X_noOutlier):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X_noOutlier.iloc[train_index, :], X_noOutlier.iloc[test_index, :]
        y_train, y_test = y_noOutlier.iloc[train_index], y_noOutlier.iloc[test_index]

        # create random forest regressor object
        regressor = RandomForestRegressor(n_estimators=200, random_state=0)

        # fit the regressor with x and y data
        regressor.fit(X_train, y_train)  # use this if age, weight, and auc are scaled by a constant

        # # Default metric is R2 for regression, which can be accessed by score(). Below is not working because of one datapoint
        # regressor.score(X_test, y_test)

        # For other metrics, we need the predictions of the model
        # Predicting a new result
        yhat = regressor.predict(X_test)

        # store
        y_true.append(y_test.to_numpy())
        y_pred.append(yhat)

        print('Mean Absolute Error:', mean_absolute_error(y_test, yhat))
        mae_errs.append(mean_absolute_error(y_test, yhat))

        # color
        c = next(palette)
        # sns.regplot(x=y_test, y=yhat, scatter_kws={"color": "red"}, line_kws={"color": "black"})
        sns.regplot(x=y_test, y=yhat, color=c, ci=None)

    # # Calculate the absolute errors
    # errors = abs(y_pred - y_test)
    # # Print out the mean absolute error (mae)
    # print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
    #
    # # Calculate mean absolute percentage error (MAPE)
    # mape = 100 * (errors / y_test)
    # # Calculate and display accuracy
    # accuracy = 100 - np.mean(mape)
    # print('Accuracy:', round(accuracy, 2), '%.')
    #
    # print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
    # print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
    # print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))

    # plot predicted data
    # fig = plt.figure()
    # sns.regplot(y_test, y_pred, scatter_kws={"color": "red"}, line_kws={"color": "black"})
    fig.legend(df['ATLaS_ID'].unique())
    plt.title('Multi output Random Forest Regressiong - individual regression lines')
    plt.xlabel('Observed Logan VT (from AIF)')
    plt.ylabel('Predicted Logan VT')

    fig = plt.figure()
    sns.regplot(x=np.vstack(y_true).flatten(), y=np.vstack(y_pred).flatten(), scatter=False)
    sns.scatterplot(x=np.vstack(y_true).flatten(), y=np.vstack(y_pred).flatten(), hue=ATLaS_ID, palette="bright", legend="full")
    slope_p, intercept_p, r_value_p, p_value_p, std_err_p = stats.linregress(np.vstack(y_true).flatten(), np.vstack(y_pred).flatten())
    plt.title('Multi output Random Forest Regression - common regression lines')
    plt.xlabel('Observed Logan VT (from AIF)')
    plt.ylabel('Predicted Logan VT')

    plt.show()

print('debug')
