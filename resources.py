import itertools
import warnings
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.linear_model as linear_model
import statsmodels.api as sm
from matplotlib.cbook import boxplot_stats
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import (
    train_test_split,
    LeaveOneOut,
    cross_val_score,
    RandomizedSearchCV,
    GridSearchCV,
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from varname import nameof

pd.options.mode.chained_assignment = None  # default='warn'
sns.set(rc={"figure.figsize":(7, 7)})
warnings.filterwarnings("ignore", category=UserWarning)

# subjs = ['1001', '1002', '1003', '1004', '1005', '1007', '1009',
#          '1013', '1015', '1020', '1022', '1023', 'PP', 'MM', 'FR', 'KT', 'JD']

updated = pd.read_csv("/Users/darenma/Downloads/PureDemographics.csv")

aif_new = updated[['ATLaS_ID', 'Age', 'Sex',
                   'NET dose (mCi)', 'Height (Feet, inches)', 'Weight (Kgs)', "Genotype"]].copy()
# new_target = pd.read_csv("../Interpolate_0203_slinear.csv")
# new_y = np.array(new_target.sum().T[2:])

# df = aif_new.copy()
# df.head()

# df.columns = ['ATLaS_ID', 'Age', 'Sex', 'Dose', 'Height',
#        'Weight', 'Genotype']
# df.loc[:, "auc"] = new_y


def break_left_right_and_join(pbif):
    
    # Separate Left and Right brain
    to_drop = ['index', 'Region', 'RegionID']
    
    pbif_left = pbif.loc[(pbif.Region.str.contains(
        "-lh-") | pbif.Region.str.contains("Left-")), :].reset_index().drop(columns=to_drop)
    pbif_right = pbif.loc[(pbif.Region.str.contains(
        "-rh-") | pbif.Region.str.contains("Right-")), :].reset_index().drop(columns=to_drop)
    
    # Rename the columns
    pl = pbif_left.columns
    pbif_left.columns = [str(name)+"_L" for name in pl]
    pr = pbif_right.columns
    pbif_right.columns = [str(name)+"_R" for name in pr]
    
    # Concat those together
    df_pbif = pd.concat((pbif_left,pbif_right), axis=1)
    
    return df_pbif.copy()


# Helper Function for metric reporting.
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


def run_gbr_model(X, y, docstring=""):
# GradientBoostingRegressor
    y_true_base, y_pred_base = list(), list()
    mae_errs_base = list()
    cv = LeaveOneOut()
    y_true, y_pred = list(), list()
    mae_errs = list()

    fig = plt.figure()
    palette = itertools.cycle(sns.color_palette())

    for train_index, test_index in cv.split(X):
#         print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        gbr = GradientBoostingRegressor(
                                random_state=42, 
                                min_samples_leaf=2, 
                                n_estimators=400, 
                                max_features=20, 
                                max_leaf_nodes=4,
                                min_samples_split=2,
                                learning_rate=0.05,
        )
        model_gbr = MultiOutputRegressor(gbr)
        # Look at parameters used by our current forest
    #     print('Parameters currently in use:\n')
    #     pprint(model_gbr.estimator.get_params())
        # Base model evaluation: Fit the regressor with x and y data
        model_gbr.fit(X_train.values, y_train.values)  # use this if age, weight, and auc are scaled by a constant
        base_model_accuracy = evaluate(model_gbr, X_test.to_numpy(), y_test.to_numpy())

        yhat = model_gbr.predict(X_test)

        # store
        y_true_base.append(y_test.to_numpy())
        y_pred_base.append(yhat)

        print('MAE:', mean_absolute_error(y_test, yhat))
        mae_errs_base.append(mean_absolute_error(y_test, yhat))

        c = next(palette)
        # sns.regplot(x=y_test, y=yhat, scatter_kws={"color": "red"}, line_kws={"color": "black"})
    #     sns.set(rc={"figure.figsize":(7, 7)})
        sns.regplot(x=yhat.reshape(-1), y=np.array(y_test).reshape(-1), color=c, ci=None)
        fig.legend(df['ATLaS_ID'].unique())
        plt.title('Multi output GBR - individual regression lines')
        plt.ylabel('Observed Logan VT (from AIF)')
        plt.xlabel('Predicted Logan VT')
        plt.axis('equal')
        plt.savefig(f"GBR-Lineplot{docstring}.png")
        

    fig = plt.figure()
    sns.set(rc={"figure.figsize":(7, 7)})
    sns.regplot(x=np.vstack(y_pred_base).flatten(), y=np.vstack(y_true_base).flatten(), scatter=False)
    sns.scatterplot(x=np.vstack(y_pred_base).flatten(), y=np.vstack(y_true_base).flatten(), hue=ATLaS_ID, palette="bright", legend="full")
    slope_p, intercept_p, r_value_p, p_value_p, std_err_p = stats.linregress(np.vstack(y_true_base).flatten(), np.vstack(y_pred_base).flatten())
    plt.title('Multi output GBR - common regression lines')
    plt.ylabel('Observed Logan VT (from AIF)')
    plt.xlabel('Predicted Logan VT')
    plt.axis('equal')
    plt.savefig(f"GBR-Scatter{docstring}.png")
    
    print('Pearson correlation R and p-val are: ', r_value_p.round(5), ',', p_value_p)

    return model_gbr, mae_errs_base


def lou_model(new_df, paralist=["Age", "lambda"], y_column="auc", mode="linear", annotations=1, verbose=0):
# GradientBoostingRegressor

    y_true_base, y_pred_base = list(), list()
    mae_errs_base = list()
    cv = LeaveOneOut()
    print("Leave-One-Out model predicting ...")
    y_true, y_pred = list(), list()
    mae_errs = list()
    y = new_df[y_column]
    X = new_df[paralist]
    
    if mode == "linear":
        X = sm.add_constant(X)
    
    for train_index, test_index in cv.split(X):
        if verbose == 1:
            print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        if mode == "linear":
            model_s = sm.OLS(y_train, X_train)
            results_s = model_s.fit()
            X_test = sm.add_constant(X_test)
            results_s.summary()
            yhat = results_s.predict(X_test)
        
        elif mode == "gbr":
            gbr = GradientBoostingRegressor(random_state=42, 
                                min_samples_leaf=2, 
                                n_estimators=400, 
                                max_features=2, 
                                max_leaf_nodes=4,
                                min_samples_split=2,
                                learning_rate=0.05,
                                       )

            gbr.fit(X_train.values, y_train.values)  
            yhat = gbr.predict(X_test)
        else:
            print("Not A Valid Model Type. Stop.")
            return 0, 0, 0
        
        y_true_base.append(y_test.to_numpy())
        y_pred_base.append(yhat)

#         print('Mean Absolute Error of base model:', mean_absolute_error(y_test, yhat))
        mae_errs_base.append(mean_absolute_error(y_test, yhat))
    plt.scatter(y_true_base, y_pred_base)
    plt.xlabel = "True AUCs"
    plt.ylabel = "Predicted AUCs"
    for i, label in enumerate(annotations):
        plt.text(y_true_base[i], y_pred_base[i],label)
#     plt.lineplot()
#     plt.set_aspect('equal')
    xpoints = ypoints = plt.xlim()
    plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=3, scalex=False, scaley=False)
    plt.show()
    if mode == "linear":
        return model_s, mae_errs_base, y_true_base, y_pred_base
    return gbr, mae_errs_base, y_true_base, y_pred_base
