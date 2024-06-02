import os
import pandas as pd
import re
import chardet
import sqlalchemy
import pickle
from sqlalchemy import create_engine
from datetime import datetime
import mysql.connector
import joblib
from sklearn.inspection import permutation_importance
import shap
from sklearn.model_selection import learning_curve
from mysql.connector import FieldType
from sklearn.metrics import roc_curve, auc, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import seaborn as sns
import unidecode
from bs4 import BeautifulSoup
from mysql.connector.cursor import MySQLCursor
import sys
import requests
from IPython.display import clear_output
import re
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, \
    train_test_split, GroupShuffleSplit, StratifiedGroupKFold
import statistics
from sklearn.metrics import precision_score, classification_report, recall_score, ConfusionMatrixDisplay, \
    precision_recall_fscore_support, confusion_matrix, f1_score, make_scorer
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def get_info_col(colname, f):
    url = "https://adni.loni.usc.edu/data-dictionary-search/?q=" + colname

    response = requests.get(url)

    soup = BeautifulSoup(response.text, features="lxml")
    if len(soup.find_all("table")) > 0:
        df = pd.read_html(response.text)
        df = df[0]
        df1 = df[df["Term"] == colname]
        df1 = df1[df1["Table"] == f]
        df2 = df[df["Term"] == colname.upper()]
        df2 = df2[df2["Table"] == f]
        if df1.equals(df2):
            return df1
        return pd.concat([df1, df2])
    else:
        return pd.DataFrame()


def make_dates(df_og):
    df = df_og.copy()
    for i in df.columns:
        df[i] = pd.to_datetime(df[i])
    return df


def supp_row(df, per, verbose=False, excet=[]):
    if verbose:
        p = (df.isna().sum().sum()) / (df.shape[0] * df.shape[1])
        print("total nan values before: " + str(df.isna().sum().sum()))
        print("percentage of nan: " + str(p * 100) + "%")
    if df.isna().sum().sum() == 0:
        print("no need to delete rows")
        return df

    df_copy = df.copy()
    r = pd.isnull(df_copy).sum(axis=1) / len(list(df_copy.columns)) > per
    print(r)
    for exclude in excet:
        r.at[exclude] = False
    print("\n\nwe are going to delete these ROWS due to too many nan values: \n")
    display(df_copy.loc[r, :])
    print("\n")
    df_copy = df_copy.loc[~r, :]

    if verbose:
        p = (df_copy.isna().sum().sum()) / (df_copy.shape[0] * df_copy.shape[1])
        print("total nan values after: " + str(df_copy.isna().sum().sum()))
        print("percentage of nan: " + str(p * 100) + "%")
    return df_copy


def supp_col(df, per, verbose=False, excet=[]):
    if verbose:
        p = (df.isna().sum().sum()) / (df.shape[0] * df.shape[1])
        print("total nan values before: " + str(df.isna().sum().sum()))
        print("percentage of nan: " + str(p * 100) + "%")
    if df.isna().sum().sum() == 0:
        print("no need to delete columns")
        return df

    df_copy = df.copy()
    s = pd.isnull(df_copy).sum() / len(df_copy.index) > per
    for exclude in excet:
        s.at[exclude] = False

    print("\n\nwe are going to delete these COLUMNS due to too many nan values: \n" + str(
        df_copy.columns[s].tolist()) + "\n")
    df_copy = df_copy[df_copy.columns[~s]]

    if verbose:
        p = (df_copy.isna().sum().sum()) / (df_copy.shape[0] * df_copy.shape[1])
        print("total nan values after: " + str(df_copy.isna().sum().sum()))
        print("percentage of nan: " + str(p * 100) + "%")
    return df_copy


def date_within(df_accurate, nbr_days, list_dates, v=False):
    df = df_accurate.copy()
    for i in range(len(list_dates)):
        for j in range(i + 1, len(list_dates)):
            d1 = list_dates[i]
            d2 = list_dates[j]
            if v:
                print(d1 + "_" + d2)
            df.dropna(subset=[d1, d2], how="any", inplace=True)
            df[d1 + "_" + d2] = df[d1] - df[d2]
            df[d1 + "_" + d2] = df[d1 + "_" + d2].abs()
            df_copy = df.copy()
            df = df[df[d1 + "_" + d2].dt.days <= nbr_days]
            if v:
                print("those lines were more than a year apart")
                missed = list(set(df_copy.index).difference(set(df.index)))
                if "id_patient" in df_copy.columns:
                    display(
                        (df_copy.loc[missed, :])[["id_patient", d1, d2, d1 + "_" + d2]].sort_values(by=d1 + "_" + d2,
                                                                                                    ascending=True))
                else:
                    display((df_copy.loc[missed, :])[["RID", d1, d2, d1 + "_" + d2]].sort_values(by=d1 + "_" + d2,
                                                                                                 ascending=True))
            df.drop(columns=[d1 + "_" + d2], inplace=True)
    df.drop_duplicates(inplace=True)
    return df


def multiclass_score_matrix(m, targets, score):
    TP = np.diag(m)
    FP = np.sum(m, axis=0) - TP
    FN = np.sum(m, axis=1) - TP
    tp = 0
    fp = 0
    fn = 0
    for i in targets:
        tp += TP[i]
        fp += FP[i]
        fn += FN[i]
    if score == 'recall':
        if (fn + tp) != 0:
            return tp / (fn + tp)
    elif score == 'f1':
        if ((2 * tp) + fn + fp) != 0:
            return (2 * tp) / ((2 * tp) + fn + fp)
    return 0


def expand_diag(df):
    id_patient = df["id_patient"].unique()
    print("######################### NEEDED ? #########################################")
    for i in id_patient:
        r = df.loc[df["id_patient"] == i]
        if len(r.index[r['diagnostic'] == "amy- cn"].tolist()) > 1:
            print("HI")
            display(r[['diagnostic', 'date_feat', 'date_irm']])
            print(r.index[r['diagnostic'] == "amy- cn"].tolist())
    print("######################### NEEDED ? #########################################")
    return df


def print_diag_rep(df):
    diag = df["diagnostic"].unique()
    for i in diag:
        print(str(i) + " have " + str(len(df[df["diagnostic"] == i])))


def get_hemispher(df, start_left, start_right):
    left = pd.DataFrame()
    right = pd.DataFrame()
    both = pd.DataFrame()
    for i in df.columns:
        print(i)
        if i.startswith(start_left):
            left = pd.concat([left, df[[i]]], axis=1)
            left.rename(columns={i: i.split(start_left)[-1]}, inplace=True)
        elif i.startswith(start_right):
            right = pd.concat([right, df[[i]]], axis=1)
            right.rename(columns={i: i.split(start_right)[-1]}, inplace=True)
        else:
            both = pd.concat([both, df[[i]]], axis=1)

    left["hemispher"] = "L"
    right["hemispher"] = "R"
    both["hemispher"] = "B"
    return pd.concat([left, right, both])


def multiclass_score(ground_truth, pred, targets=[1, 3], score=roc_auc_score, binary=True, labels=[0, 1, 2, 3]):
    g = ground_truth.copy()
    p = pred.copy()
    if score == f1_score and binary:
        if not isinstance(g, pd.Series):
            g = pd.Series(g)
        if not isinstance(p, pd.Series):
            p = pd.Series(p)
        for i in set(g).union(set(p)):
            if i in targets:
                g.replace(i, 1, inplace=True)
                p.replace(i, 1, inplace=True)
            else:
                g.replace(i, 0, inplace=True)
                p.replace(i, 0, inplace=True)
        return score(g, p, average='binary', labels=labels)
    elif score == roc_auc_score:
        return score(g, p, multi_class="ovo", average="weighted", labels=labels)
    else:
        return score(g, p, average="weighted", labels=labels)


def grid_searches(X, y_cop, model, scoring_call, param_grid, path, name, v=4, basic_par={}, cv_grid=None,
                  cv_out_grid=None, group=None):
    m = model(**basic_par)
    y = y_cop["diagnostic"]

    train_index, test_index = list(cv_out_grid.split(X, y, group))[0]

    X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]

    X_test.copy().assign(y=y_test).to_csv(path + "test_train_set/" + name + "_test_set.csv")
    X_train.copy().assign(y=y_train).to_csv(path + "test_train_set/" + name + "_train_set.csv")

    grid = GridSearchCV(m, param_grid, refit=True, scoring="roc_auc_ovo_weighted", verbose=v, cv=cv_grid)
    grid.fit(X_train, y_train, groups=group[group.index.isin(X_train.index)])

    with open(path + "gridsearch_models/" + name + '.pickle', 'wb') as f:
        pickle.dump(grid, f)
        
    
    if len(y.unique()) == 2:
        y_pred_proba = grid.best_estimator_.predict_proba(X_test)[:, 1]
    else:
        y_pred_proba = grid.predict_proba(X_test)

    return grid.best_params_, grid.best_score_, roc_auc_score(y_test, y_pred_proba, multi_class="ovo", average="weighted"), grid.best_estimator_


def custom_cross_val(X, y_cop, model, cv, targets, scoring, binary, labels, group):
    y = y_cop["diagnostic"]

    splitting = cv.split(X, y, group)

    conf_matrix_list_of_arrays = []
    scores = []
    for train_index, test_index in splitting:

        X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[
            test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        conf_matrix = confusion_matrix(y_test, y_pred, labels=labels)
        if scoring == roc_auc_score:
            if len(y_test.unique()) == 2:
                y_pred = model.predict_proba(X_test)[:, 1]
            else:
                y_pred = model.predict_proba(X_test)

        conf_matrix_list_of_arrays.append(conf_matrix)
        scores.append(multiclass_score(y_test, y_pred, targets=targets, score=scoring, binary=binary, labels=labels))

    mean_of_conf_matrix_arrays = np.mean(conf_matrix_list_of_arrays, axis=0)
    sum_of_conf_matrix_arrays = np.sum(conf_matrix_list_of_arrays, axis=0)
    return mean_of_conf_matrix_arrays, sum_of_conf_matrix_arrays, scores, model.classes_


def build_all_datasets(df, apoe=False):
    dbs = {}
    amys = ["amy_2", "amy_3"]
    atlases = ['ba_exvivo', 'destrieux', 'dkt', 'wmparc', 'segsouscort', 'desikan']
    candidates = [set(['ba_exvivo']), set(['destrieux']), set(['dkt']), set(['wmparc']), set(['segsouscort']),
                  set(['desikan'])]
    combination_atlas = candidates
    """while len(candidates) > 0:
        new_can = []
        for atlas in atlases:
            for can in candidates:
                if atlas not in can:
                    can_cop = can.copy()
                    can_cop.add(atlas)
                    new_can.append(can_cop)
        combination_atlas += new_can 
        candidates = new_can"""

    for amy in amys:
        for atlas in atlases:
            df_cop = df.copy()
            # we take amy
            df_cop.drop(columns=[amys[(amys.index(amy) + 1) % 2]], inplace=True)
            df_cop = pd.get_dummies(df_cop, columns=[amy])

            # we take IRM
            atlas_to_drop = atlases[:]
            atlas_to_drop.remove(atlas)
            drop_col = [columns for dropping in atlas_to_drop for columns in df_cop.columns if dropping in columns]
            df_cop.drop(columns=drop_col, inplace=True)

            if apoe:
                dbs[amy + '_' + atlas + "_brut"] = df_cop.copy()
                df_cop["APOE_nbr_allelle_4"] = df_cop.apply(lambda x: x['APOE'].str.count('4'), axis=1)
                df_cop.drop(columns=["APOE"])
                dbs[amy + '_' + atlas + "_nbrallelle"] = df_cop.copy()
            else:
                dbs[amy + '_' + atlas] = df_cop.copy()
    return dbs


def reload_latest_gridsearches(path, only_estimator=True, date_form="%d-%m-%Y_%Hh%Mm%Ss", final_path=None):
    files = os.listdir(path)
    # Filtering only the files.
    dir_grid = max([datetime.strptime(f, date_form) for f in files if os.path.isdir(path + '/' + f)]).strftime(
        "%d-%m-%Y_%Hh%Mm%Ss")
    if final_path == None:
        final_path = path + "/" + dir_grid
    grids_list = {}
    for grids in os.listdir(final_path + "/gridsearch_models"):
        with open(final_path + "/gridsearch_models" + "/" + grids, 'rb') as f:
            grid = pickle.load(f)
        l = grids.split("_")
        stratml = l[0]
        k = l[1]
        if "onehotscaled" in l:
            options = "_".join(l[2:l.index("onehotscaled")])
        else:
            options = "_".join(l[2:l.index("score")])

        if stratml not in grids_list:
            grids_list[stratml] = {}
        if k not in grids_list[stratml]:
            grids_list[stratml][k] = {}

        if only_estimator:
            grids_list[stratml][k][options] = grid.best_estimator_
        else:
            grids_list[stratml][k][options] = grid

    return grids_list, final_path


def plot_disparities(X, y, col, target):
    X_cop = X.copy()
    X_y = X_cop.join(y)

    # Créer l'histogramme
    plt.hist((X_y[X_y[target] == 1])[col], color='skyblue', edgecolor='black')

    # Ajouter des étiquettes et un titre
    plt.xlabel('Valeurs')
    plt.ylabel('Fréquence')
    plt.title('Histogramme valeurs positives pour colonne ' + col)

    # Afficher l'histogramme
    plt.show()
    plt.clf()

    plt.hist((X_y[X_y[target] == 0])[col], color='skyblue', edgecolor='black')

    # Ajouter des étiquettes et un titre
    plt.xlabel('Valeurs')
    plt.ylabel('Fréquence')
    plt.title('Histogramme valeurs négatives pour colonne ' + col)

    # Afficher l'histogramme
    plt.show()
    plt.clf()

    plt.title("répartition pour " + col)
    plt.plot((X_y[X_y[target] == 0])[col].value_counts().sort_index().index,
             (X_y[X_y[target] == 0])[col].value_counts().sort_index().values, '--', marker='o')
    plt.plot((X_y[X_y[target] == 1])[col].value_counts().sort_index().index,
             (X_y[X_y[target] == 1])[col].value_counts().sort_index().values, '--', marker='o')
    plt.xlabel(col)
    plt.ylabel("# of occurences")
    plt.show()
    plt.clf()

    plt.title("répartition pour " + col)
    plt.scatter((X_y[X_y[target] == 0])[col].value_counts().sort_index().index,
                (X_y[X_y[target] == 0])[col].value_counts().sort_index().values, marker='o')
    plt.scatter((X_y[X_y[target] == 1])[col].value_counts().sort_index().index,
                (X_y[X_y[target] == 1])[col].value_counts().sort_index().values, marker='o')
    plt.xlabel(col)
    plt.ylabel("# of occurences")
    plt.show()
    plt.clf()


def plot_scores(path, plots=["best_scores_in_grid", "score_on_test", "score_on_cross_val"], v=False):
    files = os.listdir(path)
    # Filtering only the files.

    dir_grid = max(
        [datetime.strptime(f, "%d-%m-%Y_%Hh%Mm%Ss") for f in files if os.path.isdir(path + '/' + f)]).strftime(
        "%d-%m-%Y_%Hh%Mm%Ss")

    for scores in os.listdir(path + "/" + dir_grid):
        if "best_scores_in_grid" in scores and "best_scores_in_grid" in plots:
            strat_df_best_score = pd.read_csv(path + "/" + dir_grid + "/" + scores, index_col=0)
        if "score_on_test" in scores and "score_on_test" in plots:
            strat_df_score_test = pd.read_csv(path + "/" + dir_grid + "/" + scores, index_col=0)
        if "score_on_cross_val.csv" in scores and "score_on_cross_val" in plots:
            strat_df_score_cross_val = pd.read_csv(path + "/" + dir_grid + "/" + scores, index_col=0)

    fig, axs = plt.subplots(len(plots), 1, figsize=(20, 12))
    cmap = sns.cm.rocket_r

    for ind, p in enumerate(plots):
        if len(plots) > 1:
            ax = axs[ind]
        else:
            ax = axs
        if p == "best_scores_in_grid":
            if v:
                print("the best score in grid search:")
                display(strat_df_best_score)
            sns.heatmap(strat_df_best_score, annot=True, ax=ax, cmap=cmap)
            ax.set_title("best score in grid search")
        elif p == "score_on_test":
            if v:
                print("the score on test: ")
                display(strat_df_score_test)
            sns.heatmap(strat_df_score_test, annot=True, ax=ax, cmap=cmap)
            ax.set_title("best score with test out of gridsearch")

        elif p == "score_on_cross_val":
            if v:
                print("the score on cross_val: ")
                display(strat_df_score_cross_val)
            sns.heatmap(strat_df_score_cross_val, annot=True, ax=ax, cmap=cmap)
            ax.set_title("best score with cross validation")

    fig.savefig(path + "/" + dir_grid + "/scores_fig.png")


"""
def plot_learning_curves(new_dbs, path, target, nsplit, nrep, labels, v=False, scoring=f1_score, target_classes=[1,3], binary=True):
    cv = StratifiedGroupKFold(n_splits=nsplit)
    grids_list, final_path = reload_latest_gridsearches(path)
    for stratml in grids_list:
        for k in grids_list[stratml]:
            for options in grids_list[stratml][k]:
                if len(target) == 0 or k+"_"+options in target:
                    print(stratml + " with value "+k+"\noption:"+options+"\n")

                    # Créer un classifieur SVM
                    estimator = grids_list[stratml][k][options]
                    print(estimator)

                    X, y, fs, g = new_dbs[options]
                    mean_conf_mat, sum_conf_mat, scores, classes = custom_cross_val(X, y, estimator, labels, folds=4, targets=target_classes, scoring=scoring, cv_given=cv, group=g)
                    print(sum(scores)/len(scores))


                    # Définir les tailles des ensembles d'entraînement à tester
                    train_sizes = np.linspace(0.1, 1.0, 100)

                    # Calculer la courbe d'apprentissage
                    
                    response_method = "predict"
                    if scoring == roc_auc_score:
                        response_method = "predict_proba"
                    train_sizes, train_scores, test_scores = learning_curve(
                        estimator, X, y["diagnostic"], train_sizes=train_sizes, cv=cv, 
                        scoring=make_scorer(multiclass_score, greater_is_better=True, targets=[1, 3], score=scoring, binary=binary),
                        groups=g, response_method=response_method)
                    #print("train scores: "+str(train_scores))
                    #print("test scores: "+str(test_scores))

                    # Calculer les moyennes et les écarts-types des scores d'apprentissage et de validation
                    train_scores_mean = np.mean(train_scores, axis=1)
                    train_scores_std = np.std(train_scores, axis=1)
                    test_scores_mean = np.mean(test_scores, axis=1)
                    test_scores_std = np.std(test_scores, axis=1)

                    # Tracer la courbe d'apprentissage
                    plt.figure()
                    plt.title("Courbe d'apprentissage SVM")
                    plt.xlabel("Taille de l'ensemble d'entraînement")
                    plt.ylabel("Score")
                    plt.grid()

                    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                                     train_scores_mean + train_scores_std, alpha=0.1,
                                     color="r")
                    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
                    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                             label="Score d'entraînement")
                    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                             label="Score de validation")

                    plt.legend(loc="best")
                    if len(target) > 0:
                        print("saving learning curves in "+final_path)
                        plt.savefig(final_path+"/learning_curve_"+k+"_"+options+".png")
                    if v:
                        plt.show()
                    plt.clf()
"""


def XAI_analyses(new_dbs, path, target, scoring, v=False, nsplits_exp=5, shap=True):
    grids_list, final_path = reload_latest_gridsearches(path)
    if len(target) == 0:
        raise Exception("target needs to be filled")
    for stratml in grids_list:
        for k in grids_list[stratml]:
            for options in grids_list[stratml][k]:
                if k + "_" + options in target:
                    print(stratml + " with value " + k + "\noption:" + options + "\n")

                    # Créer un classifieur SVM
                    clf = grids_list[stratml][k][options]

                    X, y, fs, group = new_dbs[options]

                    train_index, test_index = list(StratifiedGroupKFold(n_splits=nsplits_exp).split(X, y, group))[0]

                    X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], \
                                                       y.iloc[test_index]
                    clf.fit(X_train, y_train)

                    feature_names = clf.feature_names_in_

                    result = permutation_importance(
                        clf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2, scoring=scoring
                    )
                    forest_importances = pd.DataFrame({"mean": result.importances_mean, "std": result.importances_std},
                                                      index=feature_names)
                    forest_importances = forest_importances.sort_values(by=["mean"], ascending=False).head(5)

                    fig, ax = plt.subplots(figsize=(10, 6))

                    forest_importances["mean"].plot.bar(yerr=forest_importances["std"], ax=ax)
                    ax.set_title("Top 5 Feature Permutation")
                    ax.set_ylabel("Mean score decrease")
                    fig.tight_layout()
                    print("saving feature permutation in " + final_path)
                    plt.savefig(final_path + "/feature_permutation_" + stratml + "_" + k + "_" + options + ".png")
                    if v:
                        plt.show()
                    plt.clf()

                    if shap:
                        # Initialize explainer
                        explainer = shap.KernelExplainer(clf.predict, X_train)

                        # Calculate SHAP values
                        shap_values = explainer.shap_values(X_test)

                        # Get feature importance
                        shap.summary_plot(shap_values, X, plot_type="bar", max_display=5, show=False)
                        print("saving SHAP in " + final_path)
                        plt.savefig(final_path + "/shap_" + stratml + "_" + k + "_" + options + ".png")
                        if v:
                            shap.summary_plot(shap_values, X, plot_type="bar", max_display=5, show=True)
                        plt.clf()


# voir pour multiclass
def plot_roc_curve(new_dbs, path, target, cv):
    grids_list, final_path = reload_latest_gridsearches(path)
    if len(target) == 0:
        raise Exception("target needs to be filled")
    for stratml in grids_list:
        for k in grids_list[stratml]:
            for options in grids_list[stratml][k]:
                if k + "_" + options in target:
                    print(stratml + " with value " + k + "\noption:" + options + "\n")

                    # Créer un classifieur SVM
                    clf = grids_list[stratml][k][options]

                    X, y, fs, g = new_dbs[options]

                    train_index, test_index = list(cv.split(X, y, g))[0]

                    X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], \
                                                       y.iloc[test_index]
                    clf.fit(X_train, y_train)

                    # df_probs = pd.DataFrame(clf.predict_proba(X_test), columns=[i for i in clf.classes_])
                    y_pred_proba = clf.predict_proba(X_test)[:, 1]

                    # Calculate ROC curve
                    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
                    roc_auc = auc(fpr, tpr)
                    # Plot the ROC curve
                    plt.figure()
                    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
                    plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve')
                    plt.legend()
                    plt.savefig(final_path + "/roccurve_" + stratml + "_" + k + "_" + options + ".png")
                    plt.show()


def plot_confusion_matrices(new_dbs, path, target, cv, scoring, binary, labels, target_classes, summat=True):
    grids_list, final_path = reload_latest_gridsearches(path)
    if len(target) == 0:
        raise Exception("target needs to be filled")
    for stratml in grids_list:
        for k in grids_list[stratml]:
            for options in grids_list[stratml][k]:
                if k + "_" + options in target:
                    print(stratml + " with value " + k + "\noption:" + options + "\n")

                    estimator = grids_list[stratml][k][options]
                    print(estimator)

                    X, y, fs, g = new_dbs[options]

                    mean_conf_mat, sum_conf_mat, scores, classes = custom_cross_val(X, y, estimator, cv, target_classes,
                                                                                    scoring, binary, labels, g)
                    print(sum(scores) / len(scores))
                    if summat:
                        ConfusionMatrixDisplay(sum_conf_mat, display_labels=classes).plot()
                    else:

                        # Calcul de la somme des éléments de chaque ligne
                        row_sums = np.sum(sum_conf_mat, axis=1)

                        # Mise à l'échelle en divisant chaque élément par la somme de sa ligne
                        scaled_arr = sum_conf_mat / row_sums[:, np.newaxis]
                        # cm = ConfusionMatrixDisplay(sum_conf_mat / np.sum(sum_conf_mat), display_labels=classes).plot()
                        ConfusionMatrixDisplay(scaled_arr, display_labels=classes).plot()

                    plt.savefig(final_path + "/cm_" + stratml + "_" + k + "_" + options + ".png")
