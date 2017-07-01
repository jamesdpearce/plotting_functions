import time
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from progress import ProgressBar

def calibrated_proba(f_train, y, f_predict, verbose = 0, C = 1):
    """ 
    Platt scaling is an algorithm to produce probability estimates via a logistic transformation of the
    classifier scores f(X), where the two logistic parameters are learned by traning
    with the original training set. To avoid overfitting to this set cross-validation is used.
    
    https://en.wikipedia.org/wiki/Platt_scaling
    """
    #Train Logistic Regression with cross-validation and make predictions
    return LogisticRegression(verbose = verbose, C = C)\
            .fit(f_train.reshape(-1,1), y)\
            .predict_proba(f_predict.reshape(-1,1)).T[1]

# Plotting
def plot_roc_curve(fpr, tpr):
    plt.figure()
    plt.hold(True)
    plt.plot(fpr, tpr, label = 'AUC = %.3f' % auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], 'k--', label = 'Random')
    plt.hold(False)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize = 16)
    plt.ylabel('True Positive Rate', fontsize = 16)
    plt.title('Receiver Operating Characteristic', fontsize = 18)
    plt.legend()
    sns.despine()
    plt.show()

def partial_plot(clf, X_, x_name, labels, n_points = 100, lims = None, n_samp = 1000, categorical = False):
    X = X_.copy()
    N = len(X)
    if lims == None:
        x_min = X[x_name].min()
        x_max = X[x_name].max()
    else:
        x_min = lims[0]
        x_max = lims[1]
    if categorical: 
        x = np.array([x_min, x_max]*int(n_points/2.))
    else:
        x = np.linspace(x_min, x_max, n_points)
    p = []
    pb = ProgressBar()
    for i, x_i in enumerate(x):
        X[x_name] = [x_i]*N
        _idx = np.random.randint(N,size = n_samp) #sub sample to reduce time to evaluate 
        p.append(clf.predict_proba(X.values[_idx], labels = labels[_idx])[1].mean(0))
        pb.update_progress(i/n_points)
    return x, np.array(p)    
    

def plot_multi_roc_curve(fprs, tprs, y_labels = None, title = None):
    if y_labels == None:
        y_labels = ['']*len(fprs)
    plt.figure()
    plt.hold(True)
    for fpr, tpr, label in zip(fprs,tprs,y_labels):
        plt.plot(fpr, tpr, label = label + ' AUC = %.3f' % auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], 'k--', label = 'Random')
    plt.hold(False)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize = 16)
    plt.ylabel('True Positive Rate', fontsize = 16)
    if title == None: 
        plt.title('Receiver Operating Characteristic', fontsize = 18)
    else:
        plt.title(title, fontsize = 18)
    plt.legend(loc = 4, fontsize = 14)
    sns.despine()
    plt.show()
    
def plot_reliability_curve(y_train_pred_proba, y_cali_pred_proba, y_train, show_pred = True, 
                           save = False, figures_path = ''): 
  
    if show_pred :
        plot_series_x = pd.rolling_mean(pd.Series(np.sort(y_train_pred_proba)), 
                                len(y_train_pred_proba)/200)
    plot_series_cali_x = pd.rolling_mean(pd.Series(np.sort(y_cali_pred_proba)), 
                                len(y_train_pred_proba)/200)
    plot_series_y = pd.rolling_mean(pd.Series(y_train[np.argsort(y_train_pred_proba)]),
                                len(y_train_pred_proba)/200)

    # Plot of accuracy as a function of predicted probability quantile
    # 0.5% resolution
    plt.hold(True)
    # Accuracy
    plt.plot(plot_series_y, label = 'Empirical')
    if show_pred:
        plt.plot(plot_series_x, label = 'Predicted')
    plt.plot(plot_series_cali_x, label = 'Calibrated')
    # Theoretical optimum
    #plt.plot(1.1*np.max(plot_series_y)*pd.rolling_mean(pd.Series(y_train[np.argsort(y_train)]),
    #                                             len(y_pred_proba)/100))
    # Baseline
    plt.axhline(plot_series_y.mean(), color = 'k', linestyle = '--', label = 'Baseline')
    plt.hold(False)
    plt.suptitle('Churn Detection', fontsize=20)
    plt.ylabel('Churn Rate', fontsize=16)
    plt.xlabel('Merchant (ordered by increasing predicted probability)', fontsize=16)
    sns.despine()
    plt.legend(loc = 2)
    if save : plt.savefig(figures_path + 'churn_quantile_accuracy.png')
    plt.show()

    # Plot of calibration
    plt.hold(True)
    if show_pred:
        plt.plot(plot_series_x, plot_series_y, label = 'Uncalibrated')
    plt.plot(plot_series_cali_x, plot_series_y, label = 'Calibrated')
    min_ = min(np.min(plot_series_cali_x), np.min(plot_series_y))
    max_ = max(np.max(plot_series_cali_x), np.max(plot_series_y))
    plt.plot([min_, max_], [min_, max_], color = 'k', linestyle = '--', label = 'Optimal calibration')
    plt.hold(False)
    plt.suptitle('Churn Model Calibration', fontsize=20)
    plt.xlabel('Predicted churn probability', fontsize=16)
    plt.ylabel('Empirical churn probability', fontsize=16)
    sns.despine()
    if save : plt.savefig(figures_path + 'churn_calibration.png')
    plt.legend(loc = 4 )
    plt.show()
    

def test_reliability(proba, y_train, nbins = 25, weighted = False, save = False, figures_path = ''):
    """
    Test's the accuracy of the probabily output. 
    
    1) Calculate the empirical probability in each bin of the predicted probabilites histogram
    2) Calculate error bars of empirical probability for each bin
    3) Calculate reliabilty metric (R-squared)
    4) Plot it
    
    Args
    ----
        
        proba : predicted probability scores of validation set
        
        y_train : class labels of validation set. These must correspond to the predicted probs
        
        nbins : Number of bins for histogram
    """
    
    # 1) 
    hist, edges = np.histogram(proba, bins = nbins)
    y = y_train[np.argsort(proba)]
    
    df = pd.DataFrame()
    
    #Find middle of bins for predicted probability
    df['pred_proba'] = (edges[:-1] + edges[1:])/2.
    
    #Calculate empirical probabilities in each bin
    emp_proba = []
    emp_pos = []
    emp_N = []
    sum_bins = 0.0
    for current_bin in hist:
        if current_bin == 0:
            emp_proba.append(np.nan)
            emp_pos.append(np.nan)
            emp_N.append(np.nan)
            continue
        pos_b = float(y[sum_bins:sum_bins + current_bin].sum())
        N_b = float(current_bin)
        emp_pos.append(pos_b)
        emp_N.append(N_b)
        emp_proba.append(pos_b/N_b)
        sum_bins += current_bin
        

    # 2)
    df['emp_pos'] = np.array(emp_pos)
    df['emp_N'] = np.array(emp_N)
    df['emp_proba'] = np.array(emp_proba)
    
    df.dropna(inplace = True)
    df = df[df.emp_pos > 0]
    df = df[df.emp_N > 10]
    
    df['emp_errors'] = np.sqrt(df.emp_pos)/df.emp_N * np.sqrt( 1 - df.emp_pos/df.emp_N)
    
    # 3) 
    # Use classic R^2 (weighted = False); R^2 with terms weighted by number or points in bins
    if weighted:
        weights = df.emp_N
    else:
        weights = 1.0
        
    SS_res = sum( weights*(df.emp_proba - df.pred_proba)**2 )
    SS_tot = sum( weights*(df.emp_proba - df.emp_proba.mean())**2 )
    R_2 = 1 - SS_res/SS_tot
    
    
    # 4)
    plt.figure()
    plt.hold(True)
    plt.errorbar(df.pred_proba, df.emp_proba, yerr = df.emp_errors, fmt='o')
    plt.xlim([0.0, df.pred_proba.max() + 0.1])
    plt.ylim([0.0, df.emp_proba.max() + 0.1])
    plt.xlabel('Predicted Probability', fontsize = 16)
    plt.ylabel('Empirical Probability', fontsize = 16)
    plt.title('Reliability plot', fontsize = 20)
    plt.plot([0, 1], [0, 1], 'k--', label = 'Perfect correlation')
    plt.axhline(y_train.mean(), color = 'r', linestyle = '--', label = 'Random')
    plt.legend(loc = 2, fontsize = 16)
    sns.despine()
    plt.text(0.03, df.emp_proba.max() - df.emp_proba.max()/100*12 ,
             '$R^2 = $ %.4f' % R_2, fontsize=16)
    plt.show()
    
    if save : plt.savefig(figures_path + 'reliability_plot.png')
    
    return df
    
def test_reliability_multiclass(proba_arr, y_train_arr, y_labels = None, nbins = 25, weighted = False, save = False, figures_path = ''):
    """
       Same as test_reliability but extended for multiclass labels 
    """
    
    if y_labels == None: 
        y_labels = ['']*proba_arr.shape[1]
    
    df = pd.DataFrame()
    
    plt.figure()
    plt.hold(True)
    
    for proba, y_train, label in zip(proba_arr.T, y_train_arr.T, y_labels):
    
        # 1) 
        hist, edges = np.histogram(proba, bins = nbins)
        y = y_train[np.argsort(proba)]

        #Find middle of bins for predicted probability
        pred_proba = (edges[:-1] + edges[1:])/2.

        #Calculate empirical probabilities in each bin
        emp_proba = []
        emp_pos = []
        emp_N = []
        sum_bins = 0.0
        for current_bin in hist:
            if current_bin == 0:
                emp_proba.append(np.nan)
                emp_pos.append(np.nan)
                emp_N.append(np.nan)
                continue
            pos_b = float(y[sum_bins:sum_bins + current_bin].sum())
            N_b = float(current_bin)
            emp_pos.append(pos_b)
            emp_N.append(N_b)
            emp_proba.append(pos_b/N_b)
            sum_bins += current_bin


        # 2)
        emp_pos = np.array(emp_pos)
        demp_N = np.array(emp_N)
        emp_proba = np.array(emp_proba)

        #df.dropna(inplace = True)
        #df = df[df.emp_pos > 0]
        #df = df[df.emp_N > 10]

        emp_errors = np.sqrt(emp_pos)/emp_N * np.sqrt( 1 - emp_pos/emp_N)

        # 3) 
        # Use classic R^2 (weighted = False); R^2 with terms weighted by number or points in bins
        if weighted:
            weights = emp_N
        else:
            weights = 1.0

        SS_res = sum( weights*(emp_proba - pred_proba)**2 )
        SS_tot = sum( weights*(emp_proba - np.mean(emp_proba)**2 ))
        R_2 = 1 - SS_res/SS_tot


        # 4)
        plt.errorbar(pred_proba, emp_proba, yerr = emp_errors, fmt='o', label = label + ' $R^2 = $ %.4f' % R_2)
    
    plt.xlim([0,1.05])
    plt.ylim([0,1.05])
    plt.xlabel('Predicted Probability', fontsize = 16)
    plt.ylabel('Empirical Probability', fontsize = 16)
    plt.title('Reliability plot', fontsize = 20)
    plt.plot([0, 1], [0, 1], 'k--', label = 'Perfect correlation')
    plt.legend(loc = 2, fontsize = 12)
    sns.despine()
    plt.show()

    if save : plt.savefig(figures_path + 'reliability_plot_multiclass.png')

    return 




