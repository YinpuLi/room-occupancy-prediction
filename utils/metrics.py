from sklearn import metrics
import pandas as pd
import numpy as np


def compute_metrics(df : pd.DataFrame, # the df include two columns: (actual, forecast)
                    y_pred_prob,  # for predict prob
                    metrics_list: list # the list include every metrics you wanna calculate
                    ) -> dict:
    output = {}
    for m in metrics_list:
        if m != 'auc_weighted':
            output[m] = globals()[m](df)
        else:
            output[m] = globals()[m](df ,y_pred_prob)
    return output


def wbias(df):
    """
    weighted bias
    wbias = sum(Ai - Fi)/sum(Ai)
    """
    return np.nansum(df.actual - df.forecast)/np.nansum(df.actual)

def mape(df):
    """
    mean absolute error, using sklearn: https://scikit-learn.org/stable/modules/model_evaluation.html#mean-absolute-percentage-error
    mape = mean(|Ai - Fi|/|Ai|)
    """
    return metrics.mean_absolute_percentage_error(df.actual, df.forecast)

def wmape(df):
    """
    weighted mean absolute percentage error
    wmape = sum(|Ai - Fi|)/sum(|Ai|)
    """
    return np.nansum(np.abs(df.forecast - df.actual)) / np.nansum(np.abs(df.actual))

def rmse(df):
    """
    rooted mean squared error, using sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
    rmse = sqrt(mean((Ai - Fi)^2))
    """
    return metrics.mean_squared_error(df.actual, df.forecast, squared = False)

def wuforec(df):
    """
    weighetd underforecasting: the magnitude of underforecasting normalized by absolute mean scale of the actuals
    """
    if sum(df.actual != 0) == 0:
        return np.nan

    under_df = df[df.forecast < df.actual]  # If forecast < actual then underforecast
    return (under_df.actual - under_df.forecast).sum() / np.abs(df.actual).sum()

def woforec(df):
    """
    weighetd overforecasting: the magnitude of underforecasting normalized by absolute mean scale of the actuals
    """
    if sum(df.actual != 0) == 0:
        return np.nan

    over_df = df[df.forecast > df.actual]  # If forecast > actual then underforecast
    return (over_df.forecast - over_df.actual).sum() / np.abs(df.actual).sum()



def f1_weighted(df):

    """
    Compute the F1 score, also known as balanced F-score or F-measure.
    F1 = 2 * (precision * recall) / (precision + recall)
    In the multi-class and multi-label case, this is the average of the F1 score of each class with weighting depending on the average parameter.
    """


    return metrics.f1_score(df.actual, df.forecast, average = 'weighted')




def auc_weighted(df, y_pred_prob):
    
    
    """
    Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
    
    'ovr': Stands for One-vs-rest. Computes the AUC of each class against the rest [3] [4]. This treats the multiclass case in the same way as the multilabel case. Sensitive to class imbalance even when average == 'macro', because class imbalance affects the composition of each of the ‘rest’ groupings.
    """
    return metrics.roc_auc_score(df.actual, y_pred_prob, average = 'weighted', multi_class = 'ovo', labels = [0,1,2,3])




def accuracy_score(df):

    """
The balanced accuracy in binary and multiclass classification problems to deal with imbalanced datasets. It is defined as the average of recall obtained on each class.
    """


    return metrics.accuracy_score(df.actual, df.forecast)




def balanced_accuracy_score(df):

    """
The balanced accuracy in binary and multiclass classification problems to deal with imbalanced datasets. It is defined as the average of recall obtained on each class.
    """


    return metrics.balanced_accuracy_score(df.actual, df.forecast)


# def average_precision_score(df):

#     """
# The balanced accuracy in binary and multiclass classification problems to deal with imbalanced datasets. It is defined as the average of recall obtained on each class.
#     """


#     return metrics.average_precision_score(df.actual, df.forecast, average='weighted')


def precision_score(df):

    """
The balanced accuracy in binary and multiclass classification problems to deal with imbalanced datasets. It is defined as the average of recall obtained on each class.
    """


    return metrics.precision_score(df.actual, df.forecast, average='weighted')

def recall_score(df):

    """
The balanced accuracy in binary and multiclass classification problems to deal with imbalanced datasets. It is defined as the average of recall obtained on each class.
    """


    return metrics.recall_score(df.actual, df.forecast, average='weighted')



def jaccard_score(df):

    """
The balanced accuracy in binary and multiclass classification problems to deal with imbalanced datasets. It is defined as the average of recall obtained on each class.
    """


    return metrics.jaccard_score(df.actual, df.forecast, average='weighted')