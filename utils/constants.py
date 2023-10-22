
p = 16 # num of numerical features
K = 2 # time features 


# train_start = "2016-01-28"
# test_end = '2018-01-01'



column_numerical = {
'S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp', 'S1_Light',
       'S2_Light', 'S3_Light', 'S4_Light', 'S1_Sound', 'S2_Sound', 'S3_Sound',
       'S4_Sound', 'S5_CO2', 'S5_CO2_Slope', 'S6_PIR', 'S7_PIR'

}


column_data_extended_types = {

}


column_names_raw = [

]

column_names_extended = [

]


EVAL_METRIC_LIST = ['f1_weighted' , 'auc_weighted', 'accuracy_score', 'balanced_accuracy_score', 'precision_score', 'recall_score', 'jaccard_score']# ['rmse','mape','wmape','wbias','wuforec','woforec'] #################################################################################### !!!!!!!!!!!!!!!!!!!!!!!! all change to clasifications 

RANDOM_SEED = 42 # global random_state