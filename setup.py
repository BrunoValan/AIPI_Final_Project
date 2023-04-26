import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV, KFold
from sklearn.feature_selection import f_regression, mutual_info_regression, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report
from sklearn.metrics import plot_roc_curve
from joblib import dump

def prep_data(): 
    
    '''This function loads the and merges data, drops irrelevant features, handles missingness and outliers and performs feature engineering
    Inputs: 
        none 
    Returns: 
        df_prepped (pd.DataFrame): data frame with cleaned patient and geographic data

    '''

    #loads patient data
    patient_data = pd.read_csv('./Data_Clean/RTED_ADIMERGE.csv')
    geographic_data = pd.read_csv('./Data_Clean/geographic_data_clean.csv')

    #drop state since it is already in the patient data frame
    geographic_data = geographic_data.drop(labels = ['STATE'], axis = 1)

    #encode ZIP_5 as a object for the merge on geographic data frame   
    geographic_data['ZIP_5'] = geographic_data['ZIP_5'].astype(str)

    #encode ZIP_5 as a object for merge on patient data frame
    patient_data['ZIP_5'] = patient_data['ZIP_5'].astype(str)

    #still some float values in there which we are causing data loss on the merge - grab first 5 characters of the string
    patient_data['ZIP_5'] = patient_data['ZIP_5'].str[:5]

    #merge patient and geographic data frame
    df_full = pd.merge(patient_data, geographic_data, how = 'left', on = 'ZIP_5')

    #cols to drop due to leakge or redundancy 
    cols_drop = ['RET_CSN', 'RET_DAYS', 'RET_HOSPITAL', 'RET_ED_DISPO', 'RET_CHIEF_COMPLAINT', 
                'RET_CLINICAL_IMPRESSION', 'RET_HB_PRIM_DX_CODE', 'RET_HB_PRIM_DX_NAME', 'RET_ED_DENOM',
                'RET_ED30_NUMER', 'EDRevisitDischargedPatient', 'READMISSION', 'Readmission90', 'EDRevisit90', 
                'PAT_ZIP', 'COUNTY', 'CITY', 'location', 'Latitude', 'Longitude', 'WEIGHTED_ADI', 
                'PAYOR_NAME', 'ATTENDING_PROV', 'OR_LOGS', 'OR_LOG_ROW', 'LOCATION_NAME', 'LOCATION_NM', 
                'SERVICE_NAME', 'PRIMARY_PHYSICIAN_NM', 'CLIN_DEP', 'PRIMARY_PROCEDURE_NM', 'PRIMARY_PROCEDURE_CPT']
    
    #drop na for columns with minimal na values
    cols_dropna = ['RACE', 'ETHNIC_GROUP', 'DX_HYPERTENSION', 'DX_RENAL_FAILURE', 'DX_COPD', 'DX_TYPE_2_DM', 'DX_HIP_FRACTURE', 'DX_OSTEOPOROSIS', 'STATE', 'PRIMARY_PROC_CPT_CODE']

    #we will fill with median imputation for these columns
    cols_fillna = ['BMI', 'distance_to_hospital']
    
    #drops columns
    df_simplified = df_full.drop(labels = cols_drop, axis = 1)

    #drop na for columns with only a few missing values 
    df_clean = df_simplified.dropna(subset = cols_dropna)

    #fill BMI and distance with the median due to right tailed distribution (will handle outliers later)  
    df_clean[cols_fillna] = df_clean[cols_fillna].fillna(df_clean[cols_fillna].median())

    #re-assign to clean_df
    clean_df = df_clean.copy()

    #rename response columns 
    clean_df['RETURN_ED_90DAY'] = clean_df['ED90Day']

    #drop original
    clean_df = clean_df.drop(labels = ['ED90Day'], axis = 1)

    #creates list of BMI anomolies to drop 
    k = 3
    col = clean_df['BMI']
    col_std = np.std(clean_df['BMI'])
    col_mean = np.mean(clean_df['BMI'])
    thresh = col_std * k
    lower_limit  = col_mean - thresh
    upper_limit = col_mean + thresh
    BMI_anomalies = list(col.index[(col>upper_limit) | (col<lower_limit)])

    #drops BMI outliers
    df_no_outliers = clean_df.drop(BMI_anomalies, axis =0)

    #drops instances when LOS is >90 days 
    df_no_outliers = df_no_outliers[df_no_outliers['LOS_DAYS'] <= 90]

    #next we will create binary flag variables for minority status and ethnic minority status
    minority_races = ['Black or African American', 'Other', 'Asian', 'American Indian or Alaskan Native', 'Native Hawaiian or Other Pacific Islander']
    ethnic_minority = ['Hispanic Other', 'Hispanic Mexican', 'Hispanic Puerto Rican', 'Hispanic Cuban']

    #creates racial minority column 
    df_no_outliers['RacialMinority'] = np.where(df_no_outliers['RACE'].isin(minority_races), 1,0)

    #creates ethnic minority column 
    df_no_outliers['EthnicMinority'] = np.where(df_no_outliers['ETHNIC_GROUP'].isin(ethnic_minority), 1,0)

    #creates language not english column 
    df_no_outliers['LanguageNotEnglish'] = np.where(df_no_outliers['PAT_LANGUAGE'] == 'English', 0,1)

    df_modelling = df_no_outliers.copy()

    #columns deemed unimportant from univariate testing
    cols_unimportant = ['PAT_CLASS', 'BMI', 'PAT_LANGUAGE', 'ETHNIC_GROUP', 'FINANCIAL_CLASS_NAME', 
             'RACE', 'ZIP_5', 'DISCH_DEPT', 'LOCATION_ID', 'ZIP_5',
             'WhiteNonHipanic', 'Race_OtherAsian', 'Race_Other', 'Race_NotReported', 'Race_NativeHawaiian',
             'Race_White', 'Race_Black', 'Race_Asian', 'Race_AmericanIndian', 'Race_NotValid', 
             'Hispanic_7', 'Hispanic_NotHispanic', 'Hispanic_5', 'Hispanic_4', 'Hispanic_3', 
             'Hispanic_2', 'Hispanic_1', 'RaceDummy_5', 'RaceDummy_4', 'FinancialClass_Commercial',
             'FinancialClass_Liability', 'FinancialClass_ManagedCare','FinancialClass_MedicaidPending', 
             'FinancialClass_MediCARE','FinancialClass_MedicareAdvantage','FinancialClass_CommercialBlueCross', 
             'FinancialClass_MedicaidNC','FinancialClass_MedicaidManaged', 'FinancialClass_CommercialBCOOS',
            'FinancialClass_Medcaid', 'FinancialClass_12Unkonwn','FinancialClass_13', 'FinancialClass_14', 
            'FinancialClass_15','FinancialClass_WorkersComp', 'Sex_Female', 'Sex_2', 'Sex_3', 'PRIMARY_PROC_CPT_CODE']
    
    #drop unimportant columns
    df_prepped = df_modelling.drop(labels = cols_unimportant, axis = 1)

    return df_prepped

def encode_feats(df_prepped): 
    '''Encodes the remaining catagorical features using one hot encoding, then filters for only the relevant features. Splits data into training and test sets

    Inputs: 
        df_prepped (pd.DataFrame): cleaned data set from prep_data()

    Returns: 
        X_train (pd.DataFrame): encoded training set 
        X_test (pd.DataFrame): encoded test set 
        y_train (numpy array): training labels 
        y_test (numpy array): test labels
    '''
    #cols to encode 
    cols_onehot = ['SEX', 'DISCH_LOC_ABBR', 'DISCHARGE_DISPO', 'CASE_CLASS_NM', 'CLIN_DIV', 'PAT_BASE_CLASS', 'STATE']

    #Make sure all categorical columns are string type
    for col in cols_onehot:
        df_prepped[col] = df_prepped[col].astype(str)

    # Encode categorical variables
    onehot_enc = OneHotEncoder(handle_unknown='ignore')
    # Fit encoder on training data
    onehot_enc.fit(df_prepped[cols_onehot])
    # Get the names of the new columns created
    colnames = columns=list(onehot_enc.get_feature_names_out(input_features=cols_onehot))
    # Transform the data
    onehot_vals = onehot_enc.transform(df_prepped[cols_onehot]).toarray()
    # Put transformed data into dataframe.  Make sure index matches
    enc_df = pd.DataFrame(onehot_vals,columns=colnames,index=df_prepped.index)
    # Add onehot columns back onto original dataframe and drop the original columns
    encoded_df = pd.concat([df_prepped,enc_df],axis=1).drop(cols_onehot,axis=1)

    #list of top 20 important features
    top_20 = ['distance_to_hospital', 'AGE', 'LOS_DAYS', 'SevereObesity', 'RacialMinority', 
              'DX_HYPERTENSION', 'DX_TYPE_2_DM', 'SEX_Female', 'Elderly65', 'SEX_Male', 'DISCH_LOC_ABBR_DUH', 
              'CLIN_DIV_TOTAL JOINT', 'MedicaidBinary', 'DX_COPD', 'DX_RENAL_FAILURE', 'DISCH_LOC_ABBR_DRAH', 
              'DISCH_LOC_ABBR_DRH', 'CLIN_DIV_HAND', 'PAT_BASE_CLASS_Inpatient', 'DISCHARGE_DISPO_Home or Self Care', 'RETURN_ED_90DAY']
    
    final_df = encoded_df[top_20]

    return final_df

def modelling(final_df): 
    '''
    Funciton splits final data set into training and test set, trains optimized random forest and makes prediction on test set. 
    It also saves the trained model object into the model directory and prints the classification report for the predictions 
    on the test set. 

    Inputs: final_df (pd.DataFrame): data frame of the cleaned and encoded data set with only the relevant features we will use in our final model

    Returns: final model object (sklearn.ensemble._forest.RandomForestClassifier)
    '''

    # Create feature matrix
    X = final_df.drop(labels='RETURN_ED_90DAY',axis=1)

    # Create target vector
    y = final_df['RETURN_ED_90DAY'].copy().to_numpy()

    # Split our data
    X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=0,test_size=0.2, stratify= y)

    #define final model
    final_model = RandomForestClassifier(criterion = 'gini', max_features= 0.1, max_depth = None, 
                                 max_samples= 0.3, min_samples_leaf= 1, n_estimators= 1000, random_state = 0)
    #fit model to training & test data
    final_model.fit(X_train,y_train)

    #get final preditions
    final_preds = final_model.predict(X_test)

    #get probability from the model
    predict_proba = final_model.predict_proba(X_test)[:, 1]

    #change prediction threshold 0.2 
    preds_custom = (predict_proba >= 0.2).astype(int)

    #saves classification report as an object w/ lower prediction threshold 
    report_custom = classification_report(y_test, preds_custom)

    #saves classification report as an object w/ lower default threshold 
    report_default = classification_report(y_test, final_preds)
    
    #prints both classification reports
    print(f'the classification report for the optimal RF on TEST set (prediction threshold = 0.2) \n {report_custom}')
    print(f'the classification report for the optimal RF on the TEST set (prediction threshold = 0.5) is\n {report_default}')

    dump(final_model, './models/optimized_random_forest.pkl')

    return final_model

def main():
    prepped_df = prep_data()
    encoded_df = encode_feats(prepped_df)
    final_model = modelling(encoded_df)
if __name__ == "__main__":
    main()
