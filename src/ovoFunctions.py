import pandas as pd
import numpy as np
from collections import Counter
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import cross_validate, cross_val_predict, train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.under_sampling import CondensedNearestNeighbour, RandomUnderSampler
import operator
from joblib import dump, load

    
def ovoModel(_data, _labels, test, under=True, n_us=6000):
    # ====== DESCOMPOSICIÓN: ====== #
    # Creación de modelos one-vs-one:
    
    # OFFICE vs RETAIL
    data = pd.concat([_data,_labels], axis=1)
    data_1 = data[data.CLASE == "OFFICE"]
    data_2 = data[data.CLASE == "RETAIL"]
    data = pd.concat([data_1, data_2])
    labels = data.CLASE
    data = data.drop('CLASE', axis=1)
    
    data = data.drop('GEOM_R2', axis=1)
    
    data['MAXBUILDINGFLOOR2'] = data['MAXBUILDINGFLOOR']**2
    data['Q_G_3_0_32'] = data['Q_G_3_0_3']**2
    
    
    X = data.values
    y = labels.values
    model_office_vs_retail = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    
    # OFFICE vs PUBLIC
    data = pd.concat([_data,_labels], axis=1)
    data_1 = data[data.CLASE == "OFFICE"]
    data_2 = data[data.CLASE == "PUBLIC"]
    data = pd.concat([data_1, data_2])
    labels = data.CLASE
    data = data.drop('CLASE', axis=1)
    
    data = data.drop('GEOM_R3', axis=1)
    
    data['Y2'] = data['Y']**2
    data['Q_NIR_8_0_92'] = data['Q_NIR_8_0_9']**2
    data['Q_B_2_0_32'] = data['Q_B_2_0_3']**2
    
    X = data.values
    y = labels.values
    model_office_vs_public = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)

    # OFFICE vs OTHER
    data = pd.concat([_data,_labels], axis=1)
    data_1 = data[data.CLASE == "OFFICE"]
    data_2 = data[data.CLASE == "OTHER"]
    data = pd.concat([data_1, data_2])
    labels = data.CLASE
    data = data.drop('CLASE', axis=1)
    
    data = data.drop('Q_B_2_0_2', axis=1)
    data = data.drop('Q_B_2_1_0', axis=1)
    data = data.drop('Q_NIR_8_0_8', axis=1)
    
    data['CADASTRALQUALITYID2'] = data['CADASTRALQUALITYID']**2
    data['Q_NIR_8_0_52'] = data['Q_NIR_8_0_5']**2
    data['Q_G_3_0_22'] = data['Q_G_3_0_2']**2
    
    X = data.values
    y = labels.values
    model_office_vs_other = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    
    # OFFICE vs INDUSTRIAL
    data = pd.concat([_data,_labels], axis=1)
    data_1 = data[data.CLASE == "OFFICE"]
    data_2 = data[data.CLASE == "INDUSTRIAL"]
    data = pd.concat([data_1, data_2])
    labels = data.CLASE
    data = data.drop('CLASE', axis=1)
    
    data = data.drop('Q_NIR_8_0_8', axis=1)
    
    data['Q_B_2_0_62'] = data['Q_B_2_0_6']**2
    
    X = data.values
    y = labels.values
    model_office_vs_industrial = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    
    # OFFICE vs AGRICULTURE
    data = pd.concat([_data,_labels], axis=1)
    data_1 = data[data.CLASE == "OFFICE"]
    data_2 = data[data.CLASE == "AGRICULTURE"]
    data = pd.concat([data_1, data_2])
    labels = data.CLASE
    data = data.drop('CLASE', axis=1)
    
    data = data.drop('Q_G_3_0_2', axis=1)
    
    data['Y2'] = data['Y']**2
    data['X2'] = data['X']**2
    data['Q_NIR_8_0_32'] = data['Q_NIR_8_0_3']**2
    
    X = data.values
    y = labels.values
    model_office_vs_agriculture = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    
    # OFFICE vs RESIDENTIAL
    data = pd.concat([_data,_labels], axis=1)
    data_1 = data[data.CLASE == "OFFICE"]
    data_2 = data[data.CLASE == "RESIDENTIAL"]
    data = pd.concat([data_1, data_2])
    labels = data.CLASE
    data = data.drop('CLASE', axis=1)
    X = data.values
    y = labels.values
    
    if(under):
        sampling_dict = {'RESIDENTIAL': n_us}
        us = RandomUnderSampler(random_state=42, sampling_strategy=sampling_dict) 
        X, y = us.fit_resample(X, y)
    
    model_office_vs_residential = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    
    # AGRICULTURE vs RETAIL
    data = pd.concat([_data,_labels], axis=1)
    data_1 = data[data.CLASE == "AGRICULTURE"]
    data_2 = data[data.CLASE == "RETAIL"]
    data = pd.concat([data_1, data_2])
    labels = data.CLASE
    data = data.drop('CLASE', axis=1)
    X = data.values
    y = labels.values
    model_agriculture_vs_retail = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    
    # AGRICULTURE vs PUBLIC
    data = pd.concat([_data,_labels], axis=1)
    data_1 = data[data.CLASE == "AGRICULTURE"]
    data_2 = data[data.CLASE == "PUBLIC"]
    data = pd.concat([data_1, data_2])
    labels = data.CLASE
    data = data.drop('CLASE', axis=1)
    X = data.values
    y = labels.values
    model_agriculture_vs_public = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    
    # AGRICULTURE vs OTHER
    data = pd.concat([_data,_labels], axis=1)
    data_1 = data[data.CLASE == "AGRICULTURE"]
    data_2 = data[data.CLASE == "OTHER"]
    data = pd.concat([data_1, data_2])
    labels = data.CLASE
    data = data.drop('CLASE', axis=1)
    X = data.values
    y = labels.values
    model_agriculture_vs_other = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    
    # AGRICULTURE vs INDUSTRIAL
    data = pd.concat([_data,_labels], axis=1)
    data_1 = data[data.CLASE == "AGRICULTURE"]
    data_2 = data[data.CLASE == "INDUSTRIAL"]
    data = pd.concat([data_1, data_2])
    labels = data.CLASE
    data = data.drop('CLASE', axis=1)
    X = data.values
    y = labels.values
    model_agriculture_vs_industrial = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    
    # AGRICULTURE vs RESIDENTIAL
    data = pd.concat([_data,_labels], axis=1)
    data_1 = data[data.CLASE == "AGRICULTURE"]
    data_2 = data[data.CLASE == "RESIDENTIAL"]
    data = pd.concat([data_1, data_2])
    labels = data.CLASE
    data = data.drop('CLASE', axis=1)
    X = data.values
    y = labels.values
    
    if(under):
        sampling_dict = {'RESIDENTIAL': n_us}
        us = RandomUnderSampler(random_state=42, sampling_strategy=sampling_dict) 
        X, y = us.fit_resample(X, y)
    
    model_agriculture_vs_residential = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    
    # INDUSTRIAL vs RETAIL
    data = pd.concat([_data,_labels], axis=1)
    data_1 = data[data.CLASE == "INDUSTRIAL"]
    data_2 = data[data.CLASE == "RETAIL"]
    data = pd.concat([data_1, data_2])
    labels = data.CLASE
    data = data.drop('CLASE', axis=1)
    X = data.values
    y = labels.values
    model_industrial_vs_retail = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    
    # INDUSTRIAL vs PUBLIC
    data = pd.concat([_data,_labels], axis=1)
    data_1 = data[data.CLASE == "INDUSTRIAL"]
    data_2 = data[data.CLASE == "PUBLIC"]
    data = pd.concat([data_1, data_2])
    labels = data.CLASE
    data = data.drop('CLASE', axis=1)
    X = data.values
    y = labels.values
    model_industrial_vs_public = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    
    # INDUSTRIAL vs OTHER
    data = pd.concat([_data,_labels], axis=1)
    data_1 = data[data.CLASE == "INDUSTRIAL"]
    data_2 = data[data.CLASE == "OTHER"]
    data = pd.concat([data_1, data_2])
    labels = data.CLASE
    data = data.drop('CLASE', axis=1)
    X = data.values
    y = labels.values
    model_industrial_vs_other = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    
    # INDUSTRIAL vs RESIDENTIAL
    data = pd.concat([_data,_labels], axis=1)
    data_1 = data[data.CLASE == "INDUSTRIAL"]
    data_2 = data[data.CLASE == "RESIDENTIAL"]
    data = pd.concat([data_1, data_2])
    labels = data.CLASE
    data = data.drop('CLASE', axis=1)
    X = data.values
    y = labels.values
    
    if(under):
        sampling_dict = {'RESIDENTIAL': n_us}
        us = RandomUnderSampler(random_state=42, sampling_strategy=sampling_dict) 
        X, y = us.fit_resample(X, y)
    
    model_industrial_vs_residential = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    
    # OTHER vs RETAIL
    data = pd.concat([_data,_labels], axis=1)
    data_1 = data[data.CLASE == "OTHER"]
    data_2 = data[data.CLASE == "RETAIL"]
    data = pd.concat([data_1, data_2])
    labels = data.CLASE
    data = data.drop('CLASE', axis=1)
    X = data.values
    y = labels.values
    model_other_vs_retail = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    
    # OTHER vs PUBLIC
    data = pd.concat([_data,_labels], axis=1)
    data_1 = data[data.CLASE == "OTHER"]
    data_2 = data[data.CLASE == "PUBLIC"]
    data = pd.concat([data_1, data_2])
    labels = data.CLASE
    data = data.drop('CLASE', axis=1)
    X = data.values
    y = labels.values
    model_other_vs_public = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    
    # OTHER vs RESIDENTIAL
    data = pd.concat([_data,_labels], axis=1)
    data_1 = data[data.CLASE == "OTHER"]
    data_2 = data[data.CLASE == "RESIDENTIAL"]
    data = pd.concat([data_1, data_2])
    labels = data.CLASE
    data = data.drop('CLASE', axis=1)
    X = data.values
    y = labels.values
    
    if(under):
        sampling_dict = {'RESIDENTIAL': n_us}
        us = RandomUnderSampler(random_state=42, sampling_strategy=sampling_dict) 
        X, y = us.fit_resample(X, y)
    
    model_other_vs_residential = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    
    # PUBLIC vs RETAIL
    data = pd.concat([_data,_labels], axis=1)
    data_1 = data[data.CLASE == "PUBLIC"]
    data_2 = data[data.CLASE == "RETAIL"]
    data = pd.concat([data_1, data_2])
    labels = data.CLASE
    data = data.drop('CLASE', axis=1)
    X = data.values
    y = labels.values
    model_public_vs_retail = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    
    # PUBLIC vs RESIDENTIAL
    data = pd.concat([_data,_labels], axis=1)
    data_1 = data[data.CLASE == "PUBLIC"]
    data_2 = data[data.CLASE == "RESIDENTIAL"]
    data = pd.concat([data_1, data_2])
    labels = data.CLASE
    data = data.drop('CLASE', axis=1)
    X = data.values
    y = labels.values
    
    if(under):
        sampling_dict = {'RESIDENTIAL': n_us}
        us = RandomUnderSampler(random_state=42, sampling_strategy=sampling_dict) 
        X, y = us.fit_resample(X, y)
        
    model_public_vs_residential = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    
    # RETAIL vs RESIDENTIAL
    data = pd.concat([_data,_labels], axis=1)
    data_1 = data[data.CLASE == "RETAIL"]
    data_2 = data[data.CLASE == "RESIDENTIAL"]
    data = pd.concat([data_1, data_2])
    labels = data.CLASE
    data = data.drop('CLASE', axis=1)
    X = data.values
    y = labels.values
    
    if(under):
        sampling_dict = {'RESIDENTIAL': n_us}
        us = RandomUnderSampler(random_state=42, sampling_strategy=sampling_dict) 
        X, y = us.fit_resample(X, y)
    
    model_retail_vs_residential = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    
    print("FIT MODELS: DONE")
    
    # Diccionario que guarda las prob de cada modelo
    dict_prob = {}
    
    # Diccionario clases:
    dict_class = {
        'OFFICEvsRETAIL':['OFFICE','RETAIL'],
        'OFFICEvsPUBLIC':['OFFICE','PUBLIC'],
        'OFFICEvsOTHER':['OFFICE','OTHER'],
        'OFFICEvsINDUSTRIAL':['INDUSTRIAL','OFFICE'],
        'OFFICEvsAGRICULTURE':['AGRICULTURE','OFFICE'],
        'OFFICEvsRESIDENTIAL':['OFFICE','RESIDENTIAL'],
        'AGRICULTUREvsRETAIL':['AGRICULTURE','RETAIL'],
        'AGRICULTUREvsPUBLIC':['AGRICULTURE','PUBLIC'],
        'AGRICULTUREvsOTHER':['AGRICULTURE','OTHER'],
        'AGRICULTUREvsINDUSTRIAL':['AGRICULTURE','INDUSTRIAL'],
        'AGRICULTUREvsRESIDENTIAL':['AGRICULTURE','RESIDENTIAL'],
        'INDUSTRIALvsRETAIL':['INDUSTRIAL','RETAIL'],
        'INDUSTRIALvsPUBLIC':['INDUSTRIAL','PUBLIC'],
        'INDUSTRIALvsOTHER':['INDUSTRIAL','OTHER'],
        'INDUSTRIALvsRESIDENTIAL':['INDUSTRIAL','RESIDENTIAL'],
        'OTHERvsRETAIL':['OTHER','RETAIL'],
        'OTHERvsPUBLIC':['OTHER','PUBLIC'],
        'OTHERvsRESIDENTIAL':['OTHER','RESIDENTIAL'],
        'PUBLICvsRETAIL':['PUBLIC','RETAIL'],
        'PUBLICvsRESIDENTIAL':['PUBLIC','RESIDENTIAL'],
        'RETAILvsRESIDENTIAL':['RESIDENTIAL','RETAIL']
    }

    # Predicción de las probabilidades:

    # => Sin selección/creación de características:
    
#         dict_prob['OFFICEvsRETAIL'] = model_office_vs_retail.predict_proba(test.values)
#         dict_prob['OFFICEvsPUBLIC'] = model_office_vs_public.predict_proba(test.values)
#         dict_prob['OFFICEvsOTHER'] = model_office_vs_other.predict_proba(test.values)
#         dict_prob['OFFICEvsINDUSTRIAL'] = model_office_vs_industrial.predict_proba(test.values)
#         dict_prob['OFFICEvsAGRICULTURE'] = model_office_vs_agriculture.predict_proba(test.values)
#         dict_prob['OFFICEvsRESIDENTIAL'] = model_office_vs_residential.predict_proba(test.values)
#         dict_prob['AGRICULTUREvsRETAIL'] = model_agriculture_vs_retail.predict_proba(test.values)
#         dict_prob['AGRICULTUREvsPUBLIC'] = model_agriculture_vs_public.predict_proba(test.values)
#         dict_prob['AGRICULTUREvsOTHER'] = model_agriculture_vs_other.predict_proba(test.values)
#         dict_prob['AGRICULTUREvsINDUSTRIAL'] = model_agriculture_vs_industrial.predict_proba(test.values)
#         dict_prob['AGRICULTUREvsRESIDENTIAL'] = model_agriculture_vs_residential.predict_proba(test.values)
#         dict_prob['INDUSTRIALvsRETAIL'] = model_industrial_vs_retail.predict_proba(test.values)
#         dict_prob['INDUSTRIALvsPUBLIC'] = model_industrial_vs_public.predict_proba(test.values)
#         dict_prob['INDUSTRIALvsOTHER'] = model_industrial_vs_other.predict_proba(test.values)
#         dict_prob['INDUSTRIALvsRESIDENTIAL'] = model_industrial_vs_residential.predict_proba(test.values)
#         dict_prob['OTHERvsRETAIL'] = model_other_vs_retail.predict_proba(test.values)
#         dict_prob['OTHERvsPUBLIC'] = model_other_vs_public.predict_proba(test.values)
#         dict_prob['OTHERvsRESIDENTIAL'] = model_other_vs_residential.predict_proba(test.values)
#         dict_prob['PUBLICvsRETAIL'] = model_public_vs_retail.predict_proba(test.values)
#         dict_prob['PUBLICvsRESIDENTIAL'] = model_public_vs_residential.predict_proba(test.values)
#         dict_prob['RETAILvsRESIDENTIAL'] = model_retail_vs_residential.predict_proba(test.values)

    # => Con selección/creación de características:

    aux = test.copy()
    aux = aux.drop('GEOM_R2', axis=1)
    aux['MAXBUILDINGFLOOR2'] = aux['MAXBUILDINGFLOOR']**2
    aux['Q_G_3_0_32'] = aux['Q_G_3_0_3']**2
    dict_prob['OFFICEvsRETAIL'] = model_office_vs_retail.predict_proba(aux.values)

    aux = test.copy()
    aux = aux.drop('GEOM_R3', axis=1)
    aux['Y2'] = aux['Y']**2
    aux['Q_NIR_8_0_92'] = aux['Q_NIR_8_0_9']**2
    aux['Q_B_2_0_32'] = aux['Q_B_2_0_3']**2
    dict_prob['OFFICEvsPUBLIC'] = model_office_vs_public.predict_proba(aux.values)

    aux = test.copy()
    aux = aux.drop('Q_B_2_0_2', axis=1)
    aux = aux.drop('Q_B_2_1_0', axis=1)
    aux = aux.drop('Q_NIR_8_0_8', axis=1)
    aux['CADASTRALQUALITYID2'] = aux['CADASTRALQUALITYID']**2
    aux['Q_NIR_8_0_52'] = aux['Q_NIR_8_0_5']**2
    aux['Q_G_3_0_22'] = aux['Q_G_3_0_2']**2
    dict_prob['OFFICEvsOTHER'] = model_office_vs_other.predict_proba(aux.values)

    aux = test.copy()
    aux = aux.drop('Q_NIR_8_0_8', axis=1)
    aux['Q_B_2_0_62'] = aux['Q_B_2_0_6']**2
    dict_prob['OFFICEvsINDUSTRIAL'] = model_office_vs_industrial.predict_proba(aux.values)

    aux = test.copy()
    aux = aux.drop('Q_G_3_0_2', axis=1)
    aux['Y2'] = aux['Y']**2
    aux['X2'] = aux['X']**2
    aux['Q_NIR_8_0_32'] = aux['Q_NIR_8_0_3']**2
    dict_prob['OFFICEvsAGRICULTURE'] = model_office_vs_agriculture.predict_proba(aux.values)

    dict_prob['OFFICEvsRESIDENTIAL'] = model_office_vs_residential.predict_proba(test.values)

    dict_prob['AGRICULTUREvsRETAIL'] = model_agriculture_vs_retail.predict_proba(test.values)

    dict_prob['AGRICULTUREvsPUBLIC'] = model_agriculture_vs_public.predict_proba(test.values)

    dict_prob['AGRICULTUREvsOTHER'] = model_agriculture_vs_other.predict_proba(test.values)

    dict_prob['AGRICULTUREvsINDUSTRIAL'] = model_agriculture_vs_industrial.predict_proba(test.values)

    dict_prob['AGRICULTUREvsRESIDENTIAL'] = model_agriculture_vs_residential.predict_proba(test.values)

    dict_prob['INDUSTRIALvsRETAIL'] = model_industrial_vs_retail.predict_proba(test.values)

    dict_prob['INDUSTRIALvsPUBLIC'] = model_industrial_vs_public.predict_proba(test.values)

    dict_prob['INDUSTRIALvsOTHER'] = model_industrial_vs_other.predict_proba(test.values)

    dict_prob['INDUSTRIALvsRESIDENTIAL'] = model_industrial_vs_residential.predict_proba(test.values)

    dict_prob['OTHERvsRETAIL'] = model_other_vs_retail.predict_proba(test.values)

    dict_prob['OTHERvsPUBLIC'] = model_other_vs_public.predict_proba(test.values)

    dict_prob['OTHERvsRESIDENTIAL'] = model_other_vs_residential.predict_proba(test.values)

    dict_prob['PUBLICvsRETAIL'] = model_public_vs_retail.predict_proba(test.values)

    dict_prob['PUBLICvsRESIDENTIAL'] = model_public_vs_residential.predict_proba(test.values)

    dict_prob['RETAILvsRESIDENTIAL'] = model_retail_vs_residential.predict_proba(test.values)

    print("PREDICT: DONE")
    
    # ====== AGREGACIÓN: ====== #
    pred = []
    
    for i in range(0, test.shape[0]):

        dict_count = {
            'OFFICE': 0,
            'AGRICULTURE': 0,
            'INDUSTRIAL': 0,
            'OTHER': 0,
            'PUBLIC': 0,
            'RETAIL': 0,
            'RESIDENTIAL': 0
        }

        for key in dict_prob.keys():
            
            # ==> VOTO SIMPLE:
            if (dict_prob[key][i][0]>dict_prob[key][i][1]):
                dict_count[dict_class[key][0]] +=1
            else:
                dict_count[dict_class[key][1]] +=1

            # ==> SUMA PROB DE CADA CLASE:
#             dict_count[dict_class[key][0]] += dict_prob[key][i][0]
#             dict_count[dict_class[key][1]] += dict_prob[key][i][1]

            # ==> PROB VENCEDORA - PROB PERDEDORA:
#             if (dict_prob[key][i][0]>dict_prob[key][i][1]):
#                 dict_count[dict_class[key][0]] += dict_prob[key][i][0]-dict_prob[key][i][1]
#             else:
#                 dict_count[dict_class[key][1]] += dict_prob[key][i][1]-dict_prob[key][i][0]

        predicted_class = max(dict_count.items(), key=operator.itemgetter(1))[0]
        pred.append(predicted_class)
    
    return(pred)
    
def generateOVOModels(_data, _labels, test, under=True, n_us=6000):
    
    # ====== DESCOMPOSICIÓN: ====== #
    # Creación de modelos one-vs-one:
    
    # OFFICE vs RETAIL
    data = pd.concat([_data,_labels], axis=1)
    data_1 = data[data.CLASE == "OFFICE"]
    data_2 = data[data.CLASE == "RETAIL"]
    data = pd.concat([data_1, data_2])
    labels = data.CLASE
    data = data.drop('CLASE', axis=1)
    
    data = data.drop('GEOM_R2', axis=1)
    
    data['MAXBUILDINGFLOOR2'] = data['MAXBUILDINGFLOOR']**2
    data['Q_G_3_0_32'] = data['Q_G_3_0_3']**2
    
    
    X = data.values
    y = labels.values
    model_office_vs_retail = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    dump(model_office_vs_retail, './ovoModels/model_office_vs_retail.sav') 
    
    # OFFICE vs PUBLIC
    data = pd.concat([_data,_labels], axis=1)
    data_1 = data[data.CLASE == "OFFICE"]
    data_2 = data[data.CLASE == "PUBLIC"]
    data = pd.concat([data_1, data_2])
    labels = data.CLASE
    data = data.drop('CLASE', axis=1)
    
    data = data.drop('GEOM_R3', axis=1)
    
    data['Y2'] = data['Y']**2
    data['Q_NIR_8_0_92'] = data['Q_NIR_8_0_9']**2
    data['Q_B_2_0_32'] = data['Q_B_2_0_3']**2
    
    X = data.values
    y = labels.values
    model_office_vs_public = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    dump(model_office_vs_public,'./ovoModels/model_office_vs_public.sav')

    # OFFICE vs OTHER
    data = pd.concat([_data,_labels], axis=1)
    data_1 = data[data.CLASE == "OFFICE"]
    data_2 = data[data.CLASE == "OTHER"]
    data = pd.concat([data_1, data_2])
    labels = data.CLASE
    data = data.drop('CLASE', axis=1)
    
    data = data.drop('Q_B_2_0_2', axis=1)
    data = data.drop('Q_B_2_1_0', axis=1)
    data = data.drop('Q_NIR_8_0_8', axis=1)
    
    data['CADASTRALQUALITYID2'] = data['CADASTRALQUALITYID']**2
    data['Q_NIR_8_0_52'] = data['Q_NIR_8_0_5']**2
    data['Q_G_3_0_22'] = data['Q_G_3_0_2']**2
    
    X = data.values
    y = labels.values
    model_office_vs_other = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    dump(model_office_vs_other,'./ovoModels/model_office_vs_other.sav')
    
    # OFFICE vs INDUSTRIAL
    data = pd.concat([_data,_labels], axis=1)
    data_1 = data[data.CLASE == "OFFICE"]
    data_2 = data[data.CLASE == "INDUSTRIAL"]
    data = pd.concat([data_1, data_2])
    labels = data.CLASE
    data = data.drop('CLASE', axis=1)
    
    data = data.drop('Q_NIR_8_0_8', axis=1)
    
    data['Q_B_2_0_62'] = data['Q_B_2_0_6']**2
    
    X = data.values
    y = labels.values
    model_office_vs_industrial = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    dump(model_office_vs_industrial,'./ovoModels/model_office_vs_industrial.sav')
    
    # OFFICE vs AGRICULTURE
    data = pd.concat([_data,_labels], axis=1)
    data_1 = data[data.CLASE == "OFFICE"]
    data_2 = data[data.CLASE == "AGRICULTURE"]
    data = pd.concat([data_1, data_2])
    labels = data.CLASE
    data = data.drop('CLASE', axis=1)
    
    data = data.drop('Q_G_3_0_2', axis=1)
    
    data['Y2'] = data['Y']**2
    data['X2'] = data['X']**2
    data['Q_NIR_8_0_32'] = data['Q_NIR_8_0_3']**2
    
    X = data.values
    y = labels.values
    model_office_vs_agriculture = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    dump(model_office_vs_agriculture,'./ovoModels/model_office_vs_agriculture.sav')
    
    # OFFICE vs RESIDENTIAL
    data = pd.concat([_data,_labels], axis=1)
    data_1 = data[data.CLASE == "OFFICE"]
    data_2 = data[data.CLASE == "RESIDENTIAL"]
    data = pd.concat([data_1, data_2])
    labels = data.CLASE
    data = data.drop('CLASE', axis=1)
    X = data.values
    y = labels.values
    
    if(under):
        sampling_dict = {'RESIDENTIAL': n_us}
        us = RandomUnderSampler(random_state=42, sampling_strategy=sampling_dict) 
        X, y = us.fit_resample(X, y)
    
    model_office_vs_residential = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    dump(model_office_vs_residential,'./ovoModels/model_office_vs_residential.sav')
    
    # AGRICULTURE vs RETAIL
    data = pd.concat([_data,_labels], axis=1)
    data_1 = data[data.CLASE == "AGRICULTURE"]
    data_2 = data[data.CLASE == "RETAIL"]
    data = pd.concat([data_1, data_2])
    labels = data.CLASE
    data = data.drop('CLASE', axis=1)
    X = data.values
    y = labels.values
    model_agriculture_vs_retail = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    dump(model_agriculture_vs_retail,'./ovoModels/model_agriculture_vs_retail.sav')
    
    # AGRICULTURE vs PUBLIC
    data = pd.concat([_data,_labels], axis=1)
    data_1 = data[data.CLASE == "AGRICULTURE"]
    data_2 = data[data.CLASE == "PUBLIC"]
    data = pd.concat([data_1, data_2])
    labels = data.CLASE
    data = data.drop('CLASE', axis=1)
    X = data.values
    y = labels.values
    model_agriculture_vs_public = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    dump(model_agriculture_vs_public,'./ovoModels/model_agriculture_vs_public.sav')
    
    # AGRICULTURE vs OTHER
    data = pd.concat([_data,_labels], axis=1)
    data_1 = data[data.CLASE == "AGRICULTURE"]
    data_2 = data[data.CLASE == "OTHER"]
    data = pd.concat([data_1, data_2])
    labels = data.CLASE
    data = data.drop('CLASE', axis=1)
    X = data.values
    y = labels.values
    model_agriculture_vs_other = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    dump(model_agriculture_vs_other,'./ovoModels/model_agriculture_vs_other.sav')
    
    # AGRICULTURE vs INDUSTRIAL
    data = pd.concat([_data,_labels], axis=1)
    data_1 = data[data.CLASE == "AGRICULTURE"]
    data_2 = data[data.CLASE == "INDUSTRIAL"]
    data = pd.concat([data_1, data_2])
    labels = data.CLASE
    data = data.drop('CLASE', axis=1)
    X = data.values
    y = labels.values
    model_agriculture_vs_industrial = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    dump(model_agriculture_vs_industrial,'./ovoModels/model_agriculture_vs_industrial.sav')
    
    # AGRICULTURE vs RESIDENTIAL
    data = pd.concat([_data,_labels], axis=1)
    data_1 = data[data.CLASE == "AGRICULTURE"]
    data_2 = data[data.CLASE == "RESIDENTIAL"]
    data = pd.concat([data_1, data_2])
    labels = data.CLASE
    data = data.drop('CLASE', axis=1)
    X = data.values
    y = labels.values
    
    if(under):
        sampling_dict = {'RESIDENTIAL': n_us}
        us = RandomUnderSampler(random_state=42, sampling_strategy=sampling_dict) 
        X, y = us.fit_resample(X, y)
    
    model_agriculture_vs_residential = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    dump(model_agriculture_vs_residential,'./ovoModels/model_agriculture_vs_residential.sav')
    
    # INDUSTRIAL vs RETAIL
    data = pd.concat([_data,_labels], axis=1)
    data_1 = data[data.CLASE == "INDUSTRIAL"]
    data_2 = data[data.CLASE == "RETAIL"]
    data = pd.concat([data_1, data_2])
    labels = data.CLASE
    data = data.drop('CLASE', axis=1)
    X = data.values
    y = labels.values
    model_industrial_vs_retail = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    dump(model_industrial_vs_retail,'./ovoModels/model_industrial_vs_retail.sav')
    
    # INDUSTRIAL vs PUBLIC
    data = pd.concat([_data,_labels], axis=1)
    data_1 = data[data.CLASE == "INDUSTRIAL"]
    data_2 = data[data.CLASE == "PUBLIC"]
    data = pd.concat([data_1, data_2])
    labels = data.CLASE
    data = data.drop('CLASE', axis=1)
    X = data.values
    y = labels.values
    model_industrial_vs_public = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    dump(model_industrial_vs_public,'./ovoModels/model_industrial_vs_public.sav')
    
    # INDUSTRIAL vs OTHER
    data = pd.concat([_data,_labels], axis=1)
    data_1 = data[data.CLASE == "INDUSTRIAL"]
    data_2 = data[data.CLASE == "OTHER"]
    data = pd.concat([data_1, data_2])
    labels = data.CLASE
    data = data.drop('CLASE', axis=1)
    X = data.values
    y = labels.values
    model_industrial_vs_other = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    dump(model_industrial_vs_other,'./ovoModels/model_industrial_vs_other.sav')
    
    # INDUSTRIAL vs RESIDENTIAL
    data = pd.concat([_data,_labels], axis=1)
    data_1 = data[data.CLASE == "INDUSTRIAL"]
    data_2 = data[data.CLASE == "RESIDENTIAL"]
    data = pd.concat([data_1, data_2])
    labels = data.CLASE
    data = data.drop('CLASE', axis=1)
    X = data.values
    y = labels.values
    
    if(under):
        sampling_dict = {'RESIDENTIAL': n_us}
        us = RandomUnderSampler(random_state=42, sampling_strategy=sampling_dict) 
        X, y = us.fit_resample(X, y)
    
    model_industrial_vs_residential = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    dump(model_industrial_vs_residential,'./ovoModels/model_industrial_vs_residential.sav')
    
    # OTHER vs RETAIL
    data = pd.concat([_data,_labels], axis=1)
    data_1 = data[data.CLASE == "OTHER"]
    data_2 = data[data.CLASE == "RETAIL"]
    data = pd.concat([data_1, data_2])
    labels = data.CLASE
    data = data.drop('CLASE', axis=1)
    X = data.values
    y = labels.values
    model_other_vs_retail = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    dump(model_other_vs_retail,'./ovoModels/model_other_vs_retail.sav')
    
    # OTHER vs PUBLIC
    data = pd.concat([_data,_labels], axis=1)
    data_1 = data[data.CLASE == "OTHER"]
    data_2 = data[data.CLASE == "PUBLIC"]
    data = pd.concat([data_1, data_2])
    labels = data.CLASE
    data = data.drop('CLASE', axis=1)
    X = data.values
    y = labels.values
    model_other_vs_public = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    dump(model_other_vs_public,'./ovoModels/model_other_vs_public.sav')
    
    # OTHER vs RESIDENTIAL
    data = pd.concat([_data,_labels], axis=1)
    data_1 = data[data.CLASE == "OTHER"]
    data_2 = data[data.CLASE == "RESIDENTIAL"]
    data = pd.concat([data_1, data_2])
    labels = data.CLASE
    data = data.drop('CLASE', axis=1)
    X = data.values
    y = labels.values
    
    if(under):
        sampling_dict = {'RESIDENTIAL': n_us}
        us = RandomUnderSampler(random_state=42, sampling_strategy=sampling_dict) 
        X, y = us.fit_resample(X, y)
    
    model_other_vs_residential = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    dump(model_other_vs_residential,'./ovoModels/model_other_vs_residential.sav')
    
    # PUBLIC vs RETAIL
    data = pd.concat([_data,_labels], axis=1)
    data_1 = data[data.CLASE == "PUBLIC"]
    data_2 = data[data.CLASE == "RETAIL"]
    data = pd.concat([data_1, data_2])
    labels = data.CLASE
    data = data.drop('CLASE', axis=1)
    X = data.values
    y = labels.values
    model_public_vs_retail = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    dump(model_public_vs_retail,'./ovoModels/model_public_vs_retail.sav')
    
    # PUBLIC vs RESIDENTIAL
    data = pd.concat([_data,_labels], axis=1)
    data_1 = data[data.CLASE == "PUBLIC"]
    data_2 = data[data.CLASE == "RESIDENTIAL"]
    data = pd.concat([data_1, data_2])
    labels = data.CLASE
    data = data.drop('CLASE', axis=1)
    X = data.values
    y = labels.values
    
    if(under):
        sampling_dict = {'RESIDENTIAL': n_us}
        us = RandomUnderSampler(random_state=42, sampling_strategy=sampling_dict) 
        X, y = us.fit_resample(X, y)
        
    model_public_vs_residential = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    dump(model_public_vs_residential,'./ovoModels/model_public_vs_residential.sav')
    
    # RETAIL vs RESIDENTIAL
    data = pd.concat([_data,_labels], axis=1)
    data_1 = data[data.CLASE == "RETAIL"]
    data_2 = data[data.CLASE == "RESIDENTIAL"]
    data = pd.concat([data_1, data_2])
    labels = data.CLASE
    data = data.drop('CLASE', axis=1)
    X = data.values
    y = labels.values
    
    if(under):
        sampling_dict = {'RESIDENTIAL': n_us}
        us = RandomUnderSampler(random_state=42, sampling_strategy=sampling_dict) 
        X, y = us.fit_resample(X, y)
    
    model_retail_vs_residential = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    dump(model_retail_vs_residential,'./ovoModels/model_retail_vs_residential.sav')
    
    print("FIT MODELS: DONE")


    
def predictOVOVotoSimple(test):
    
    # Carga de modelos preentrenados:
    model_office_vs_retail = load('./ovoModels/model_office_vs_retail.sav')
    model_office_vs_public = load('./ovoModels/model_office_vs_public.sav')
    model_office_vs_other = load('./ovoModels/model_office_vs_other.sav')
    model_office_vs_industrial = load('./ovoModels/model_office_vs_industrial.sav')
    model_office_vs_agriculture = load('./ovoModels/model_office_vs_agriculture.sav')
    model_office_vs_residential = load('./ovoModels/model_office_vs_residential.sav')
    model_agriculture_vs_retail = load('./ovoModels/model_agriculture_vs_retail.sav')
    model_agriculture_vs_public = load('./ovoModels/model_agriculture_vs_public.sav')
    model_agriculture_vs_other = load('./ovoModels/model_agriculture_vs_other.sav')
    model_agriculture_vs_industrial = load('./ovoModels/model_agriculture_vs_industrial.sav')
    model_agriculture_vs_residential = load('./ovoModels/model_agriculture_vs_residential.sav')
    model_industrial_vs_retail = load('./ovoModels/model_industrial_vs_retail.sav')
    model_industrial_vs_public = load('./ovoModels/model_industrial_vs_public.sav')
    model_industrial_vs_other = load('./ovoModels/model_industrial_vs_other.sav')
    model_industrial_vs_residential = load('./ovoModels/model_industrial_vs_residential.sav')
    model_other_vs_retail = load('./ovoModels/model_other_vs_retail.sav')
    model_other_vs_public = load('./ovoModels/model_other_vs_public.sav')
    model_other_vs_residential = load('./ovoModels/model_other_vs_residential.sav')
    model_public_vs_retail = load('./ovoModels/model_public_vs_retail.sav')
    model_public_vs_residential = load('./ovoModels/model_public_vs_residential.sav')
    model_retail_vs_residential = load('./ovoModels/model_retail_vs_residential.sav')

    print("DONE: LOAD MODELS")
    
    # Diccionario que guarda las prob de cada modelo
    dict_prob = {}
    
    # Diccionario clases:
    dict_class = {
        'OFFICEvsRETAIL':['OFFICE','RETAIL'],
        'OFFICEvsPUBLIC':['OFFICE','PUBLIC'],
        'OFFICEvsOTHER':['OFFICE','OTHER'],
        'OFFICEvsINDUSTRIAL':['INDUSTRIAL','OFFICE'],
        'OFFICEvsAGRICULTURE':['AGRICULTURE','OFFICE'],
        'OFFICEvsRESIDENTIAL':['OFFICE','RESIDENTIAL'],
        'AGRICULTUREvsRETAIL':['AGRICULTURE','RETAIL'],
        'AGRICULTUREvsPUBLIC':['AGRICULTURE','PUBLIC'],
        'AGRICULTUREvsOTHER':['AGRICULTURE','OTHER'],
        'AGRICULTUREvsINDUSTRIAL':['AGRICULTURE','INDUSTRIAL'],
        'AGRICULTUREvsRESIDENTIAL':['AGRICULTURE','RESIDENTIAL'],
        'INDUSTRIALvsRETAIL':['INDUSTRIAL','RETAIL'],
        'INDUSTRIALvsPUBLIC':['INDUSTRIAL','PUBLIC'],
        'INDUSTRIALvsOTHER':['INDUSTRIAL','OTHER'],
        'INDUSTRIALvsRESIDENTIAL':['INDUSTRIAL','RESIDENTIAL'],
        'OTHERvsRETAIL':['OTHER','RETAIL'],
        'OTHERvsPUBLIC':['OTHER','PUBLIC'],
        'OTHERvsRESIDENTIAL':['OTHER','RESIDENTIAL'],
        'PUBLICvsRETAIL':['PUBLIC','RETAIL'],
        'PUBLICvsRESIDENTIAL':['PUBLIC','RESIDENTIAL'],
        'RETAILvsRESIDENTIAL':['RESIDENTIAL','RETAIL']
    }

    # Predicción de las probabilidades:

    # => Sin selección/creación de características:
    
#         dict_prob['OFFICEvsRETAIL'] = model_office_vs_retail.predict_proba(test.values)
#         dict_prob['OFFICEvsPUBLIC'] = model_office_vs_public.predict_proba(test.values)
#         dict_prob['OFFICEvsOTHER'] = model_office_vs_other.predict_proba(test.values)
#         dict_prob['OFFICEvsINDUSTRIAL'] = model_office_vs_industrial.predict_proba(test.values)
#         dict_prob['OFFICEvsAGRICULTURE'] = model_office_vs_agriculture.predict_proba(test.values)
#         dict_prob['OFFICEvsRESIDENTIAL'] = model_office_vs_residential.predict_proba(test.values)
#         dict_prob['AGRICULTUREvsRETAIL'] = model_agriculture_vs_retail.predict_proba(test.values)
#         dict_prob['AGRICULTUREvsPUBLIC'] = model_agriculture_vs_public.predict_proba(test.values)
#         dict_prob['AGRICULTUREvsOTHER'] = model_agriculture_vs_other.predict_proba(test.values)
#         dict_prob['AGRICULTUREvsINDUSTRIAL'] = model_agriculture_vs_industrial.predict_proba(test.values)
#         dict_prob['AGRICULTUREvsRESIDENTIAL'] = model_agriculture_vs_residential.predict_proba(test.values)
#         dict_prob['INDUSTRIALvsRETAIL'] = model_industrial_vs_retail.predict_proba(test.values)
#         dict_prob['INDUSTRIALvsPUBLIC'] = model_industrial_vs_public.predict_proba(test.values)
#         dict_prob['INDUSTRIALvsOTHER'] = model_industrial_vs_other.predict_proba(test.values)
#         dict_prob['INDUSTRIALvsRESIDENTIAL'] = model_industrial_vs_residential.predict_proba(test.values)
#         dict_prob['OTHERvsRETAIL'] = model_other_vs_retail.predict_proba(test.values)
#         dict_prob['OTHERvsPUBLIC'] = model_other_vs_public.predict_proba(test.values)
#         dict_prob['OTHERvsRESIDENTIAL'] = model_other_vs_residential.predict_proba(test.values)
#         dict_prob['PUBLICvsRETAIL'] = model_public_vs_retail.predict_proba(test.values)
#         dict_prob['PUBLICvsRESIDENTIAL'] = model_public_vs_residential.predict_proba(test.values)
#         dict_prob['RETAILvsRESIDENTIAL'] = model_retail_vs_residential.predict_proba(test.values)

    # => Con selección/creación de características:

    aux = test.copy()
    aux = aux.drop('GEOM_R2', axis=1)
    aux['MAXBUILDINGFLOOR2'] = aux['MAXBUILDINGFLOOR']**2
    aux['Q_G_3_0_32'] = aux['Q_G_3_0_3']**2
    dict_prob['OFFICEvsRETAIL'] = model_office_vs_retail.predict_proba(aux.values)

    aux = test.copy()
    aux = aux.drop('GEOM_R3', axis=1)
    aux['Y2'] = aux['Y']**2
    aux['Q_NIR_8_0_92'] = aux['Q_NIR_8_0_9']**2
    aux['Q_B_2_0_32'] = aux['Q_B_2_0_3']**2
    dict_prob['OFFICEvsPUBLIC'] = model_office_vs_public.predict_proba(aux.values)

    aux = test.copy()
    aux = aux.drop('Q_B_2_0_2', axis=1)
    aux = aux.drop('Q_B_2_1_0', axis=1)
    aux = aux.drop('Q_NIR_8_0_8', axis=1)
    aux['CADASTRALQUALITYID2'] = aux['CADASTRALQUALITYID']**2
    aux['Q_NIR_8_0_52'] = aux['Q_NIR_8_0_5']**2
    aux['Q_G_3_0_22'] = aux['Q_G_3_0_2']**2
    dict_prob['OFFICEvsOTHER'] = model_office_vs_other.predict_proba(aux.values)

    aux = test.copy()
    aux = aux.drop('Q_NIR_8_0_8', axis=1)
    aux['Q_B_2_0_62'] = aux['Q_B_2_0_6']**2
    dict_prob['OFFICEvsINDUSTRIAL'] = model_office_vs_industrial.predict_proba(aux.values)

    aux = test.copy()
    aux = aux.drop('Q_G_3_0_2', axis=1)
    aux['Y2'] = aux['Y']**2
    aux['X2'] = aux['X']**2
    aux['Q_NIR_8_0_32'] = aux['Q_NIR_8_0_3']**2
    dict_prob['OFFICEvsAGRICULTURE'] = model_office_vs_agriculture.predict_proba(aux.values)

    dict_prob['OFFICEvsRESIDENTIAL'] = model_office_vs_residential.predict_proba(test.values)

    dict_prob['AGRICULTUREvsRETAIL'] = model_agriculture_vs_retail.predict_proba(test.values)

    dict_prob['AGRICULTUREvsPUBLIC'] = model_agriculture_vs_public.predict_proba(test.values)

    dict_prob['AGRICULTUREvsOTHER'] = model_agriculture_vs_other.predict_proba(test.values)

    dict_prob['AGRICULTUREvsINDUSTRIAL'] = model_agriculture_vs_industrial.predict_proba(test.values)

    dict_prob['AGRICULTUREvsRESIDENTIAL'] = model_agriculture_vs_residential.predict_proba(test.values)

    dict_prob['INDUSTRIALvsRETAIL'] = model_industrial_vs_retail.predict_proba(test.values)

    dict_prob['INDUSTRIALvsPUBLIC'] = model_industrial_vs_public.predict_proba(test.values)

    dict_prob['INDUSTRIALvsOTHER'] = model_industrial_vs_other.predict_proba(test.values)

    dict_prob['INDUSTRIALvsRESIDENTIAL'] = model_industrial_vs_residential.predict_proba(test.values)

    dict_prob['OTHERvsRETAIL'] = model_other_vs_retail.predict_proba(test.values)

    dict_prob['OTHERvsPUBLIC'] = model_other_vs_public.predict_proba(test.values)

    dict_prob['OTHERvsRESIDENTIAL'] = model_other_vs_residential.predict_proba(test.values)

    dict_prob['PUBLICvsRETAIL'] = model_public_vs_retail.predict_proba(test.values)

    dict_prob['PUBLICvsRESIDENTIAL'] = model_public_vs_residential.predict_proba(test.values)

    dict_prob['RETAILvsRESIDENTIAL'] = model_retail_vs_residential.predict_proba(test.values)

    print("PREDICT: DONE")
    
    # ====== AGREGACIÓN: ====== #
    pred = []
    
    for i in range(0, test.shape[0]):

        dict_count = {
            'OFFICE': 0,
            'AGRICULTURE': 0,
            'INDUSTRIAL': 0,
            'OTHER': 0,
            'PUBLIC': 0,
            'RETAIL': 0,
            'RESIDENTIAL': 0
        }

        for key in dict_prob.keys():
            
            # ==> VOTO SIMPLE:
            if (dict_prob[key][i][0]>dict_prob[key][i][1]):
                dict_count[dict_class[key][0]] +=1
            else:
                dict_count[dict_class[key][1]] +=1

            # ==> SUMA PROB DE CADA CLASE:
#             dict_count[dict_class[key][0]] += dict_prob[key][i][0]
#             dict_count[dict_class[key][1]] += dict_prob[key][i][1]

            # ==> PROB VENCEDORA - PROB PERDEDORA:
#             if (dict_prob[key][i][0]>dict_prob[key][i][1]):
#                 dict_count[dict_class[key][0]] += dict_prob[key][i][0]-dict_prob[key][i][1]
#             else:
#                 dict_count[dict_class[key][1]] += dict_prob[key][i][1]-dict_prob[key][i][0]

        predicted_class = max(dict_count.items(), key=operator.itemgetter(1))[0]
        pred.append(predicted_class)
    
    return(pred)
    
    
# def OVO(_data, _labels, test, under=True, n_us=6000, voto=0):
    
#     print(n_us)
#     # Diccionario que guarda las prob de cada modelo
#     dict_prob = {}
    
#     # ====== DESCOMPOSICIÓN: ====== #
#     # Creación de modelos one-vs-one:
    
#     # OFFICE vs RETAIL
#     data = pd.concat([_data,_labels], axis=1)
#     data_1 = data[data.CLASE == "OFFICE"]
#     data_2 = data[data.CLASE == "RETAIL"]
#     data = pd.concat([data_1, data_2])
#     labels = data.CLASE
#     data = data.drop('CLASE', axis=1)
    
#     data = data.drop('GEOM_R2', axis=1)
    
#     data['MAXBUILDINGFLOOR2'] = data['MAXBUILDINGFLOOR']**2
#     data['Q_G_3_0_32'] = data['Q_G_3_0_3']**2
    
    
#     X = data.values
#     y = labels.values
#     model_office_vs_retail = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    
    
#     # OFFICE vs PUBLIC
#     data = pd.concat([_data,_labels], axis=1)
#     data_1 = data[data.CLASE == "OFFICE"]
#     data_2 = data[data.CLASE == "PUBLIC"]
#     data = pd.concat([data_1, data_2])
#     labels = data.CLASE
#     data = data.drop('CLASE', axis=1)
    
#     data = data.drop('GEOM_R3', axis=1)
    
#     data['Y2'] = data['Y']**2
#     data['Q_NIR_8_0_92'] = data['Q_NIR_8_0_9']**2
#     data['Q_B_2_0_32'] = data['Q_B_2_0_3']**2
    
#     X = data.values
#     y = labels.values
#     model_office_vs_public = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    

#     # OFFICE vs OTHER
#     data = pd.concat([_data,_labels], axis=1)
#     data_1 = data[data.CLASE == "OFFICE"]
#     data_2 = data[data.CLASE == "OTHER"]
#     data = pd.concat([data_1, data_2])
#     labels = data.CLASE
#     data = data.drop('CLASE', axis=1)
    
#     data = data.drop('Q_B_2_0_2', axis=1)
#     data = data.drop('Q_B_2_1_0', axis=1)
#     data = data.drop('Q_NIR_8_0_8', axis=1)
    
#     data['CADASTRALQUALITYID2'] = data['CADASTRALQUALITYID']**2
#     data['Q_NIR_8_0_52'] = data['Q_NIR_8_0_5']**2
#     data['Q_G_3_0_22'] = data['Q_G_3_0_2']**2
    
#     X = data.values
#     y = labels.values
#     model_office_vs_other = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    
    
#     # OFFICE vs INDUSTRIAL
#     data = pd.concat([_data,_labels], axis=1)
#     data_1 = data[data.CLASE == "OFFICE"]
#     data_2 = data[data.CLASE == "INDUSTRIAL"]
#     data = pd.concat([data_1, data_2])
#     labels = data.CLASE
#     data = data.drop('CLASE', axis=1)
    
#     data = data.drop('Q_NIR_8_0_8', axis=1)
    
#     data['Q_B_2_0_62'] = data['Q_B_2_0_6']**2
    
#     X = data.values
#     y = labels.values
#     model_office_vs_industrial = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    
    
#     # OFFICE vs AGRICULTURE
#     data = pd.concat([_data,_labels], axis=1)
#     data_1 = data[data.CLASE == "OFFICE"]
#     data_2 = data[data.CLASE == "AGRICULTURE"]
#     data = pd.concat([data_1, data_2])
#     labels = data.CLASE
#     data = data.drop('CLASE', axis=1)
    
#     data = data.drop('Q_G_3_0_2', axis=1)
    
#     data['Y2'] = data['Y']**2
#     data['X2'] = data['X']**2
#     data['Q_NIR_8_0_32'] = data['Q_NIR_8_0_3']**2
    
#     X = data.values
#     y = labels.values
#     model_office_vs_agriculture = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    
    
#     # OFFICE vs RESIDENTIAL
#     data = pd.concat([_data,_labels], axis=1)
#     data_1 = data[data.CLASE == "OFFICE"]
#     data_2 = data[data.CLASE == "RESIDENTIAL"]
#     data = pd.concat([data_1, data_2])
#     labels = data.CLASE
#     data = data.drop('CLASE', axis=1)
#     X = data.values
#     y = labels.values
    
#     if(under):
#         sampling_dict = {'RESIDENTIAL': n_us}
#         us = RandomUnderSampler(random_state=42, sampling_strategy=sampling_dict) 
#         X, y = us.fit_resample(X, y)
    
#     model_office_vs_residential = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    
    
#     # AGRICULTURE vs RETAIL
#     data = pd.concat([_data,_labels], axis=1)
#     data_1 = data[data.CLASE == "AGRICULTURE"]
#     data_2 = data[data.CLASE == "RETAIL"]
#     data = pd.concat([data_1, data_2])
#     labels = data.CLASE
#     data = data.drop('CLASE', axis=1)
#     X = data.values
#     y = labels.values
#     model_agriculture_vs_retail = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    
    
#     # AGRICULTURE vs PUBLIC
#     data = pd.concat([_data,_labels], axis=1)
#     data_1 = data[data.CLASE == "AGRICULTURE"]
#     data_2 = data[data.CLASE == "PUBLIC"]
#     data = pd.concat([data_1, data_2])
#     labels = data.CLASE
#     data = data.drop('CLASE', axis=1)
#     X = data.values
#     y = labels.values
#     model_agriculture_vs_public = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    
    
#     # AGRICULTURE vs OTHER
#     data = pd.concat([_data,_labels], axis=1)
#     data_1 = data[data.CLASE == "AGRICULTURE"]
#     data_2 = data[data.CLASE == "OTHER"]
#     data = pd.concat([data_1, data_2])
#     labels = data.CLASE
#     data = data.drop('CLASE', axis=1)
#     X = data.values
#     y = labels.values
#     model_agriculture_vs_other = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    
    
#     # AGRICULTURE vs INDUSTRIAL
#     data = pd.concat([_data,_labels], axis=1)
#     data_1 = data[data.CLASE == "AGRICULTURE"]
#     data_2 = data[data.CLASE == "INDUSTRIAL"]
#     data = pd.concat([data_1, data_2])
#     labels = data.CLASE
#     data = data.drop('CLASE', axis=1)
#     X = data.values
#     y = labels.values
#     model_agriculture_vs_industrial = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    
    
#     # AGRICULTURE vs RESIDENTIAL
#     data = pd.concat([_data,_labels], axis=1)
#     data_1 = data[data.CLASE == "AGRICULTURE"]
#     data_2 = data[data.CLASE == "RESIDENTIAL"]
#     data = pd.concat([data_1, data_2])
#     labels = data.CLASE
#     data = data.drop('CLASE', axis=1)
#     X = data.values
#     y = labels.values
    
#     if(under):
#         sampling_dict = {'RESIDENTIAL': n_us}
#         us = RandomUnderSampler(random_state=42, sampling_strategy=sampling_dict) 
#         X, y = us.fit_resample(X, y)
    
#     model_agriculture_vs_residential = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    
    
#     # INDUSTRIAL vs RETAIL
#     data = pd.concat([_data,_labels], axis=1)
#     data_1 = data[data.CLASE == "INDUSTRIAL"]
#     data_2 = data[data.CLASE == "RETAIL"]
#     data = pd.concat([data_1, data_2])
#     labels = data.CLASE
#     data = data.drop('CLASE', axis=1)
#     X = data.values
#     y = labels.values
#     model_industrial_vs_retail = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    
    
#     # INDUSTRIAL vs PUBLIC
#     data = pd.concat([_data,_labels], axis=1)
#     data_1 = data[data.CLASE == "INDUSTRIAL"]
#     data_2 = data[data.CLASE == "PUBLIC"]
#     data = pd.concat([data_1, data_2])
#     labels = data.CLASE
#     data = data.drop('CLASE', axis=1)
#     X = data.values
#     y = labels.values
#     model_industrial_vs_public = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    
    
#     # INDUSTRIAL vs OTHER
#     data = pd.concat([_data,_labels], axis=1)
#     data_1 = data[data.CLASE == "INDUSTRIAL"]
#     data_2 = data[data.CLASE == "OTHER"]
#     data = pd.concat([data_1, data_2])
#     labels = data.CLASE
#     data = data.drop('CLASE', axis=1)
#     X = data.values
#     y = labels.values
#     model_industrial_vs_other = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    
    
#     # INDUSTRIAL vs RESIDENTIAL
#     data = pd.concat([_data,_labels], axis=1)
#     data_1 = data[data.CLASE == "INDUSTRIAL"]
#     data_2 = data[data.CLASE == "RESIDENTIAL"]
#     data = pd.concat([data_1, data_2])
#     labels = data.CLASE
#     data = data.drop('CLASE', axis=1)
#     X = data.values
#     y = labels.values
    
#     if(under):
#         sampling_dict = {'RESIDENTIAL': n_us}
#         us = RandomUnderSampler(random_state=42, sampling_strategy=sampling_dict) 
#         X, y = us.fit_resample(X, y)
    
#     model_industrial_vs_residential = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    
    
#     # OTHER vs RETAIL
#     data = pd.concat([_data,_labels], axis=1)
#     data_1 = data[data.CLASE == "OTHER"]
#     data_2 = data[data.CLASE == "RETAIL"]
#     data = pd.concat([data_1, data_2])
#     labels = data.CLASE
#     data = data.drop('CLASE', axis=1)
#     X = data.values
#     y = labels.values
#     model_other_vs_retail = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    
    
#     # OTHER vs PUBLIC
#     data = pd.concat([_data,_labels], axis=1)
#     data_1 = data[data.CLASE == "OTHER"]
#     data_2 = data[data.CLASE == "PUBLIC"]
#     data = pd.concat([data_1, data_2])
#     labels = data.CLASE
#     data = data.drop('CLASE', axis=1)
#     X = data.values
#     y = labels.values
#     model_other_vs_public = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    
    
#     # OTHER vs RESIDENTIAL
#     data = pd.concat([_data,_labels], axis=1)
#     data_1 = data[data.CLASE == "OTHER"]
#     data_2 = data[data.CLASE == "RESIDENTIAL"]
#     data = pd.concat([data_1, data_2])
#     labels = data.CLASE
#     data = data.drop('CLASE', axis=1)
#     X = data.values
#     y = labels.values
    
#     if(under):
#         sampling_dict = {'RESIDENTIAL': n_us}
#         us = RandomUnderSampler(random_state=42, sampling_strategy=sampling_dict) 
#         X, y = us.fit_resample(X, y)
    
#     model_other_vs_residential = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    
    
#     # PUBLIC vs RETAIL
#     data = pd.concat([_data,_labels], axis=1)
#     data_1 = data[data.CLASE == "PUBLIC"]
#     data_2 = data[data.CLASE == "RETAIL"]
#     data = pd.concat([data_1, data_2])
#     labels = data.CLASE
#     data = data.drop('CLASE', axis=1)
#     X = data.values
#     y = labels.values
#     model_public_vs_retail = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    
    
#     # PUBLIC vs RESIDENTIAL
#     data = pd.concat([_data,_labels], axis=1)
#     data_1 = data[data.CLASE == "PUBLIC"]
#     data_2 = data[data.CLASE == "RESIDENTIAL"]
#     data = pd.concat([data_1, data_2])
#     labels = data.CLASE
#     data = data.drop('CLASE', axis=1)
#     X = data.values
#     y = labels.values
    
#     if(under):
#         sampling_dict = {'RESIDENTIAL': n_us}
#         us = RandomUnderSampler(random_state=42, sampling_strategy=sampling_dict) 
#         X, y = us.fit_resample(X, y)
        
#     model_public_vs_residential = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    
    
#     # RETAIL vs RESIDENTIAL
#     data = pd.concat([_data,_labels], axis=1)
#     data_1 = data[data.CLASE == "RETAIL"]
#     data_2 = data[data.CLASE == "RESIDENTIAL"]
#     data = pd.concat([data_1, data_2])
#     labels = data.CLASE
#     data = data.drop('CLASE', axis=1)
#     X = data.values
#     y = labels.values
    
#     if(under):
#         sampling_dict = {'RESIDENTIAL': n_us}
#         us = RandomUnderSampler(random_state=42, sampling_strategy=sampling_dict) 
#         X, y = us.fit_resample(X, y)
    
#     model_retail_vs_residential = RandomForestClassifier(n_jobs=-1, n_estimators=500).fit(X,y)
    
    
#     print("FIT MODELS: DONE")
    
#     # Diccionario clases:
#     dict_class = {
#         'OFFICEvsRETAIL':['OFFICE','RETAIL'],
#         'OFFICEvsPUBLIC':['OFFICE','PUBLIC'],
#         'OFFICEvsOTHER':['OFFICE','OTHER'],
#         'OFFICEvsINDUSTRIAL':['INDUSTRIAL','OFFICE'],
#         'OFFICEvsAGRICULTURE':['AGRICULTURE','OFFICE'],
#         'OFFICEvsRESIDENTIAL':['OFFICE','RESIDENTIAL'],
#         'AGRICULTUREvsRETAIL':['AGRICULTURE','RETAIL'],
#         'AGRICULTUREvsPUBLIC':['AGRICULTURE','PUBLIC'],
#         'AGRICULTUREvsOTHER':['AGRICULTURE','OTHER'],
#         'AGRICULTUREvsINDUSTRIAL':['AGRICULTURE','INDUSTRIAL'],
#         'AGRICULTUREvsRESIDENTIAL':['AGRICULTURE','RESIDENTIAL'],
#         'INDUSTRIALvsRETAIL':['INDUSTRIAL','RETAIL'],
#         'INDUSTRIALvsPUBLIC':['INDUSTRIAL','PUBLIC'],
#         'INDUSTRIALvsOTHER':['INDUSTRIAL','OTHER'],
#         'INDUSTRIALvsRESIDENTIAL':['INDUSTRIAL','RESIDENTIAL'],
#         'OTHERvsRETAIL':['OTHER','RETAIL'],
#         'OTHERvsPUBLIC':['OTHER','PUBLIC'],
#         'OTHERvsRESIDENTIAL':['OTHER','RESIDENTIAL'],
#         'PUBLICvsRETAIL':['PUBLIC','RETAIL'],
#         'PUBLICvsRESIDENTIAL':['PUBLIC','RESIDENTIAL'],
#         'RETAILvsRESIDENTIAL':['RESIDENTIAL','RETAIL']
#     }
#     #print(dict_class)
    
#     # Diccionario de votos:
#     dict_count = {
#         'OFFICE': 0,
#         'AGRICULTURE': 0,
#         'INDUSTRIAL': 0,
#         'OTHER': 0,
#         'PUBLIC': 0,
#         'RETAIL': 0,
#         'RESIDENTIAL': 0
#     }
    
    
#     # ====== AGREGACIÓN: ====== #

#     if(voto == 0):
#         # ==> Aproximación más simple: VOTO

#         pred = []

#         # Predicción de las probabilidades:
        
#         # => Sin selección/creación de características:
# #         dict_prob['OFFICEvsRETAIL'] = model_office_vs_retail.predict_proba(test.values)
# #         dict_prob['OFFICEvsPUBLIC'] = model_office_vs_public.predict_proba(test.values)
# #         dict_prob['OFFICEvsOTHER'] = model_office_vs_other.predict_proba(test.values)
# #         dict_prob['OFFICEvsINDUSTRIAL'] = model_office_vs_industrial.predict_proba(test.values)
# #         dict_prob['OFFICEvsAGRICULTURE'] = model_office_vs_agriculture.predict_proba(test.values)
# #         dict_prob['OFFICEvsRESIDENTIAL'] = model_office_vs_residential.predict_proba(test.values)
# #         dict_prob['AGRICULTUREvsRETAIL'] = model_agriculture_vs_retail.predict_proba(test.values)
# #         dict_prob['AGRICULTUREvsPUBLIC'] = model_agriculture_vs_public.predict_proba(test.values)
# #         dict_prob['AGRICULTUREvsOTHER'] = model_agriculture_vs_other.predict_proba(test.values)
# #         dict_prob['AGRICULTUREvsINDUSTRIAL'] = model_agriculture_vs_industrial.predict_proba(test.values)
# #         dict_prob['AGRICULTUREvsRESIDENTIAL'] = model_agriculture_vs_residential.predict_proba(test.values)
# #         dict_prob['INDUSTRIALvsRETAIL'] = model_industrial_vs_retail.predict_proba(test.values)
# #         dict_prob['INDUSTRIALvsPUBLIC'] = model_industrial_vs_public.predict_proba(test.values)
# #         dict_prob['INDUSTRIALvsOTHER'] = model_industrial_vs_other.predict_proba(test.values)
# #         dict_prob['INDUSTRIALvsRESIDENTIAL'] = model_industrial_vs_residential.predict_proba(test.values)
# #         dict_prob['OTHERvsRETAIL'] = model_other_vs_retail.predict_proba(test.values)
# #         dict_prob['OTHERvsPUBLIC'] = model_other_vs_public.predict_proba(test.values)
# #         dict_prob['OTHERvsRESIDENTIAL'] = model_other_vs_residential.predict_proba(test.values)
# #         dict_prob['PUBLICvsRETAIL'] = model_public_vs_retail.predict_proba(test.values)
# #         dict_prob['PUBLICvsRESIDENTIAL'] = model_public_vs_residential.predict_proba(test.values)
# #         dict_prob['RETAILvsRESIDENTIAL'] = model_retail_vs_residential.predict_proba(test.values)
        
#         # => Con selección/creación de características:
        
#         aux = test.copy()
#         aux = aux.drop('GEOM_R2', axis=1)
#         aux['MAXBUILDINGFLOOR2'] = aux['MAXBUILDINGFLOOR']**2
#         aux['Q_G_3_0_32'] = aux['Q_G_3_0_3']**2
#         dict_prob['OFFICEvsRETAIL'] = model_office_vs_retail.predict_proba(aux.values)
        
#         aux = test.copy()
#         aux = aux.drop('GEOM_R3', axis=1)
#         aux['Y2'] = aux['Y']**2
#         aux['Q_NIR_8_0_92'] = aux['Q_NIR_8_0_9']**2
#         aux['Q_B_2_0_32'] = aux['Q_B_2_0_3']**2
#         dict_prob['OFFICEvsPUBLIC'] = model_office_vs_public.predict_proba(aux.values)
        
#         aux = test.copy()
#         aux = aux.drop('Q_B_2_0_2', axis=1)
#         aux = aux.drop('Q_B_2_1_0', axis=1)
#         aux = aux.drop('Q_NIR_8_0_8', axis=1)
#         aux['CADASTRALQUALITYID2'] = aux['CADASTRALQUALITYID']**2
#         aux['Q_NIR_8_0_52'] = aux['Q_NIR_8_0_5']**2
#         aux['Q_G_3_0_22'] = aux['Q_G_3_0_2']**2
#         dict_prob['OFFICEvsOTHER'] = model_office_vs_other.predict_proba(aux.values)
        
#         aux = test.copy()
#         aux = aux.drop('Q_NIR_8_0_8', axis=1)
#         aux['Q_B_2_0_62'] = aux['Q_B_2_0_6']**2
#         dict_prob['OFFICEvsINDUSTRIAL'] = model_office_vs_industrial.predict_proba(aux.values)
        
#         aux = test.copy()
#         aux = aux.drop('Q_G_3_0_2', axis=1)
#         aux['Y2'] = aux['Y']**2
#         aux['X2'] = aux['X']**2
#         aux['Q_NIR_8_0_32'] = aux['Q_NIR_8_0_3']**2
#         dict_prob['OFFICEvsAGRICULTURE'] = model_office_vs_agriculture.predict_proba(aux.values)
        
#         dict_prob['OFFICEvsRESIDENTIAL'] = model_office_vs_residential.predict_proba(test.values)
        
#         dict_prob['AGRICULTUREvsRETAIL'] = model_agriculture_vs_retail.predict_proba(test.values)
        
#         dict_prob['AGRICULTUREvsPUBLIC'] = model_agriculture_vs_public.predict_proba(test.values)
        
#         dict_prob['AGRICULTUREvsOTHER'] = model_agriculture_vs_other.predict_proba(test.values)
        
#         dict_prob['AGRICULTUREvsINDUSTRIAL'] = model_agriculture_vs_industrial.predict_proba(test.values)
        
#         dict_prob['AGRICULTUREvsRESIDENTIAL'] = model_agriculture_vs_residential.predict_proba(test.values)
        
#         dict_prob['INDUSTRIALvsRETAIL'] = model_industrial_vs_retail.predict_proba(test.values)
        
#         dict_prob['INDUSTRIALvsPUBLIC'] = model_industrial_vs_public.predict_proba(test.values)
        
#         dict_prob['INDUSTRIALvsOTHER'] = model_industrial_vs_other.predict_proba(test.values)
        
#         dict_prob['INDUSTRIALvsRESIDENTIAL'] = model_industrial_vs_residential.predict_proba(test.values)
        
#         dict_prob['OTHERvsRETAIL'] = model_other_vs_retail.predict_proba(test.values)
        
#         dict_prob['OTHERvsPUBLIC'] = model_other_vs_public.predict_proba(test.values)
        
#         dict_prob['OTHERvsRESIDENTIAL'] = model_other_vs_residential.predict_proba(test.values)
        
#         dict_prob['PUBLICvsRETAIL'] = model_public_vs_retail.predict_proba(test.values)
        
#         dict_prob['PUBLICvsRESIDENTIAL'] = model_public_vs_residential.predict_proba(test.values)
        
#         dict_prob['RETAILvsRESIDENTIAL'] = model_retail_vs_residential.predict_proba(test.values)

        
#         print("PREDICT: DONE")

#         dict_count = {
#             'OFFICE': 0,
#             'AGRICULTURE': 0,
#             'INDUSTRIAL': 0,
#             'OTHER': 0,
#             'PUBLIC': 0,
#             'RETAIL': 0,
#             'RESIDENTIAL': 0
#         }
#         #print(dict_prob['OTHERvsRESIDENTIAL'])

#         for i in range(0, test.shape[0]):
#             dict_count = {
#                 'OFFICE': 0,
#                 'AGRICULTURE': 0,
#                 'INDUSTRIAL': 0,
#                 'OTHER': 0,
#                 'PUBLIC': 0,
#                 'RETAIL': 0,
#                 'RESIDENTIAL': 0
#             }
#             #print("===============",i)
#             for key in dict_prob.keys():

#                 if (dict_prob[key][i][0]>dict_prob[key][i][1]):
#                     dict_count[dict_class[key][0]] +=1
#                 else:
#                     dict_count[dict_class[key][1]] +=1

#             predicted_class = max(dict_count.items(), key=operator.itemgetter(1))[0]
#             pred.append(predicted_class)
    
    
# #     # ==> VOTO PONDERADO (Suma de probabilidades) 
# #     pred = []
    
# #     # Predicción de las probabilidades:
# #     dict_prob['OFFICEvsRETAIL'] = model_office_vs_retail.predict_proba(test.values)
# #     dict_prob['OFFICEvsPUBLIC'] = model_office_vs_public.predict_proba(test.values)
# #     dict_prob['OFFICEvsOTHER'] = model_office_vs_other.predict_proba(test.values)
# #     dict_prob['OFFICEvsINDUSTRIAL'] = model_office_vs_industrial.predict_proba(test.values)
# #     dict_prob['OFFICEvsAGRICULTURE'] = model_office_vs_agriculture.predict_proba(test.values)
# #     dict_prob['OFFICEvsRESIDENTIAL'] = model_office_vs_residential.predict_proba(test.values)
# #     dict_prob['AGRICULTUREvsRETAIL'] = model_agriculture_vs_retail.predict_proba(test.values)
# #     dict_prob['AGRICULTUREvsPUBLIC'] = model_agriculture_vs_public.predict_proba(test.values)
# #     dict_prob['AGRICULTUREvsOTHER'] = model_agriculture_vs_other.predict_proba(test.values)
# #     dict_prob['AGRICULTUREvsINDUSTRIAL'] = model_agriculture_vs_industrial.predict_proba(test.values)
# #     dict_prob['AGRICULTUREvsRESIDENTIAL'] = model_agriculture_vs_residential.predict_proba(test.values)
# #     dict_prob['INDUSTRIALvsRETAIL'] = model_industrial_vs_retail.predict_proba(test.values)
# #     dict_prob['INDUSTRIALvsPUBLIC'] = model_industrial_vs_public.predict_proba(test.values)
# #     dict_prob['INDUSTRIALvsOTHER'] = model_industrial_vs_other.predict_proba(test.values)
# #     dict_prob['INDUSTRIALvsRESIDENTIAL'] = model_industrial_vs_residential.predict_proba(test.values)
# #     dict_prob['OTHERvsRETAIL'] = model_other_vs_retail.predict_proba(test.values)
# #     dict_prob['OTHERvsPUBLIC'] = model_other_vs_public.predict_proba(test.values)
# #     dict_prob['OTHERvsRESIDENTIAL'] = model_other_vs_residential.predict_proba(test.values)
# #     dict_prob['PUBLICvsRETAIL'] = model_public_vs_retail.predict_proba(test.values)
# #     dict_prob['PUBLICvsRESIDENTIAL'] = model_public_vs_residential.predict_proba(test.values)
# #     dict_prob['RETAILvsRESIDENTIAL'] = model_retail_vs_residential.predict_proba(test.values)
        
# #     print("PREDICT: DONE")
    
# #     dict_count = {
# #         'OFFICE': 0,
# #         'AGRICULTURE': 0,
# #         'INDUSTRIAL': 0,
# #         'OTHER': 0,
# #         'PUBLIC': 0,
# #         'RETAIL': 0,
# #         'RESIDENTIAL': 0
# #     }
# #     #print(dict_prob['OTHERvsRESIDENTIAL'])
    
# #     for i in range(0, test.shape[0]):
# #         dict_count = {
# #             'OFFICE': 0,
# #             'AGRICULTURE': 0,
# #             'INDUSTRIAL': 0,
# #             'OTHER': 0,
# #             'PUBLIC': 0,
# #             'RETAIL': 0,
# #             'RESIDENTIAL': 0
# #         }
# #         #print("===============",i)
# #         for key in dict_prob.keys():
# #             dict_count[dict_class[key][0]] += dict_prob[key][i][0]
# #             dict_count[dict_class[key][1]] += dict_prob[key][i][1]

# #         predicted_class = max(dict_count.items(), key=operator.itemgetter(1))[0]
# #         pred.append(predicted_class)

#     else:
#         # ==> (PROB VENCEDORA - PROB PERDEDORA) 
#         pred = []

#         # Predicción de las probabilidades:
        
#         data = data.drop('GEOM_R2', axis=1)
#         data['MAXBUILDINGFLOOR2'] = data['MAXBUILDINGFLOOR']**2
#         data['Q_G_3_0_32'] = data['Q_G_3_0_3']**2
#         dict_prob['OFFICEvsRETAIL'] = model_office_vs_retail.predict_proba(test.values)
        
#         data = data.drop('GEOM_R3', axis=1)
#         data['Y2'] = data['Y']**2
#         data['Q_NIR_8_0_92'] = data['Q_NIR_8_0_9']**2
#         data['Q_B_2_0_32'] = data['Q_B_2_0_3']**2
#         dict_prob['OFFICEvsPUBLIC'] = model_office_vs_public.predict_proba(test.values)
        
#         data = data.drop('Q_B_2_0_2', axis=1)
#         data = data.drop('Q_B_2_1_0', axis=1)
#         data = data.drop('Q_NIR_8_0_8', axis=1)
#         data['CADASTRALQUALITYID2'] = data['CADASTRALQUALITYID']**2
#         data['Q_NIR_8_0_52'] = data['Q_NIR_8_0_5']**2
#         data['Q_G_3_0_22'] = data['Q_G_3_0_2']**2
#         dict_prob['OFFICEvsOTHER'] = model_office_vs_other.predict_proba(test.values)
        
#         data = data.drop('Q_NIR_8_0_8', axis=1)
#         data['Q_B_2_0_62'] = data['Q_B_2_0_6']**2
#         dict_prob['OFFICEvsINDUSTRIAL'] = model_office_vs_industrial.predict_proba(test.values)
        
#         data = data.drop('Q_G_3_0_2', axis=1)
#         data['Y2'] = data['Y']**2
#         data['X2'] = data['X']**2
#         data['Q_NIR_8_0_32'] = data['Q_NIR_8_0_3']**2
#         dict_prob['OFFICEvsAGRICULTURE'] = model_office_vs_agriculture.predict_proba(test.values)
        
#         dict_prob['OFFICEvsRESIDENTIAL'] = model_office_vs_residential.predict_proba(test.values)
        
#         dict_prob['AGRICULTUREvsRETAIL'] = model_agriculture_vs_retail.predict_proba(test.values)
        
#         dict_prob['AGRICULTUREvsPUBLIC'] = model_agriculture_vs_public.predict_proba(test.values)
        
#         dict_prob['AGRICULTUREvsOTHER'] = model_agriculture_vs_other.predict_proba(test.values)
        
#         dict_prob['AGRICULTUREvsINDUSTRIAL'] = model_agriculture_vs_industrial.predict_proba(test.values)
        
#         dict_prob['AGRICULTUREvsRESIDENTIAL'] = model_agriculture_vs_residential.predict_proba(test.values)
        
#         dict_prob['INDUSTRIALvsRETAIL'] = model_industrial_vs_retail.predict_proba(test.values)
        
#         dict_prob['INDUSTRIALvsPUBLIC'] = model_industrial_vs_public.predict_proba(test.values)
        
#         dict_prob['INDUSTRIALvsOTHER'] = model_industrial_vs_other.predict_proba(test.values)
        
#         dict_prob['INDUSTRIALvsRESIDENTIAL'] = model_industrial_vs_residential.predict_proba(test.values)
        
#         dict_prob['OTHERvsRETAIL'] = model_other_vs_retail.predict_proba(test.values)
        
#         dict_prob['OTHERvsPUBLIC'] = model_other_vs_public.predict_proba(test.values)
        
#         dict_prob['OTHERvsRESIDENTIAL'] = model_other_vs_residential.predict_proba(test.values)
        
#         dict_prob['PUBLICvsRETAIL'] = model_public_vs_retail.predict_proba(test.values)
        
#         dict_prob['PUBLICvsRESIDENTIAL'] = model_public_vs_residential.predict_proba(test.values)
        
#         dict_prob['RETAILvsRESIDENTIAL'] = model_retail_vs_residential.predict_proba(test.values)

        
#         print("PREDICT: DONE")

#         dict_count = {
#             'OFFICE': 0,
#             'AGRICULTURE': 0,
#             'INDUSTRIAL': 0,
#             'OTHER': 0,
#             'PUBLIC': 0,
#             'RETAIL': 0,
#             'RESIDENTIAL': 0
#         }
#         #print(dict_prob['OTHERvsRESIDENTIAL'])

#         for i in range(0, test.shape[0]):
#             dict_count = {
#                 'OFFICE': 0,
#                 'AGRICULTURE': 0,
#                 'INDUSTRIAL': 0,
#                 'OTHER': 0,
#                 'PUBLIC': 0,
#                 'RETAIL': 0,
#                 'RESIDENTIAL': 0
#             }
#             #print("===============",i)
#             for key in dict_prob.keys():     
#                 if (dict_prob[key][i][0]>dict_prob[key][i][1]):
#                     dict_count[dict_class[key][0]] += dict_prob[key][i][0]-dict_prob[key][i][1]
#                 else:
#                     dict_count[dict_class[key][1]] += dict_prob[key][i][1]-dict_prob[key][i][0]
#             predicted_class = max(dict_count.items(), key=operator.itemgetter(1))[0]
#             pred.append(predicted_class)


#     return(pred)
        