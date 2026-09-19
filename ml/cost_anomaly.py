import json
import numpy as np
import pandas as pd
from sklearn .ensemble import IsolationForest
from sklearn .preprocessing import RobustScaler
import streamlit as st
import warnings
warnings .filterwarnings ("ignore")

MIN_PEER_SIZE =8
CONTAMINATION =0.05

def _peer_group_key (row ):
    return f"{row ['work_category']}|{row ['state']}"

try :
    import shap
    HAS_SHAP =True
except ImportError :
    HAS_SHAP =False

def _score_group (group_df ,peer_median ,peer_p90 ):
    valid_mask =group_df ["sanction_amount"].notna ()
    if not valid_mask .any ():
        return pd .Series (0 ,index =group_df .index ),None

    amounts =group_df .loc [valid_mask ,"sanction_amount"].values

    features =pd .DataFrame ({
    "amount_lakhs":amounts /1e5 ,
    "ratio_to_median":amounts /(peer_median if peer_median >0 else 1 ),
    "ratio_to_p90":amounts /(peer_p90 if peer_p90 >0 else 1 )
    },index =group_df .loc [valid_mask ].index )

    scaler =RobustScaler ()
    X =scaler .fit_transform (features )
    iso =IsolationForest (n_estimators =100 ,contamination =CONTAMINATION ,random_state =42 ,n_jobs =-1 )
    iso .fit (X )

    raw_scores =iso .decision_function (X )
    min_s ,max_s =raw_scores .min (),raw_scores .max ()

    if max_s ==min_s :
        normalized_scores =pd .Series (np .zeros (len (raw_scores )),index =features .index )
        shap_values_dict ={idx :{}for idx in features .index }
    else :
        normalized_scores =pd .Series (1 -(raw_scores -min_s )/(max_s -min_s ),index =features .index )

        shap_values_dict ={}
        if HAS_SHAP :
            try :
                explainer =shap .TreeExplainer (iso )
                shap_vals =explainer .shap_values (X )
                feature_names =features .columns
                for i ,idx in enumerate (features .index ):
                    contributions =list (zip (feature_names ,shap_vals [i ]))
                    contributions .sort (key =lambda x :abs (x [1 ]),reverse =True )
                    shap_values_dict [idx ]={k :float (v )for k ,v in contributions }
            except Exception :
                for idx in features .index :
                    shap_values_dict [idx ]={"amount_lakhs":float (normalized_scores .loc [idx ])}
        else :
            for idx in features .index :
                shap_values_dict [idx ]={"amount_lakhs":float (normalized_scores .loc [idx ])}

    return normalized_scores ,shap_values_dict

@st .cache_data (show_spinner ="Running cost anomaly detection\u2026",ttl =3600 )
def detect_cost_anomalies (df ):
    result_df =df .copy ()
    result_df ["peer_group_key"]=result_df .apply (_peer_group_key ,axis =1 )
    for col in ["cost_anomaly_score","peer_median","peer_p75","peer_p90"]:
        result_df [col ]=0.0 if col =="cost_anomaly_score"else np .nan
    result_df ["peer_size"]=0
    result_df ["peer_group_label"]=""
    result_df ["cost_anomaly_flag"]=False
    result_df ["cost_reason"]="Within normal peer range"

    result_df ["shap_values"]=None

    for key ,group_idx in result_df .groupby ("peer_group_key").groups .items ():
        group =result_df .loc [group_idx ].copy ()
        valid =group ["sanction_amount"].notna ()&(group ["sanction_amount"]>0 )
        valid_group =group [valid ]
        peer_size =len (valid_group )
        peer_median =valid_group ["sanction_amount"].median ()
        peer_p75 =valid_group ["sanction_amount"].quantile (0.75 )
        peer_p90 =valid_group ["sanction_amount"].quantile (0.90 )
        result_df .loc [group_idx ,"peer_median"]=peer_median
        result_df .loc [group_idx ,"peer_p75"]=peer_p75
        result_df .loc [group_idx ,"peer_p90"]=peer_p90
        result_df .loc [group_idx ,"peer_size"]=peer_size
        cat ,state =key .split ("|",1 )
        label =f"{cat } / {state }"
        result_df .loc [group_idx ,"peer_group_label"]=label

        if peer_size <MIN_PEER_SIZE :
            for idx in group_idx :
                amt =result_df .loc [idx ,"sanction_amount"]
                if pd .notna (amt )and peer_median >0 :
                    ratio =amt /peer_median
                    score =min (1.0 ,max (0.0 ,(ratio -1 )/4 ))
                    result_df .loc [idx ,"cost_anomaly_score"]=round (score ,4 )
                    result_df .at [idx ,"shap_values"]=json .dumps ({"amount_lakhs":score })
                    if ratio >3 :
                        result_df .loc [idx ,"cost_anomaly_flag"]=True
                        result_df .loc [idx ,"cost_reason"]=(
                        f"Amount \u20b9{amt /1e5 :.1f}L is {ratio :.1f}\u00d7 peer median "
                        f"\u20b9{peer_median /1e5 :.1f}L (small group \u2014 {peer_size } works)"
                        )
            continue

        if_scores ,shap_dict =_score_group (valid_group ,peer_median ,peer_p90 )

        result_df .loc [valid_group .index ,"cost_anomaly_score"]=if_scores .round (4 )
        for idx ,s_vals in shap_dict .items ():
            result_df .at [idx ,"shap_values"]=json .dumps (s_vals )

        threshold =if_scores .quantile (1 -CONTAMINATION )
        for idx in valid_group .index :
            amt =result_df .loc [idx ,"sanction_amount"]
            score =result_df .loc [idx ,"cost_anomaly_score"]
            ratio =amt /peer_median if peer_median >0 else 1.0
            if score >=threshold :
                result_df .loc [idx ,"cost_anomaly_flag"]=True
                if ratio >=2 :
                    result_df .loc [idx ,"cost_reason"]=(
                    f"Amount \u20b9{amt /1e5 :.1f}L is {ratio :.1f}\u00d7 peer median "
                    f"\u20b9{peer_median /1e5 :.1f}L for {label } ({peer_size } works)"
                    )
                else :
                    result_df .loc [idx ,"cost_reason"]=(
                    f"Isolation Forest flagged unusual cost pattern in {label } ({peer_size } works)"
                    )

    result_df ["cost_anomaly_score_pct"]=(result_df ["cost_anomaly_score"]*100 ).round (1 )
    return result_df
