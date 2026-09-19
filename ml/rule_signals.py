import pandas as pd
import numpy as np

def detect_rule_violations (df ):
    df =df .copy ()

    df ["pattern_score"]=0
    df ["pattern_bulk_msg"]=""
    df ["pattern_dup_msg"]=""
    df ["pattern_limit_msg"]=""
    df ["pattern_lag_msg"]=""

    bulk_counts =df .groupby (["mp_name_clean","district","sanction_date"]).size ().reset_index (name ='daily_sanctions')
    df =df .merge (bulk_counts ,on =["mp_name_clean","district","sanction_date"],how ="left")

    mask_bulk =df ["daily_sanctions"]>=5
    df .loc [mask_bulk ,"pattern_score"]+=np .minimum (10 ,(df .loc [mask_bulk ,"daily_sanctions"]-4 )*2 )

    def format_bulk_msg (row ):
        if not pd .isna (row ["sanction_date"]):
            dt_str =row ["sanction_date"].strftime ("%d-%b-%Y")
        else :
            dt_str ="Unknown"
        return f"{int (row ['daily_sanctions'])} works same MP + district par {dt_str } ko sanction hue"

    df .loc [mask_bulk ,"pattern_bulk_msg"]=df [mask_bulk ].apply (format_bulk_msg ,axis =1 )

    df ["desc_clean"]=df ["work_description"].fillna ("").astype (str ).str .lower ().str .strip ()
    valid_desc =df ["desc_clean"].str .len ()>10
    dup_mask =df .duplicated (subset =["district","desc_clean"],keep =False )&valid_desc
    df .loc [dup_mask ,"pattern_score"]+=7
    df .loc [dup_mask ,"pattern_dup_msg"]="Exact matching description found in the same district"
    df .drop (columns =["desc_clean"],inplace =True )

    mask_limit =(
    (df ["amount_lakhs"]>=4.5 )&(df ["amount_lakhs"]<=4.99 )|
    (df ["amount_lakhs"]>=9.5 )&(df ["amount_lakhs"]<=9.99 )
    )
    df .loc [mask_limit ,"pattern_score"]+=5
    df .loc [mask_limit ,"pattern_limit_msg"]="Threshold-hugging amount (Weak signal: clustering just below typical limits)"

    if "rec_to_sanction_days"in df .columns :
        p95 =df ["rec_to_sanction_days"].quantile (0.95 )
        mask_lag =(df ["rec_to_sanction_days"]<=1 )|(df ["rec_to_sanction_days"]>p95 )
        df .loc [mask_lag ,"pattern_score"]+=3
        df .loc [mask_lag ,"pattern_lag_msg"]="Anomalous delay between recommendation and sanction"

    df ["pattern_score"]=df ["pattern_score"].clip (0 ,25 )
    return df
