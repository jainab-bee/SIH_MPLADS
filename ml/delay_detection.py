import numpy as np
import pandas as pd
import streamlit as st

OLD_PENDING_DAYS =365
VERY_OLD_DAYS =730
CRITICAL_OLD_DAYS =1095
REC_TO_SANCTION_NORMAL =75
REC_TO_SANCTION_HIGH =180
REC_TO_SANCTION_VERY_HIGH =365

COMPLETE_STATUSES ={"Work Completed"}

def _old_pending_score (row ):
    if row ["is_completed"]:
        return 0.0 ,"Work completed"
    days =row .get ("days_since_sanction",np .nan )
    if pd .isna (days ):
        return 0.0 ,"Sanction date not available"
    days =float (days )
    if days >=CRITICAL_OLD_DAYS :
        score =1.0
        reason =(
        f"Work sanctioned {int (days )} days ago ({days /365 :.1f} years) "
        f"with status '{row ['work_status']}' \u2014 critically old pending"
        )
    elif days >=VERY_OLD_DAYS :
        score =0.75 +0.25 *(days -VERY_OLD_DAYS )/(CRITICAL_OLD_DAYS -VERY_OLD_DAYS )
        reason =(
        f"Work sanctioned {int (days )} days ago ({days /365 :.1f} years) "
        f"with status '{row ['work_status']}' \u2014 very old pending"
        )
    elif days >=OLD_PENDING_DAYS :
        score =0.4 +0.35 *(days -OLD_PENDING_DAYS )/(VERY_OLD_DAYS -OLD_PENDING_DAYS )
        reason =(
        f"Work sanctioned {int (days )} days ago with status '{row ['work_status']}' "
        f"\u2014 beyond 1-year MPLADS target"
        )
    else :
        score =max (0.0 ,(days -90 )/(OLD_PENDING_DAYS -90 ))*0.3
        reason =f"Work in progress ({int (days )} days since sanction)"
    return round (min (1.0 ,score ),4 ),reason

def _sanction_delay_score (row ):
    days =row .get ("rec_to_sanction_days",np .nan )
    if pd .isna (days )or days <0 :
        return 0.0 ,"Processing gap not calculable"
    days =float (days )
    if days >=REC_TO_SANCTION_VERY_HIGH :
        score =min (1.0 ,0.7 +(days -REC_TO_SANCTION_VERY_HIGH )/730 )
        reason =f"Recommendation to sanction took {int (days )} days (expected \u226475 days)"
    elif days >=REC_TO_SANCTION_HIGH :
        score =0.4
        reason =f"Recommendation to sanction took {int (days )} days"
    elif days >=REC_TO_SANCTION_NORMAL :
        score =0.15
        reason =f"Recommendation to sanction took {int (days )} days (slightly above norm)"
    else :
        score =0.0
        reason =f"Sanction processed in {int (days )} days (within guideline)"
    return round (score ,4 ),reason

@st .cache_data (show_spinner ="Running delay/pending detection\u2026",ttl =3600 )
def detect_delays (df ):
    result_df =df .copy ()
    old_scores ,old_reasons ,san_scores ,san_reasons =[],[],[],[]
    for _ ,row in result_df .iterrows ():
        os_ ,or_ =_old_pending_score (row )
        ss_ ,sr_ =_sanction_delay_score (row )
        old_scores .append (os_ )
        old_reasons .append (or_ )
        san_scores .append (ss_ )
        san_reasons .append (sr_ )
    result_df ["old_pending_score"]=old_scores
    result_df ["sanction_delay_score"]=san_scores
    result_df ["delay_score"]=(
    pd .Series (old_scores )*0.70 +pd .Series (san_scores )*0.30
    ).round (4 ).values
    result_df ["delay_flag"]=result_df ["delay_score"]>0.4
    delay_reasons =[]
    for i ,row in result_df .iterrows ():
        parts =[]
        if old_reasons [row .name ]:
            parts .append (old_reasons [row .name ])
        if san_reasons [row .name ]and san_scores [row .name ]>0.1 :
            parts .append (san_reasons [row .name ])
        delay_reasons .append (" | ".join (parts )if parts else "No delay signals")
    result_df ["delay_reason"]=delay_reasons
    result_df ["delay_score_pct"]=(result_df ["delay_score"]*100 ).round (1 )
    return result_df
