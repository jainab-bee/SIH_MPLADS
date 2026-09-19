import pandas as pd
import numpy as np
import streamlit as st

WEIGHT_COST =30
WEIGHT_DUPLICATE =25
WEIGHT_DELAY =20
WEIGHT_RULES =25

RISK_HIGH_THRESHOLD =70
RISK_MEDIUM_THRESHOLD =40

def _risk_level (score ):
    if score >=RISK_HIGH_THRESHOLD :
        return "HIGH"
    if score >=RISK_MEDIUM_THRESHOLD :
        return "MEDIUM"
    return "LOW"

def _risk_emoji (level ):
    return {"HIGH":"\U0001f534","MEDIUM":"\U0001f7e1","LOW":"\U0001f7e2"}.get (level ,"\u26aa")

def _to_component (raw_score_01 ,weight ):
    return round (float (np .clip (raw_score_01 ,0 ,1 ))*weight ,2 )

def _build_explanation (row ):
    cost_pts =row .get ("cost_component",0 )
    dup_pts =row .get ("dup_component",0 )
    delay_pts =row .get ("delay_component",0 )
    rule_pts =row .get ("rule_component",0 )
    total =row .get ("risk_score",0 )
    level =row .get ("risk_level","LOW")
    reasons =[]

    if cost_pts >=10 :
        reasons .append ({
        "signal":"\u26a0 Cost Anomaly",
        "points":f"+{cost_pts :.0f} pts",
        "detail":row .get ("cost_reason",""),
        "severity":"HIGH"if cost_pts >=20 else "MEDIUM",
        })
    if dup_pts >=10 :
        reasons .append ({
        "signal":"\u26a0 Potential Similar / Duplicate Work",
        "points":f"+{dup_pts :.0f} pts",
        "detail":row .get ("dup_reason",""),
        "severity":"HIGH"if dup_pts >=15 else "MEDIUM",
        })
    if delay_pts >=10 :
        reasons .append ({
        "signal":"\u26a0 Old Pending / Potential Delay",
        "points":f"+{delay_pts :.0f} pts",
        "detail":row .get ("delay_reason",""),
        "severity":"HIGH"if delay_pts >=15 else "MEDIUM",
        })
    if rule_pts >0 :
        rule_details =[]
        if row .get ("pattern_bulk_msg"):rule_details .append (row ["pattern_bulk_msg"])
        if row .get ("pattern_limit_msg"):rule_details .append (row ["pattern_limit_msg"])
        if row .get ("pattern_dup_msg"):rule_details .append (row ["pattern_dup_msg"])
        if row .get ("pattern_lag_msg"):rule_details .append (row ["pattern_lag_msg"])

        reasons .append ({
        "signal":"\U0001f6a8 Pattern Violations",
        "points":f"+{rule_pts :.0f} pts",
        "detail":" | ".join ([d for d in rule_details if d ]),
        "severity":"HIGH"if rule_pts >=15 else "MEDIUM",
        })

    if not reasons :
        reasons .append ({
        "signal":"\u2705 No significant anomaly detected",
        "points":"",
        "detail":"This work is within expected parameters.",
        "severity":"LOW",
        })
    return {
    "risk_score":round (total ,1 ),
    "risk_level":level ,
    "risk_emoji":_risk_emoji (level ),
    "cost_pts":cost_pts ,
    "dup_pts":dup_pts ,
    "delay_pts":delay_pts ,
    "rule_pts":rule_pts ,
    "reasons":reasons ,
    "recommended_action":_recommended_action (level ),
    }

def _recommended_action (level ):
    actions ={
    "HIGH":(
    "Recommended for priority review. Verify sanction records, "
    "physical progress, and implementing agency credentials. Consider field inspection."
    ),
    "MEDIUM":(
    "Flag for routine audit review. Cross-check with district authority records "
    "and similar works in the same area."
    ),
    "LOW":"No immediate action required. Include in standard periodic review.",
    }
    return actions .get (level ,"Review as per standard procedure.")

def compute_risk_scores (df ,weight_cost =30 ,weight_duplicate =25 ,weight_delay =20 ,weight_rules =25 ):
    result_df =df .copy ()
    result_df ["cost_component"]=result_df ["cost_anomaly_score"].apply (lambda x :_to_component (x ,weight_cost ))
    result_df ["dup_component"]=result_df ["dup_score"].apply (lambda x :_to_component (x ,weight_duplicate ))
    result_df ["delay_component"]=result_df ["delay_score"].apply (lambda x :_to_component (x ,weight_delay ))

    result_df ["rule_component"]=(result_df ["pattern_score"]/25.0 ).apply (lambda x :_to_component (x ,weight_rules ))

    result_df ["risk_score"]=(
    result_df ["cost_component"]+result_df ["dup_component"]+result_df ["delay_component"]+result_df ["rule_component"]
    ).clip (0 ,100 ).round (1 )
    result_df ["risk_level"]=result_df ["risk_score"].apply (_risk_level )
    result_df ["risk_emoji"]=result_df ["risk_level"].apply (_risk_emoji )

    def _main_reason (row ):
        parts =[]
        if row .get ("cost_component",0 )>=10 :parts .append ("High cost anomaly")
        if row .get ("dup_component",0 )>=10 :parts .append ("Similar work detected")
        if row .get ("delay_component",0 )>=10 :parts .append ("Old pending work")
        if row .get ("rule_component",0 )>0 :parts .append ("Pattern violation")
        return " \u00b7 ".join (parts )if parts else "Within normal range"

    result_df ["main_reason"]=result_df .apply (_main_reason ,axis =1 )
    return result_df

def get_explanation (row ):
    return _build_explanation (row )
