import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
import re
import random

CSV_FILENAME ="Works Sanctioned.csv"

COL_MAP ={
"Sr. No.":"sr_no",
"Work category":"work_category",
"Work":"work_code",
"State":"state",
"IDA":"ida",
"Hon'ble Members of Parliament":"mp_name",
"Constituency":"constituency",
"Work description":"work_description",
"Recommended date":"recommended_date",
"Sanction Date":"sanction_date",
"Sanction Amount ( \u20b9 )":"sanction_amount",
"Work Status":"work_status",
}

INCOMPLETE_STATUSES ={
"Vendor Identification","Physical Inspection","Sanction",
"Time Estimation","Work partially Completed","Work in Progress",
}
COMPLETE_STATUSES ={"Work Completed"}

DEMO_STATES =[
"Uttar Pradesh","Maharashtra","Bihar","Rajasthan","Madhya Pradesh",
"West Bengal","Tamil Nadu","Karnataka","Gujarat","Andhra Pradesh",
"Odisha","Telangana","Jharkhand","Punjab","Haryana",
]

DEMO_CATEGORIES =[
"Normal/Others","Drinking Water","Education","Health",
"Roads and Bridges","Sanitation","Sports","Environment",
]

DEMO_DESCRIPTIONS =[
"Construction of CC Road from {a} to {b}",
"Construction of Pucca Road in Village {a}",
"Installation of Hand Pump in Village {a}",
"Construction of Community Hall at {a}",
"Construction of Class Room at {a} School",
"Renovation of Primary School Building at {a}",
"Construction of Toilet Block at {a}",
"Construction of Boundary Wall at {a} School",
"Supply of Furniture to Government School {a}",
"Construction of Water Tank at Village {a}",
"Repair of Road from {a} to {b}",
"Construction of Drain in {a} Ward",
"Installation of Solar Street Lights at {a}",
"Construction of Anganwadi Centre at {a}",
"Construction of Bridge over Nala at {a}",
"Construction of Primary Health Sub Centre at {a}",
"Plantation of Trees along {a} Road",
"Construction of Open Gym at {a} Park",
"Construction of Sports Ground at {a}",
"Repair of Drinking Water Pipeline at {a}",
]

DEMO_VILLAGES =[
"Rampur","Lakhimpur","Sitapur","Hardoi","Unnao","Fatehpur",
"Pratapgarh","Sultanpur","Ambedkar Nagar","Gorakhpur","Deoria",
"Kushinagar","Mahoba","Chitrakoot","Banda","Hamirpur","Jalaun",
"Kolkata","Howrah","Bardhaman","Midnapore","Murshidabad",
"Pune","Nashik","Aurangabad","Solapur","Nagpur",
"Jaipur","Jodhpur","Bikaner","Kota","Ajmer","Udaipur",
"Patna","Gaya","Bhagalpur","Muzaffarpur","Darbhanga",
"Bhopal","Indore","Jabalpur","Gwalior","Sagar",
]

DEMO_MPs =[
"Shri Ram Prasad Verma (2019-24)","Smt. Sunita Devi (2019-24)",
"Shri Ajay Kumar Singh (2019-24)","Shri Rajesh Gupta (2022-28)",
"Smt. Priya Sharma (2019-24)","Shri Mukesh Yadav (2019-24)",
"Shri Anil Kumar Tiwari (2019-24)","Smt. Meena Kumari (2022-28)",
"Shri Dinesh Prasad (2019-24)","Shri Santosh Mishra (2019-24)",
]

DEMO_STATUSES =[
"Work Completed","Work Completed","Work Completed",
"Work in Progress","Work in Progress",
"Vendor Identification","Vendor Identification",
"Physical Inspection","Sanction","Time Estimation",
"Work partially Completed",
]

DEMO_CONSTITUENCIES =[
"Lucknow","Kanpur","Varanasi","Allahabad","Agra",
"Meerut","Bareilly","Moradabad","Gorakhpur","Jhansi",
"Patna","Gaya","Muzaffarpur","Darbhanga","Bhagalpur",
"Jaipur","Jodhpur","Kota","Bikaner","Udaipur",
]

def _generate_synthetic_data (n =800 ,seed =42 ):
    rng =random .Random (seed )
    np_rng =np .random .default_rng (seed )
    rows =[]
    base_date =pd .Timestamp ("2023-01-01")
    for i in range (1 ,n +1 ):
        state =rng .choice (DEMO_STATES )
        category =rng .choice (DEMO_CATEGORIES )
        village_a =rng .choice (DEMO_VILLAGES )
        village_b =rng .choice (DEMO_VILLAGES )
        desc_tmpl =rng .choice (DEMO_DESCRIPTIONS )
        desc =desc_tmpl .replace ("{a}",village_a ).replace ("{b}",village_b )
        mp =rng .choice (DEMO_MPs )
        constituency =rng .choice (DEMO_CONSTITUENCIES )
        fy_year =rng .choice (["2022-2023","2023-2024","2024-2025"])
        work_code =f"WS/MP{rng .randint (100 ,999 )}/{fy_year }/{i }"
        ida =f"{village_a .upper ()} (DISTRICT MAGISTRATE {village_a .upper ()}_IDA)"
        rec_days_offset =rng .randint (0 ,600 )
        rec_date =base_date +pd .Timedelta (days =rec_days_offset )
        sanction_gap =rng .randint (15 ,400 )
        sanction_date =rec_date +pd .Timedelta (days =sanction_gap )
        base_amount =np_rng .lognormal (mean =13.5 ,sigma =0.8 )
        if rng .random ()<0.05 :
            base_amount *=rng .uniform (3.5 ,6.0 )
        amount =round (base_amount /10000 )*10000
        amount =max (50000 ,min (5000000 ,amount ))
        status =rng .choice (DEMO_STATUSES )
        rows .append ({
        "Sr. No.":str (i ),
        "Work category":category ,
        "Work":work_code ,
        "State":state ,
        "IDA":ida ,
        "Hon'ble Members of Parliament":mp ,
        "Constituency":constituency ,
        "Work description":desc ,
        "Recommended date":rec_date .strftime ("%d-%b-%Y"),
        "Sanction Date":sanction_date .strftime ("%d-%b-%Y"),
        "Sanction Amount ( \u20b9 )":str (int (amount )),
        "Work Status":status ,
        })
    return pd .DataFrame (rows )

def _find_csv (app_dir ):
    candidates =[app_dir /CSV_FILENAME ,app_dir .parent /CSV_FILENAME ,Path (CSV_FILENAME )]
    for p in candidates :
        if p .exists ():
            return p
    return None

def _extract_district (ida ):
    if pd .isna (ida ):
        return "Unknown"
    match =re .match (r"^([^(]+)",str (ida ).strip ())
    return match .group (1 ).strip ().title ()if match else str (ida ).strip ().title ()

def _extract_financial_year (work_code ):
    match =re .search (r"(\d{4}-\d{4})",str (work_code ))
    return match .group (1 )if match else "Unknown"

def _extract_work_sub_category (work_code ):
    if pd .isna (work_code ):
        return "Unknown"
    parts =str (work_code ).split ("-",maxsplit =4 )
    return parts [-1 ].strip ()if len (parts )>=2 else str (work_code ).strip ()

def _parse_amount (val ):
    if pd .isna (val ):
        return np .nan
    s =str (val ).replace (",","").replace ("\u20b9","").strip ()
    try :
        return float (s )
    except ValueError :
        return np .nan

def _parse_date (s ,fmt ="%d-%b-%Y"):
    return pd .to_datetime (s ,format =fmt ,errors ="coerce")

def _process_df (df ):

    raw_amount =df ["sanction_amount"].copy ()
    df ["sanction_amount"]=raw_amount .apply (_parse_amount )
    corrupt_amount_mask =raw_amount .notna ()&df ["sanction_amount"].isna ()

    df ["recommended_date"]=df ["recommended_date"].apply (_parse_date )
    df ["sanction_date"]=df ["sanction_date"].apply (_parse_date )
    df ["district"]=df ["ida"].apply (_extract_district )
    df ["sub_category"]=df ["work_code"].apply (_extract_work_sub_category )
    df ["financial_year"]=df ["work_code"].apply (_extract_financial_year )
    df ["mp_name_clean"]=df ["mp_name"].str .replace (r"\s*\(\d{4}-\d{2,4}\)","",regex =True ).str .strip ()
    df ["rec_to_sanction_days"]=(df ["sanction_date"]-df ["recommended_date"]).dt .days
    today =pd .Timestamp .today ().normalize ()
    df ["days_since_sanction"]=(today -df ["sanction_date"]).dt .days
    df ["amount_lakhs"]=(df ["sanction_amount"]/1e5 ).round (2 )
    df ["is_completed"]=df ["work_status"].isin (COMPLETE_STATUSES )
    df ["is_incomplete"]=~df ["is_completed"]

    bad_desc_mask =df ["work_description"].fillna ("").astype (str ).str .len ()<5

    df ["data_quality_flag"]=False
    df .loc [corrupt_amount_mask |bad_desc_mask ,"data_quality_flag"]=True

    df =df .reset_index (drop =True )
    df ["row_id"]=df .index +1
    return df

@st .cache_data (show_spinner ="Loading MPLADS data\u2026",ttl =3600 )
def load_data ():
    app_dir =Path (__file__ ).resolve ().parent .parent
    csv_path =_find_csv (app_dir )

    if csv_path is not None :
        raw =pd .read_csv (csv_path ,encoding ="utf-8",dtype =str ,low_memory =False )
        raw =raw [raw .iloc [:,0 ].str .strip ()!="Grand Total"].copy ()
        df =raw .rename (columns =COL_MAP )
        known_cols =list (COL_MAP .values ())
        df =df [[c for c in known_cols if c in df .columns ]].copy ()
        data_source ="real"
    else :
        pass
        raw =_generate_synthetic_data (n =800 )
        df =raw .rename (columns =COL_MAP )
        data_source ="synthetic"

    df =_process_df (df )
    df ["data_source"]=data_source
    return df

def get_data_summary (df ):
    total_amount =df ["sanction_amount"].sum ()if "sanction_amount"in df .columns else 0
    old_pending =df [~df ["is_completed"]&(df ["days_since_sanction"]>365 )]if "is_completed"in df .columns and "days_since_sanction"in df .columns else []
    source =df ["data_source"].iloc [0 ]if "data_source"in df .columns and len (df )>0 else "unknown"
    fy =sorted (df ["financial_year"].dropna ().unique ().tolist ())if "financial_year"in df .columns else []
    return {
    "total_works":len (df ),
    "total_amount":total_amount ,
    "total_amount_cr":round (total_amount /1e7 ,2 ),
    "completed":int (df ["is_completed"].sum ())if "is_completed"in df .columns else 0 ,
    "incomplete":int (df ["is_incomplete"].sum ())if "is_incomplete"in df .columns else 0 ,
    "old_pending":len (old_pending ),
    "unique_states":df ["state"].nunique ()if "state"in df .columns else 0 ,
    "unique_mps":df ["mp_name_clean"].nunique ()if "mp_name_clean"in df .columns else 0 ,
    "unique_districts":df ["district"].nunique ()if "district"in df .columns else 0 ,
    "financial_years":fy ,
    "data_source":source ,
    }
