import re

import pandas as pd


SPECIALTY_ALIASES = {
    "acute internal medicine": "Acute Internal Medicine",
    "acute internal medicine service": "Acute Internal Medicine",
    "breast surgery": "Breast Surgery",
    "breast surgery service": "Breast Surgery",
    "cardiology": "Cardiology",
    "cardiology service": "Cardiology",
    "chemical pathology": "Chemical Pathology",
    "chemical pathology service": "Chemical Pathology",
    "clinical haematology": "Clinical Haematology",
    "clinical haematology service": "Clinical Haematology",
    "clinical oncology": "Clinical Oncology",
    "clinical oncology service": "Clinical Oncology",
    "colorectal surgery": "Colorectal Surgery",
    "colorectal surgery service": "Colorectal Surgery",
    "dermatology": "Dermatology",
    "dermatology service": "Dermatology",
    "ear nose and throat": "ENT",
    "ear nose and throat service": "ENT",
    "ent": "ENT",
    "endocrinology": "Endocrinology",
    "endocrinology service": "Endocrinology",
    "gastroenterology": "Gastroenterology",
    "gastroenterology service": "Gastroenterology",
    "general internal medicine": "General Internal Medicine",
    "general internal medicine service": "General Internal Medicine",
    "general surgery": "General Surgery",
    "general surgery service": "General Surgery",
    "gynaecology": "Gynaecology",
    "gynaecology service": "Gynaecology",
    "medical oncology": "Medical Oncology",
    "medical oncology service": "Medical Oncology",
    "neurology": "Neurology",
    "neurology service": "Neurology",
    "ophthalmology": "Ophthalmology",
    "ophthalmology service": "Ophthalmology",
    "oral surgery": "Oral Surgery",
    "oral surgery service": "Oral Surgery",
    "paediatric clinical haematology": "Paediatric Clinical Haematology",
    "paediatric clinical haematology service": "Paediatric Clinical Haematology",
    "paediatric ear nose and throat": "Paediatric ENT",
    "paediatric ear nose and throat service": "Paediatric ENT",
    "paediatric ent": "Paediatric ENT",
    "paediatric urology": "Paediatric Urology",
    "paediatric urology service": "Paediatric Urology",
    "paediatrics": "Paediatrics",
    "paediatrics service": "Paediatrics",
    "physiotherapy": "Physiotherapy",
    "physiotherapy service": "Physiotherapy",
    "respiratory medicine": "Respiratory Medicine",
    "respiratory medicine service": "Respiratory Medicine",
    "rheumatology": "Rheumatology",
    "rheumatology service": "Rheumatology",
    "trauma and orthopaedics": "Trauma & Orthopaedics",
    "trauma and orthopaedics service": "Trauma & Orthopaedics",
    "trauma and orthopaedic": "Trauma & Orthopaedics",
    "trauma and orthopaedic service": "Trauma & Orthopaedics",
    "trauma and orthopedics": "Trauma & Orthopaedics",
    "trauma and orthopedics service": "Trauma & Orthopaedics",
    "trauma and orthapedics": "Trauma & Orthopaedics",
    "trauma and orthapedics service": "Trauma & Orthopaedics",
    "trauma and ortha edics": "Trauma & Orthopaedics",
    "trauma and ortha edics service": "Trauma & Orthopaedics",
    "trauma orthopaedics": "Trauma & Orthopaedics",
    "trauma orthopaedics service": "Trauma & Orthopaedics",
    "trauma orthopaedic": "Trauma & Orthopaedics",
    "trauma orthopaedic service": "Trauma & Orthopaedics",
    "trauma orthopedics": "Trauma & Orthopaedics",
    "trauma orthopedics service": "Trauma & Orthopaedics",
    "truma and orthopaedics": "Trauma & Orthopaedics",
    "truma and orthopaedics service": "Trauma & Orthopaedics",
    "truma and orthopaedic": "Trauma & Orthopaedics",
    "truma and orthopaedic service": "Trauma & Orthopaedics",
    "truma and orthapedics": "Trauma & Orthopaedics",
    "truma and orthapedics service": "Trauma & Orthopaedics",
    "truma orthopaedics": "Trauma & Orthopaedics",
    "truma orthopaedics service": "Trauma & Orthopaedics",
    "orthopaedic": "Trauma & Orthopaedics",
    "orthopaedic service": "Trauma & Orthopaedics",
    "orthopaedics": "Trauma & Orthopaedics",
    "orthopaedics service": "Trauma & Orthopaedics",
    "orthopedic": "Trauma & Orthopaedics",
    "orthopedic service": "Trauma & Orthopaedics",
    "orthopedics": "Trauma & Orthopaedics",
    "orthopedics service": "Trauma & Orthopaedics",
    "urology": "Urology",
    "urology oak unit pah": "Urology",
    "urology service": "Urology",
    "urlogy": "Urology",
    "urlogy service": "Urology",
    "vascular surgery": "Vascular Surgery",
    "vascular surgery service": "Vascular Surgery",
}


def normalise_specialty_key(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = text.strip().lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def standardise_specialty_value(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    if not text:
        return "Unknown"

    key = normalise_specialty_key(text)
    return SPECIALTY_ALIASES.get(key, text)


def standardise_specialty_series(series: pd.Series) -> pd.Series:
    return series.apply(standardise_specialty_value)
