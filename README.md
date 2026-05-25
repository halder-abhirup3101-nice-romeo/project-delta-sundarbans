# project-delta-sundarbans
Block-level post-cyclone displacement spike prediction  for the Indian Sundarbans using XGBoost and SHAP.  Trained on Cyclone Aila 2009, validated on Amphan 2020.
# Project Delta
## Post-Cyclone Displacement Spike Prediction in the Indian Sundarbans

**Author:** Abhirup Halder  
**Affiliation:** Independent Undergraduate Researcher, Kolkata, India  
**Contact:** halderabhirup3101@email.com  
**Date:** May 2026

---

## What This Project Does

This project predicts which blocks in the Indian Sundarbans 
will experience the highest displacement after a major cyclone 
— before the storm arrives.

It uses an XGBoost machine learning model trained on 
Cyclone Aila (2009) and validated against Cyclone Amphan 
(2020), achieving 84.2% accuracy and a macro AUC-ROC of 0.906.

The central finding: structural vulnerability — poverty, 
elevation, flood inundation — predicts displacement intensity 
more reliably than storm magnitude alone.

---

## The Key Result

| Model | Features Used | Accuracy | AUC-ROC |
|-------|--------------|----------|---------|
| Macro model | Meteorological | 26.3% | 0.794 |
| Structural model | Block-level vulnerability | 84.2% | 0.906 |

The structural model — using zero storm data — outperforms 
the meteorological model by 57.9 percentage points.

---

## How To Run

Run the files in this exact order:

```bash
python code/01_preprocess.py    # Load and merge raw data
python code/02_features.py      # Feature engineering
python code/03_train.py         # Train model, save model.pkl
python code/04_shap.py          # SHAP analysis and graphs
python code/05_map.py           # Generate risk map
```

---

## Data Sources

All data used in this project is open-access:

| Dataset | Source | What it provides |
|---------|--------|-----------------|
| IBTrACS v04r00 | NOAA | Cyclone track, wind, pressure |
| ERA5 Reanalysis | ECMWF | Rainfall data |
| Census of India 2011 | Government of India | Population, housing, literacy |
| SECC 2011 | Government of India | Poverty proxies |
| Sentinel-1 SAR | ESA Copernicus | Flood inundation maps |
| IDMC Database | Internal Displacement Monitoring Centre | Displacement records |

See `data/README_data.md` for download instructions.

---

## Project Structure

Run the pipeline in order:
code/          — all 5 Python scripts in order
outputs/       — SHAP charts and risk map
data/          — data download instructions
requirements.txt — Python libraries needed
