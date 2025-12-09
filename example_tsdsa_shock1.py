#!/usr/bin/env python
# coding: utf-8

# In[7]:


import tsdsa_shock1 as ts1
import importlib, os

importlib.reload(ts1)

data_path  = "AC_H3_EPM_614092.csv"
shock_time = "1998-08-26 06:20:55"
speed_km_s = 668.0

channels = ["P2", "P3", "P4", "P5"]
labels = {
    "P2": "65–115 keV",
    "P3": "115–193 keV",
    "P4": "195–315 keV",
    "P5": "315–580 keV",
}
bands = {
    "P2": (67, 115),
    "P3": (115, 193),
    "P4": (193, 315),
    "P5": (315, 580),
}

df = ts1.load_epam_csv(data_path)
df = ts1.add_distance_from_shock(df, shock_time, speed_km_s)

results = ts1.fit_channels(df, channels, labels)

# ---------- FIX HERE ----------
outdir = "figures"
os.makedirs(outdir, exist_ok=True)
# ------------------------------

ts1.plot_profiles(results, save=f"{outdir}/tsdsa1_profiles_P2_P5.png")
ts1.plot_energy_trends(results, bands, save_prefix=f"{outdir}/tsdsa1_trends_P2_P5")


# In[ ]:




