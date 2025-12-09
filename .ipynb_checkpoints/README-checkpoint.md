# AES-509 Final Project — TSDSA Shock Analysis Module

### Author: Rubaiya Shikha  
### Module: `tsdsa_shock1.py`  
### Example Script: `example_tsdsa_shock1.py`  
### Course: AES-509 – Scientific Programming  

---

## 📌 Project Summary

This project implements a full scientific analysis pipeline for energetic particle fluxes measured around the **1998-08-26 interplanetary shock**.

The project is organized into:

1. **A Python module** (`tsdsa_shock1.py`)  
   - Contains all processing, fitting, and modeling functions  
2. **An example script** (`example_tsdsa_shock1.py`)  
   - Demonstrates how to import and use the module  
   - Generates and saves all plots

The module is imported and used inside the example script.

---

## 📁 Repository Structure

```
AES-509_Project_Final/
│
├── tsdsa_shock1.py              # Main analysis module
├── example_tsdsa_shock1.py      # Demonstration script
├── figures/                     # Output plots
└── README.md                    # Documentation
```

---

## 🧩 Module Description — `tsdsa_shock1.py`

This module includes:

### **1. Data Loading & Cleaning**
- Reads ACE/EPAM CSV files
- Interpolates missing flux values
- Removes bad data

### **2. Coordinate Transformation**
Converts observation times into distance-from-shock (in AU) using:

```
x = time_elapsed * solar_wind_speed * 6.68459e-9
```

### **3. Upstream TSDSA Model (Mittag–Leffler Function)**
Applies the fractional transport model:

\[
f(x)=e^{-\lambda x}E_{lpha-1}\left[-\left(rac{x}{L}ight)^{lpha-1}ight]
\]

Extracts parameters:
- L (scale length)
- α (superdiffusion index)
- λ⁻¹ (inverse tempering)

### **4. Downstream Tempered Exponential Model**

\[
f(x) = A \exp(-\lambda |x|)
\]

Extracts:
- λ⁻¹  
- scale factor A  

### **5. Plotting Tools**
- Upstream vs downstream comparison plots  
- Energy-channel trends for:  
  - α  
  - L  
  - λ⁻¹  

---

## 🧪 Example Script — `example_tsdsa_shock1.py`

This script:

- Imports the module
- Loads EPAM data
- Converts time → distance  
- Fits P2–P5 channels  
- Saves plots into `figures/`

### **Run the example file:**

```bash
python example_tsdsa_shock1.py
```

Outputs saved to:

```
figures/
```

---

## 📦 Dependencies

Install using:

```bash
pip install numpy pandas matplotlib scipy
pip install mittag-leffler
```

---

## 🎯 Scientific Interpretation

- Upstream shows **superdiffusive transport** (1 < α < 2).  
- Downstream fluxes decay faster due to **stronger turbulence**.  
- L increases with particle energy.  
- λ⁻¹ changes differently upstream vs downstream.

---

## ✔ Skills Demonstrated

- Scientific module creation  
- Use of Mittag–Leffler functions  
- Nonlinear curve fitting  
- Structured data handling  
- Modular programming  
- GitHub documentation and project organization  

---

## 📬 Contact

For questions related to this project, feel free to reach out.

