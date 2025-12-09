# Evidence for Tempered Superdiffusive Shock Acceleration at a Nearly-Perpendicular Shock 
### AES 509 Final Project — Python Implementation  
### Author: Rubaiya Khondoker
### Module: `tsdsa_shock1.py`  
### Example Script: `example_tsdsa_shock1.py`  

---

## 🔭 Scientific Objective

The purpose of this project is to determine whether the energetic particle
profiles observed around the **26 August 1998 quasi-perpendicular interplanetary shock**
are consistent with **Tempered Superdiffusive Shock Acceleration (TSDSA)**.

TSDSA is a modern extension of classical shock acceleration theory, incorporating:

- **Lévy-like superdiffusive step-size statistics**
- **Exponential tempering**, which suppresses extremely long free paths
- **Mittag–Leffler kernels**, which naturally arise in fractional transport equations
- **Different upstream and downstream transport behaviors**

This project applies TSDSA theory directly to spacecraft observations by:

1. Reading ACE/EPAM energetic particle flux measurements  
2. Converting time into distance from the shock  
3. Fitting upstream data using a **Mittag–Leffler decay profile**  
4. Fitting downstream data using a **tempered TSDSA integral solution**  
5. Extracting:
   - fractional index α
   - superdiffusive length scale L
   - tempering scale λ⁻¹
6. Comparing trends across energy channels P2–P5

The goal is to evaluate whether SEP (Solar Energetic Particle) transport
is **weakly, moderately, or strongly superdiffusive**, and whether this behavior
is **different upstream and downstream of the shock**.

---

## 📦 Project Components

This repository contains two major code files and several helper modules.

---

### **1. `tsdsa_shock1.py` — Main TSDSA Analysis Module**

This file contains **all scientific models and utilities** needed for analysis:

#### ✔ Data Loading & Cleaning
- Reads ACE/EPAM CSV data  
- Removes invalid values  
- Interpolates small gaps  
- Normalizes flux values  

#### ✔ Distance Conversion (Shock-Normal Coordinate)
Time is converted to distance using:

```
x = (t_shock - t) * V_sw * C
```

where:

- `V_sw = 668 km/s`
- `C = 6.68459 × 10⁻9 AU/(km/s)`
- `x > 0` → upstream  
- `x < 0` → downstream 


#### ✔ Upstream TSDSA Model (Mittag–Leffler)
Upstream transport follows:

```
f_up(x) = exp(-λ x) * E_{α-1}( - (x / L)^(α-1) ),    x > 0
```

where:

- `E_{α-1}` = Mittag–Leffler function  
- `α` controls the degree of superdiffusion  
- `L` sets the intermediate decay scale  
- `λ⁻¹` is the tempering scale  

The module returns `(L_up, α_up, λ_up⁻¹)` with uncertainties.

---
 
#### ✔ Downstream TSDSA Model (Tempered Integral Kernel)

Downstream transport requires computing the kernel:

```
I(|x|) = ∫₀^{|x|} exp(-λ x') * E_{α-1}( - (x' / L_sd2)^(α-1) ) dx'
```

### ⭐ **Main downstream TSDSA equation (added)**

```
f_dn(x) = A * exp(-λ |x|) * E_{α-1}( - (|x| / L_sd2)^(α - 1) )
          + B * ( 1 - λ * I(|x|) )
```

This is the full tempered TSDSA downstream solution.

To avoid slow fitting:

- The integral is **precomputed** on a grid  
- A linear interpolator is used during curve fitting  

The downstream fit returns `(α_dn, λ_dn⁻¹, L_sd2)`.

---

#### ✔ Plotting Functions

The module includes plotting routines for:

- Upstream & downstream profiles  
- α vs energy  
- L vs energy  
- λ⁻¹ vs energy  

These create publication-quality figures.

---

### **2. `example_tsdsa_shock1.py` — Complete Reproducible Workflow**

This script ties everything together.  
Running it:

```bash
python example_tsdsa_shock1.py
```

Will:

1. Load the ACE EPAM file  
2. Convert timestamps to distance \( x \)  
3. Extract upstream (0 < x < 0.05 AU)  
4. Extract downstream (–0.05 AU < x < 0)  
5. Fit the TSDSA models to channels P2–P5  
6. Save all figures to `figures/`  
7. Print best-fit TSDSA parameters for each channel  

This script is the single point of execution required by the rubric.

---

## 🧮 Custom Mittag–Leffler Function Files

Since SciPy does not provide the Mittag–Leffler function for general parameters,
the project includes **custom implementations**:

- `special_functions/mittag_leffler.py`
- `special_functions/ml.py`
- `special_functions/mlinternational.py`

These modules ensure stable and accurate evaluation of:
Eβ(z)

where β = α − 1

---

## 📁 Folder Structure (Rubric-Compliant)

```
AES-509_Project_Final/
│
├── tsdsa_shock1.py                # Main TSDSA model + fitting functions
├── example_tsdsa_shock1.py        # Reproducible example script
├── README.md                      # This documentation
├── requirements.txt               # Python dependencies
├── LICENSE                        # License information
├── .gitignore                     # Version control hygiene
│
├── special_functions/             # Mittag–Leffler implementations
│     ├── mittag_leffler.py
│     ├── ml.py
│     └── mlinternational.py
│
├── figures/                       # Generated plots
│     ├── tsdsa1_profiles_P2_P5.png
│     ├── tsdsa1_trends_P2_P5_alpha_up_down.png
│     ├── tsdsa1_trends_P2_P5_L_up_down.png
│     └── tsdsa1_trends_P2_P5_lambda_inv_up_down.png
│
└── data/                          # Optional ACE EPAM data file
      └── AC_H3_EPM_614092.csv
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/rksm0014/AES-509_Project_Final
cd AES-509_Project_Final
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

Run the example script:

```bash
python example_tsdsa_shock1.py
```

You will see:

- Printed values of α, L, λ⁻¹ for each energy channel  
- Profile comparison plots  
- Energy trend plots  

Outputs are saved automatically inside `figures/`.

---

## 🔬 Summary of Scientific Results

Based on the fits obtained:

### ✔ Both upstream and downstream regions show **superdiffusive** behavior  
1 < α < 2

### ✔ Downstream transport is **more superdiffusive** (smaller α)

### ✔ Transport length scale \( L \) **increases with energy**

### ✔ Tempering scale λ⁻¹ is **approximately constant (~0.1 AU)**  
This indicates a consistent transition scale from superdiffusion → normal diffusion.

### ✔ TSDSA models match the observed EPAM flux profiles extremely well  
showing that the shock is consistent with **tempered superdiffusive SEP transport**.

---

## 👩‍💻 Contact

**Rubaiya Khondoker**  
Graduate Student, UAH  
AES 509 — Scientific Programming  
Feel free to contact anytime for any questions.

