import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

# Loading Real-world data
df = pd.read_csv("MauritiusCOVID.csv")
df = df[df['Entity'] == 'Mauritius'].reset_index(drop=True)

real_cases = df['Total confirmed cases of COVID-19 per million people'].to_numpy() * 1.27
real_days = np.arange(len(real_cases))  # Days since start

# Fixing  initial and final total cumulative case
P0_fixed = real_cases[0]
K_fixed = real_cases[-1]

# Logistic model with only r as a parameter
def logistic_fixed_P0_K(t, r):
    return (P0_fixed * K_fixed * np.exp(r * t)) / ((K_fixed - P0_fixed) + P0_fixed * np.exp(r * t))

# Initial guess for r
initial_guess = [0.01883]

# Fitting the model to the Real-world data
popt, _ = curve_fit(logistic_fixed_P0_K, real_days, real_cases, p0=initial_guess)
r_fit = popt[0]

print(f" Fitted growth rate r: {r_fit:.4f}")

# Generating model values
t_model = np.linspace(0, real_days[-1], 300)
model_cases = logistic_fixed_P0_K(t_model, r_fit)

# Calculating  predictions for actual days for RMSE
model_fit_to_actual = logistic_fixed_P0_K(real_days, r_fit)
rmse = np.sqrt(mean_squared_error(real_cases, model_fit_to_actual))

# Calculating  percentage error margin from RMSE
mean_actual = np.mean(real_cases)
percent_error = (rmse / mean_actual) * 100

# Outputing the information
print("\n Logistic Growth Model  Summary")
print("")
print(f" Growth Rate (r):               {r_fit:.4f}")
print(f" Initial Cumulative Cases (P₀): {P0_fixed:,.0f}")
print(f" Final Cumulative Cases (K):    {K_fixed:,.0f}")
print(f" RMSE:                          {rmse:.2f} cases")
print(f" % Error Margin : {percent_error:.2f}%")
# Plotting both graphs
plt.figure(figsize=(10, 6))
plt.scatter(real_days, real_cases, color='red', label='Real Data', s=2)
plt.plot(t_model, model_cases, color='blue', label='Fitted Logistic Model', linewidth=2)
plt.title(" Optimized Logistic Growth Curve vs Real COVID-19 Cumulative Cases (Mauritius)")
plt.xlabel("Number of Days from 2 Aug 2021")
plt.ylabel("Total Cumulative Cases")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
