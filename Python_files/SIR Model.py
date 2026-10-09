import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize


# Loading Real-world data


df = pd.read_csv("MauritiusCOVID.csv")
df = df[df['Entity'] == 'Mauritius'].reset_index(drop=True)
real_cases = df['Total confirmed cases of COVID-19 per million people'].to_numpy() * 1.27

t = np.arange(len(real_cases))
days = len(t)
N = 1_270_000  # Total population
initial_cumulative = real_cases[0]
target_final = real_cases[-1]


# Initialising  population guess

I0_guess = 267
R0_guess = initial_cumulative - I0_guess
S0_guess = N - I0_guess - R0_guess

print("Prediction for Initial Populations")
print("")
print(f"Initial Infected (I₀):      {I0_guess}")
print(f"Initial Recovered (R₀):     {R0_guess}")
print(f"Initial Susceptible (S₀):   {S0_guess}")


# SIR Model using Euler's Method

def sir_model(beta, gamma, I0, R0, days):
    S = np.zeros(days)
    I = np.zeros(days)
    R = np.zeros(days)
    cumulative = np.zeros(days)

    S[0] = N - I0 - R0
    I[0] = I0
    R[0] = R0
    cumulative[0] = I0 + R0

    for i in range(1, days):
        dS = -beta * S[i-1] * I[i-1] / N
        dI = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        dR = gamma * I[i-1]

        S[i] = S[i-1] + dS
        I[i] = I[i-1] + dI
        R[i] = R[i-1] + dR

        cumulative[i] = I[i] + R[i]

    # Scaling so that start and end match the real data
    if cumulative[-1] != cumulative[0]:
        scaling_factor = (target_final - cumulative[0]) / (cumulative[-1] - cumulative[0])
        cumulative[1:] = cumulative[0] + (cumulative[1:] - cumulative[0]) * scaling_factor

    return cumulative

# Objective function for optimization

def objective(params, t, real_cases):
    beta, gamma, I0 = params
    R0 = initial_cumulative - I0
    model = sir_model(beta, gamma, I0, R0, len(t))
    if np.any(np.isnan(model)) or np.any(np.isinf(model)):
        return np.inf
    return np.sqrt(mean_squared_error(real_cases, model))

# Optimization

initial_guess = [7.606299*(10**-7), 0.0714, I0_guess]  # [beta, gamma, I0]
bounds = [
    (0, 0.5),   # beta
    (0, 0.5),   # gamma
    (0, initial_cumulative)  # I0 must be ≤ initial cases
]

result = minimize(objective, initial_guess, args=(t, real_cases), bounds=bounds)
beta_opt, gamma_opt, I0_opt = result.x
R0_opt = initial_cumulative - I0_opt


# Final model and evaluation
# -------------------------------
final_model = sir_model(beta_opt, gamma_opt, I0_opt, R0_opt, days)
rmse = np.sqrt(mean_squared_error(real_cases, final_model))
percent_error = (rmse / np.mean(real_cases)) * 100


# Output results

print("SIR Model  Summary")
print("")
print(f"Optimized β: {beta_opt/N}")
print(f"Optimized γ: {gamma_opt:.4f}")
print(f"Optimized I₀: {I0_opt:.0f}")
print(f"Optimized R₀: {R0_opt:.0f}")
print(f"Initial Cumulative Cases: {initial_cumulative:,.0f}")
print(f"Final Model Cases:        {final_model[-1]:,.0f}")
print(f"Final Actual Cases:       {target_final:,.0f}")
print(f"RMSE:                     {rmse:.2f} cases")
print(f"% Error Margin:                  {percent_error:.2f}%")


# Plotting

plt.figure(figsize=(10, 6))
plt.plot(t, final_model, label="SIR Model", color='green', linewidth=2)
plt.scatter(t, real_cases, label="Real Data", color='red', s=5)
plt.title("Optimized SIR Model vs Real COVID-19 Cumulative Cases (Mauritius)")
plt.xlabel("Number of Days from 2 Aug 2021")
plt.ylabel("Total Cumulative Cases")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
