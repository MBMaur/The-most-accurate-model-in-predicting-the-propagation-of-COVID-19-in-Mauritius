import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

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
#SIR Model

t = np.arange(len(real_cases))
days = len(t)
N = 1_270_000  # Total population
initial_cumulative = real_cases[0]
target_final = real_cases[-1]


# Initialising  population guess

I0_guess1 = 267
R0_guess2 = initial_cumulative - I0_guess1
S0_guess3 = N - I0_guess1 - R0_guess2

print("\n Initial Population Guesses")
print("---------------------------")
print(f"Initial Infected (I₀):      {I0_guess1}")
print(f"Initial Recovered (R₀):     {R0_guess2}")
print(f"Initial Susceptible (S₀):   {S0_guess3}")


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

initial_guess = [0.2, 0.1, I0_guess1]  # [beta, gamma, I0]
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

print("\n SIR Model Evaluation Summary")
print("-----------------------------")
print(f"Optimized β: {beta_opt:.4f}")
print(f"Optimized γ: {gamma_opt:.4f}")
print(f"Optimized I₀: {I0_opt:.0f}")
print(f"Optimized R₀: {R0_opt:.0f}")
print(f"Initial Cumulative Cases: {initial_cumulative:,.0f}")
print(f"Final Model Cases:        {final_model[-1]:,.0f}")
print(f"Final Actual Cases:       {target_final:,.0f}")
print(f"RMSE:                     {rmse:.2f} cases")
print(f"% Error:                  {percent_error:.2f}%")

# SEIQR Model
# Initial population guesses

I0_guess = 267       # initial infected guess
Q0_guess = 50        # initial quarantined guess
R0_guess = initial_cumulative - I0_guess - Q0_guess
E0_guess = 100       # initial exposed guess
S0 = N - E0_guess - I0_guess - Q0_guess - R0_guess

print("\n Initial Population Guesses")
print("---------------------------")
print(f"Initial Exposed (E₀):      {E0_guess}")
print(f"Initial Infected (I₀):     {I0_guess}")
print(f"Initial Quarantined (Q₀):  {Q0_guess}")
print(f"Initial Recovered (R₀):    {R0_guess}")
print(f"Initial Susceptible (S₀):  {S0}")


# SEIQR Model using Euler's Method

def seiqr_model(beta, sigma, gammaE, gammaI, gammaQ, deltaE, deltaI, I0, Q0, E0, days):
    S, E, I, Q, R = np.zeros(days), np.zeros(days), np.zeros(days), np.zeros(days), np.zeros(days)
    cumulative = np.zeros(days)

    S[0], E[0], I[0], Q[0], R[0] = S0, E0, I0, Q0, R0_guess
    cumulative[0] = initial_cumulative

    for i in range(1, days):
        dSdt = -beta * S[i-1] * I[i-1] / N
        dEdt = beta * S[i-1] * I[i-1] / N - (sigma + gammaE + deltaE) * E[i-1]
        dIdt = sigma * E[i-1] - (gammaI + deltaI) * I[i-1]
        dQdt = deltaE * E[i-1] + deltaI * I[i-1] - gammaQ * Q[i-1]
        dRdt = gammaE * E[i-1] + gammaI * I[i-1] + gammaQ * Q[i-1]

        S[i] = max(S[i-1] + dSdt, 0)
        E[i] = max(E[i-1] + dEdt, 0)
        I[i] = max(I[i-1] + dIdt, 0)
        Q[i] = max(Q[i-1] + dQdt, 0)
        R[i] = max(R[i-1] + dRdt, 0)

        cumulative[i] = I[i] + Q[i] + R[i]

    # Scale only from day 1 onward so both start and end match
    if cumulative[-1] != cumulative[0]:
        scaling_factor = (target_final - cumulative[0]) / (cumulative[-1] - cumulative[0])
        cumulative[1:] = cumulative[0] + (cumulative[1:] - cumulative[0]) * scaling_factor

    return cumulative


# Objective function

def objective(params, t, real_cases):
    try:
        beta, sigma, gammaE, gammaI, gammaQ, deltaE, deltaI, I0, Q0, E0 = params
        model_cases = seiqr_model(beta, sigma, gammaE, gammaI, gammaQ, deltaE, deltaI, I0, Q0, E0, len(t))
        if np.any(np.isnan(model_cases)) or np.any(np.isinf(model_cases)):
            return np.inf
        return np.sqrt(mean_squared_error(real_cases, model_cases))
    except Exception:
        return np.inf

# Optimization

initial_guess = [0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, I0_guess, Q0_guess, E0_guess]
bounds = [
    (0, 0.5),  # beta
    (0, 0.5),  # sigma
    (0, 0.5),  # gammaE
    (0, 0.5),  # gammaI
    (0, 0.5),  # gammaQ
    (0, 0.5),  # deltaE
    (0, 0.5),  # deltaI
    (0, 1000),  # I0 (initial infected)
    (0, 1000),  # Q0 (initial quarantined)
    (0, 1000)   # E0 (initial exposed)
]

# Minimize the objective function
result = minimize(objective, initial_guess, args=(t, real_cases), bounds=bounds)
beta, sigma, gammaE, gammaI, gammaQ, deltaE, deltaI, I0_opt, Q0_opt, E0_opt = result.x


# Final model and evaluation

best_model = seiqr_model(beta, sigma, gammaE, gammaI, gammaQ, deltaE, deltaI, I0_opt, Q0_opt, E0_opt, len(t))
rmse = np.sqrt(mean_squared_error(real_cases, best_model))
percent_error = (rmse / np.mean(real_cases)) * 100
# Output results

print("\n SEIQR Model Evaluation Summary")
print("-------------------------------")
print(f" Optimized β: {beta:.4f}")
print(f" Optimized σ: {sigma:.4f}")
print(f" Optimized γE: {gammaE:.4f}")
print(f" Optimized γI: {gammaI:.4f}")
print(f" Optimized γQ: {gammaQ:.4f}")
print(f" Optimized δE: {deltaE:.4f}")
print(f" Optimized δI: {deltaI:.4f}")
print(f" Initial Cumulative Cases (P₀): {initial_cumulative:,.0f}")
print(f" Optimized Initial Infected (I₀): {I0_opt:.0f}")
print(f" Optimized Initial Quarantined (Q₀): {Q0_opt:.0f}")
print(f" Optimized Initial Exposed (E₀): {E0_opt:.0f}")
print(f" Initial Recovered (R₀): {R0_guess}")
print(f" Final Cumulative Cases (model): {best_model[-1]:,.0f}")
print(f" Final Cumulative Cases (actual): {target_final:,.0f}")
print(f" RMSE: {rmse:.2f} cases")
print(f"️ % Error (RMSE / mean actual): {percent_error:.2f}%")

# Plotting

# Outputing the information
print("\n Model Evaluation Summary")
print("----------------------------")
print(f" Growth Rate (r):               {r_fit:.4f}")
print(f" Initial Cumulative Cases (P₀): {P0_fixed:,.0f}")
print(f"Final Cumulative Cases (K):    {K_fixed:,.0f}")
print(f" RMSE:                          {rmse:.2f} cases")
print(f" % Error (RMSE / mean actual): {percent_error:.2f}%")
# Plotting both graphs
plt.figure(figsize=(10, 6))
plt.scatter(real_days, real_cases, color='red', label='Real Data', s=2)
plt.plot(t, final_model, label="SIR Model", color='black', linewidth=4)
plt.plot(t, best_model, label="SEIQR Model", color='#ADD8E6',linewidth=2)
plt.plot(t_model, model_cases, color='blue', label='Fitted Logistic Model', linewidth=2)
plt.title(" Optimized Models vs Real COVID-19 Cumulative Cases (Mauritius)")
plt.xlabel("Number of Days from 2 Aug 2021")
plt.ylabel("Total Cumulative Cases")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
