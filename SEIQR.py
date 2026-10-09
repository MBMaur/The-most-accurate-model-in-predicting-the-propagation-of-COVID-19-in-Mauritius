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
N = 1_270_000  # Total population
initial_cumulative = real_cases[0]
target_final = real_cases[-1]

# Initial population guesses

I0_guess = 267       # initial infected guess
Q0_guess = 50        # initial quarantined guess
R0_guess = initial_cumulative - I0_guess - Q0_guess
E0_guess = 100       # initial exposed guess
S0 = N - E0_guess - I0_guess - Q0_guess - R0_guess

print("\n Initial Population Guesses")
print("")
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

initial_guess = [7.606299*(10**-7), 0.2, 0.0714, 0.0714,0.0714, 0.1, 0.1, I0_guess, Q0_guess, E0_guess]
bounds = [
    (0, 1),  # beta
    (0, 1),  # sigma
    (0, 1),  # gammaE
    (0, 1),  # gammaI
    (0, 1),  # gammaQ
    (0, 1),  # deltaE
    (0, 1),  # deltaI
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
print("")
print(f" Optimized β: {beta/N}")
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

plt.figure(figsize=(10, 6))
plt.plot(t, best_model, label="SEIQR Model", color='purple')
plt.scatter(t, real_cases, label="Real Data", color='red', s=5)
plt.title("Optimized SEIQR Model vs Real COVID-19 Cumulative Cases (Mauritius)")
plt.xlabel("Number of Days from 2 Aug 2021")
plt.ylabel("Total Cumulative Cases")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
