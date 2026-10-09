# The-most-accurate-model-in-predicting-the-propagation-of-COVID-19-in-Mauritius
Compares the Logistic Growth, SIR and SEIQR models to find which best predicts COVID-19 spread in Mauritius (2 Aug 2021 to 30 Mar 2025). Python code fits each model to real data and evaluates it with RMSE and % error. Math AA HL Internal Assessment.

# Predicting the Propagation of COVID-19 in Mauritius

Code for my Math AA HL Internal Assessment. It compares the **Logistic Growth**, **SIR** and **SEIQR** models to find which best fits real COVID-19 cumulative case data for Mauritius (2 Aug 2021 to 30 Mar 2025, 1,337 days).

## Models

- **Logistic Growth:** one equation with growth rate `r` and carrying capacity `K`.
- **SIR:** Susceptible, Infected, Recovered.
- **SEIQR:** Susceptible, Exposed, Infected, Quarantined, Recovered.

All three assume a fixed population (1.27 million) and constant rates.

## Euler's Method

SIR and SEIQR are systems of differential equations, so they are solved numerically with Euler's method using a step size of 1 day:

```
y(t) = y(t-1) + h × f(t-1, y(t-1)),   h = 1 day
```

For example, in the SIR model:

```
S(t) = S(t-1) - β·S(t-1)·I(t-1)
I(t) = I(t-1) + β·S(t-1)·I(t-1) - γ·I(t-1)
R(t) = R(t-1) + γ·I(t-1)
```

Each population is updated day by day from its previous value plus its rate of change.

## Method

1. Load the data (Our World in Data, cumulative cases per million × 1.27).
2. Fit each model with `scipy.optimize` (`curve_fit` for Logistic Growth, `minimize` for SIR and SEIQR).
3. Evaluate with **RMSE** and **% Error Margin** = RMSE / mean of actual cases × 100.

## Results

| Model | RMSE (cases) | % Error Margin |
|-------|-------------:|---------------:|
| Logistic Growth | 28,197.48 | 10.84% |
| SIR | 12,268.31 | 4.72% |
| **SEIQR** | **12,166.32** | **4.68%** |

**SEIQR was the most accurate**, since its Exposed and Quarantined populations capture incubation and isolation effects.

## Running the code

```bash
pip install numpy pandas matplotlib scipy scikit-learn
python logistic_growth.py
python sir_model.py
python seiqr_model.py
```

Keep `MauritiusCOVID.csv` in the same folder as the scripts. Each script prints the optimised parameters, RMSE and % error, and plots the model against the real data.

## Limitations

Constant parameters mean lockdowns, policy changes and new variants are not modelled, and reinfection is ignored.

## Data sources

- [Our World in Data](https://ourworldindata.org/covid-cases)
- [Worldometer](https://www.worldometers.info/coronavirus/country/mauritius/)
