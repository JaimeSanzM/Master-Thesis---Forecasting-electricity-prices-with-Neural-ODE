# Forecasting Electricity Prices with Neural ODE

This repository contains the Python implementation and results of my Master's Thesis, "Forecasting Electricity Prices with Neural ODE," completed as part of my Master’s in Industrial Engineering and Smart Industry at the Universidad Pontificia Comillas. The thesis explores the application of neural Ordinary Differential Equations (ODEs) for short-term electricity price forecasting in the 2019 Spanish day-ahead market.

## Abstract

This master's thesis develops and evaluates neural Ordinary Differential Equations (ODEs) for short-term electricity price forecasting (EPF) in the 2019 Spanish day-ahead market. Implemented in PyTorch, the neural ODE models are compared against several benchmark models including SARIMA, Facebook Prophet, MLP, LSTM, and CNN-LSTM. The neural ODEs, both univariate and multivariate, demonstrate promising results, often outperforming traditional statistical and machine learning methods in terms of average error metrics. This innovative approach highlights the potential of neural ODEs in handling complex, dynamic systems like electricity price forecasting, where exogenous factors such as weather and market dynamics play a significant role.

### Keywords
Forecasting, Electricity Price, ODE, Day-ahead Market

## Repository Structure

- `src/`: Contains all Python scripts for the neural ODE models and benchmark comparisons.
- `data/`: Dataset directory (please add your data following the guidelines provided).
- `results/`: Results and output from the models, including prediction accuracy and computational performance metrics.
- `figures/`: Folder containing graphs and figures generated during the study. Examples include:

    <!-- ![Figure 1](figures/fig1.png)
    *Figure 1: Comparison of Model Performances*

    ![Figure 2](figures/fig2.png)
    *Figure 2: Neural ODE Model Predictions vs. Actual Prices* -->

## Usage

Instructions on how to set up and run the models:

```bash
python setup.py install
python run_models.py
