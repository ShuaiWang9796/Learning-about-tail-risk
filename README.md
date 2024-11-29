# QRTCN, ERTCN, and LWCRPS-based Deep Learning in MATLAB
# Please cite this article: 
# Learning about tail risk: machine learning and combination with regularization in market risk management, Omega, 2024, 103249, https://doi.org/10.1016/j.omega.2024.103249.

This repository contains code implemented in MATLAB. The code requires MATLAB 2022a or a higher version to ensure the proper operation of QRTCN, ERTCN, and LWCRPS-based deep learning. Lower versions may not be compatible.

## Individual Models
Models 1 - 9 are individual risk models. To obtain the prediction results, run the `main_` function.

## Forecast Combination
The main function for the prediction combination with regularization is `main_combine_evaluate`. Executing this function will yield the prediction combination results and evaluation results.

## Data
The S&P 500 index data used in this project is downloaded from an open-source website to ensure reproducibility while adhering to the copyright agreement.

## Features
In the provided example, only the lagged leverage effect is constructed. Users are encouraged to add volatility features as per their requirements.

We are committed to continuously improving and thoroughly documenting the code. We welcome researchers in related fields to join us in discussions. For any inquiries or suggestions, please contact us at vvs09061513@163.com. 
