All of the code is implemented based on MATLAB, and the version is required to be MATLAB 2022a or above. Lower versions may not be able to ensure the operation of QRTCN, ERTCN, and LWCRPS-based deep learning.

Individual models: Models 1 - 9 are individual risk models. The prediction results can be obtained by running the "main_" function.

Forecast combination: "main_combine_evaluate" is the main function for the prediction combination with regularization. Running it will yield the prediction combination results and evaluation results.

Data: We downloaded the S&P 500 index from an open-source website to ensure reproducibility without violating the copyright agreement.

Features: In the example, only the lagged leverage effect is constructed. Readers can add volatility on their own.

We will continue to refine and thoroughly comment on the code, and we invite researchers in related fields to engage in discussion with us.
