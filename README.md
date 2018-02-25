# risk_parity
risk parity strategy

# How to run the model 
Please run model/risk_parity_model.py. 
Details to be added. 

# Project Description 
In this project, we elastrate how to build up a risk parity portfolio. 
- We start with building a Naive risk parity portfolio where we assume zero-correlation among assets. 
  - We show that this portfolio works reasonably well for basket of assets with low correlations. 
- We then show how to build a proper risk parity portfolio, accounting for correlation among assets in which we equalize risk contribution from each assets. 

## Risk Parity Introduction 
 1. show performance of each asset over time 
 2. show their correlations and performance metrics
 3. show the performance of a traditional 60/40 portfolio and their risk contribution from EQ/Bond over time 
 ## Naive Risk Parity (NRP)
 In NRP, we are correlation blind (i.e., we assume 0 correlation among assets). Therefore, the risk-parity weight is dictated merely by asset volatility, and the weight for asset i is simply inverse of volatility of asset i. In this section, we show how to build Naive Risk Parity portfolios for the following cases and show the asset-wise risk contribution over time. 
  - two asset portfolio (EQ/Bond)
  - three asset portfolio (EQ/Bond/Commodity) 
  - multiple asset portfolio (multiple EQ+ multiple Bond + multiple commodity) 
  
  ## Real Risk Parity 
  
