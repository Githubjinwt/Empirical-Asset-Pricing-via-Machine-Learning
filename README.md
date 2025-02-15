# Empirical-Asset-Pricing-via-Machine-Learning
Empirical Asset Pricing via Machine Learning On Chinese A Share

## Prerequisites

Create results folder:

```
mkdir results
```

Install required modules and tools:

```
pip install backtrader xgboost lightgbm torch numba
```

## Calculate Factors

Calculate factors data for model training.

```
python calculate_factor_data.py
```

## Train Models

Train all models (OLS, PLS......) based on factors data, and save the predicted returns in the results folder.

```
python model_backtest.py
```

## Evaluate

### Backtest Results

Adopt a strategy of going long on the top 10% of stocks with predicted returns in each period, and backtest the results.

```
python bt_backtest.py
```

### Calculate Feature importance

Calculate feature importance of each model.
```
python feature_importance.py
```

## Reference

* [Empirical Asset Pricing via Machine Learning](https://link.zhihu.com/?target=https%3A//academic.oup.com/rfs/article/33/5/2223/5758276%23supplementary-data)
* [The Internet Appendix](https://link.zhihu.com/?target=https%3A//oup.silverchair-cdn.com/oup/backfile/Content_public/Journal/rfs/33/5/10.1093_rfs_hhaa009/4/hhaa009_supplementary_data.pdf%3FExpires%3D1653941700%26Signature%3DQtH8v5-qA2fRgNDSyXdEkBEoLvBKUIkK0Xr4lbWobdWxYP1I2MT32Qw3jvV8QK9iQeFWbMQI2zxcK2Uq0vFYm-rQMXe3aM074sMni-u2QH12pZ7CTzAPDw0VfVH0DoF0i3I02lA3wJAh3tJpcX9nhsMdtl9mt93AfSJJGhjJgIInygPalBrW4b1a-nDiG3zrJufNx1TjMkpkzO~olxPcAJXTguELNntONO8JoL36edF7qLTM8tMR8hq7F6SVltweHbZG1wpZb0XElrcqJ0lzI78IZCmGt3Qrb4keA10FzQhWTTgJNlarNIPxphYi7fYf9Qdz3IvRgCLYRvIDkOafEA__%26Key-Pair-Id%3DAPKAIE5G5CRDK6RD3PGA)

## Acknowledgement

The research presented in this report is the final project for the Financial Big Data course taught by Professor XingTong Zhang at Renmin University of China (RUC). We are deeply grateful to Professor for his insightful guidance and unwavering support throughout the project, which greatly contributed to the successful completion of this work. 