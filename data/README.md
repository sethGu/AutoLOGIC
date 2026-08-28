# Dataset layout

Serialized datasets:

```text
data/pkl/<dataset>.pkl
```

CSV datasets and descriptions:

```text
data/csv_data/<dataset>.csv
data/csv_data/<dataset>-description.txt
```

Benchmark tables and run records:

```text
data/paper/
data/paper/base_tables/
data/paper/base_run/
```

Environment variables:

```text
AUTOLOGIC_DATA_DIR
AUTOLOGIC_CSV_DATA_DIR
AUTOLOGIC_REGRESSION_DATA_DIR
```

## Benchmark datasets

- Classification: `cd1`, `cc1`, `ld1`, `credit-g`, `cc2`, `cd2`, `cf1`, `adult`, `bank`, `blood`, `heart`, `pc1`, `pc3`, `tic-tac-toe`, `balance-scale`, `cmc`, `eucalyptus`, `jungle_chess`, `car`, `vehicle`
- Regression: `boston`, `concrete`, `insurance`, `crab`, `winequality`, `california`, `bike`, `forest`, `wind`, `puma8nh`
- Clustering: `breast`, `glass`, `iris`, `students`, `seeds`, `cd2`, `ld1`, `cd1`, `ld2`, `cc3`
