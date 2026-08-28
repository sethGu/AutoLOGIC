# Data placement and provenance requirements

Raw datasets are intentionally absent from this pre-release candidate. Before public upload, add a machine-readable data manifest containing each dataset's source URL or stable identifier, retrieval date, checksum, target column, task type, preprocessing steps, split policy, and redistribution status.

## Expected local layout

Set `AUTOLOGIC_DATA_DIR` to a directory containing serialized dataset files:

```text
data/pkl/<dataset>.pkl
```

Set `AUTOLOGIC_CSV_DATA_DIR` and `AUTOLOGIC_REGRESSION_DATA_DIR` when CSV-based routes are used:

```text
data/csv_data/<dataset>.csv
data/csv_data/<dataset>-description.txt
```

Place publication workbooks and historical reconstruction inputs under:

```text
data/paper/
data/paper/base_tables/
data/paper/base_run/
```

## Full benchmark dataset names

- Classification: `cd1`, `cc1`, `ld1`, `credit-g`, `cc2`, `cd2`, `cf1`, `adult`, `bank`, `blood`, `heart`, `pc1`, `pc3`, `tic-tac-toe`, `balance-scale`, `cmc`, `eucalyptus`, `jungle_chess`, `car`, `vehicle`
- Regression: `boston`, `concrete`, `insurance`, `crab`, `winequality`, `california`, `bike`, `forest`, `wind`, `puma8nh`
- Clustering: `breast`, `glass`, `iris`, `students`, `seeds`, `cd2`, `ld1`, `cd1`, `ld2`, `cc3`

Aliases used by historical runners are preserved in their manifests and scripts. Do not commit API credentials, proprietary datasets, or files lacking redistribution permission.

