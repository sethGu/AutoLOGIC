# Dataset layout

The principal classification, multiclassification and regression runners read
serialized datasets relative to the `autologic` source directory:

```text
autologic/data/<dataset>.pkl
```

CSV datasets and descriptions:

```text
autologic/data/csv_data/<dataset>.csv
autologic/data/csv_data/<dataset>-description.txt
```

Benchmark tables and run records:

```text
data/paper/
data/paper/base_tables/
data/paper/base_run/
```

Some reproduction scripts, the compatibility CSV loader, and the strong cached
regression runner support environment-based paths. Check the selected entry
point before using:

```text
AUTOLOGIC_DATA_DIR
AUTOLOGIC_CSV_DATA_DIR
AUTOLOGIC_REGRESSION_DATA_DIR
```

The main classification and regression loaders do not read these overrides.
Legacy SAGE runners use corresponding paths under `legacy_sage/data/`.
The included offline example reads `demo/data/demo_binary.csv` directly and
does not need benchmark data. See [the demo guide](../demo/README.md).

## Benchmark datasets

- Classification: `cd1`, `cc1`, `ld1`, `credit-g`, `cc2`, `cd2`, `cf1`, `adult`, `bank`, `blood`, `heart`, `pc1`, `pc3`, `tic-tac-toe`, `balance-scale`, `cmc`, `eucalyptus`, `jungle_chess`, `car`, `vehicle`
- Regression: `boston`, `concrete`, `insurance`, `crab`, `winequality`, `california`, `bike`, `forest`, `wind`, `puma8nh`
- Clustering: `breast`, `glass`, `iris`, `students`, `seeds`, `cd2`, `ld1`, `cd1`, `ld2`, `cc3`
