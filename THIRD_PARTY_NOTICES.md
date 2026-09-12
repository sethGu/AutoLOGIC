# Third-party attribution

AutoLOGIC is distributed under the Apache License, Version 2.0; see `LICENSE`.

## CAAFE

Parts of `autologic/mystage1/` and `legacy_sage/mycaafe/` are adapted from
[CAAFE](https://github.com/automl/CAAFE), by Noah Hollmann, Samuel Müller and
Frank Hutter (2023). The retained upstream notice is in
`LICENSES/CAAFE-LICENSE.txt`. These modules have been modified for AutoLOGIC's
task handling, generation, execution checks, and logging. Their original
authors retain their copyright.

## Dependencies and datasets

Separately installed dependencies retain their respective licenses. This
repository's Apache-2.0 license does not relicense external benchmark datasets,
pretrained model weights, or remote API services. Dataset access is governed by
each original provider's terms. In particular, FinBench is marked CC BY-NC 4.0
by its provider.

`demo/data/demo_binary.csv` is synthetic data created by
`demo/generate_sample.py` and is included under Apache-2.0. It contains no
third-party dataset records. Public benchmark data are not bundled in the demo.
