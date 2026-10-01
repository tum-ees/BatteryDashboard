# Measurement data

Place local measurement data below this directory using one folder per battery ID:

```text
data/
└── <BATTERY_ID>/
    ├── Aging/
    │   └── *.json
    ├── CU/
    │   └── *.json
    └── GEIS/
        └── *.json
```

`Ageing/` is also accepted as an alternative spelling for `Aging/`.

`BAT_EXAMPLE` is a synthetic data set committed to the repository for demonstration and testing. Additional battery folders are ignored by the repository `.gitignore` by default.

## License of the included example data

The files in `BAT_EXAMPLE/` are demonstration data and are released under
**CC0 1.0 Universal**. See [`../DATA_LICENSE.md`](../DATA_LICENSE.md).
