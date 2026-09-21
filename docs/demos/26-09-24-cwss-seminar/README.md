# CWSS 2026 remote-Kerchunk demonstration

This directory contains the presentation notebook for the September 24, 2026
Climate and Weather Seminar Series talk, *New Tools for Accessing CMIP Data at
NERSC and Beyond*.

The [companion presentation](https://docs.google.com/presentation/d/1eDkwAIJC_peYnRnLnicplOPiR1eqiDZj2Sgwfrkvvrg/edit?slide=id.g3fa9c64b4de_0_73#slide=id.g3fa9c64b4de_0_73)
provides the broader seminar context. The notebook is its remote JSON/Kerchunk
demonstration component.

## Environment setup

Create and activate an environment that includes xCDAT and the notebook
dependencies. The exact xCDAT installation should match the environment used
for the presentation.

```bash
conda create -n cwss-kerchunk -c conda-forge \
    xcdat matplotlib cartopy fsspec zarr jupyter ipykernel
conda activate cwss-kerchunk
python -m pip install git+https://github.com/PCMDI/xsearch.git
python -m ipykernel install --user --name cwss-kerchunk \
    --display-name "CWSS Kerchunk"
```

`xsearch` is not a dependency of xCDAT. It is required only for the local
NERSC comparison; install it from the
[PCMDI/xsearch repository](https://github.com/PCMDI/xsearch). Record the
tested xsearch commit in presentation notes before publishing a reproducible
release of this demo.

## Data inputs

- `kerchunk_list.json` is a Kerchunk catalog snapshot used to identify remote
  CMIP6 reference JSON files. It is hosted in
  [`xcdat-data`](https://github.com/xCDAT/xcdat-data/blob/e31bf6cdfd478550e9a284e6d17ef35edce5ee03/resources/kerchunk_list.json),
  not this repository, because the 50 MB file is a data asset. The notebook uses
  `pooch` to retrieve the catalog from commit
  `e31bf6cdfd478550e9a284e6d17ef35edce5ee03`, verify its SHA-256 checksum, and
  cache it under the xCDAT application cache directory.
- `ecsdata.pickle` contains model equilibrium climate sensitivity (ECS) values
  prepared from Zelinka's repository. Treat this repository-managed pickle as
  trusted input; do not load untrusted pickle files.

The notebook's remote portion uses an ORNL-hosted Kerchunk reference catalog.
The local comparison additionally requires NERSC data access. Remote URLs,
catalog contents, data availability, network conditions, and filesystem caches
can change; reported timings are illustrative rather than benchmarks.
