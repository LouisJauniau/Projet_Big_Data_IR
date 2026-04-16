# Big Data - Information Retrieval

## Overview

This project implements three retrieval algorithms over the same passage index, and compares their performance on the msmarco dataset:

- SPLADE
- ColBERT
- DPR

It also includes a desktop search GUI so a user can choose an algorithm from a dropdown, type a query, and inspect ranked results in one window.

## Prerequisites

- Python 3.8+
- pip
- Tkinter (sometimes requires separate installation on Linux-based systems)

## How to use

The GUI only works after the database has been prepared locally.

You must do these steps first:

1. Run PostgreSQL locally.
2. Create schema and populate data from the notebooks.
3. Build indexes from the notebooks.

If you skip population or indexing, the GUI will not work correctly.

Use the notebooks in [notebooks](notebooks) in order:

1. [notebooks/01_data_preparation.ipynb](notebooks/01_data_preparation.ipynb): initializes schema and populates base tables. PostgreSQL must be running locally for this step.
2. [notebooks/02_splade.ipynb](notebooks/02_splade.ipynb): builds SPLADE index.
3. [notebooks/03_colbert.ipynb](notebooks/03_colbert.ipynb): builds ColBERT index.
4. [notebooks/04_dpr.ipynb](notebooks/04_dpr.ipynb): builds DPR index.

Once the database is ready, you can launch the GUI with:

```bash
python -m src gui
```

## Existing command line entrypoints

The retrieval modules can still be used directly:

```bash
python -m src.splade.search "what is a bank"
python -m src.colbert.search "what is a bank"
python -m src.dpr.search "what is a bank"
```

## Packaging

If you still want to create a package file, use:

```bash
pyinstaller --clean search_app.spec
```