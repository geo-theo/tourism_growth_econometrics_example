# Weather and National Park Visitation

This portfolio project asks a practical tourism-research question:

> How are unusually warm or wet months associated with recreation visits to
> western U.S. national parks?

The analysis combines monthly National Park Service visitation records with
weather observations requested from the NASA POWER API. It uses a panel
fixed-effects model to separate weather anomalies from each park's usual
seasonality and from shocks shared across parks in a given year.

The notebook emphasizes Glacier National Park while estimating relationships
from a broader sample of 18 western parks.

[Open the executed notebook](tourism_weather_econometrics.ipynb)

In the primary 4,104-observation model, a month that is 1 C warmer than normal
is associated with about 3.2% more visits in normally cold park-months, while
an additional 1 mm/day of precipitation is associated with about 3.0% fewer
visits. These are conditional associations, not complete causal effects.

## What This Demonstrates

- Reproducible ingestion from a public bulk-data endpoint and REST API
- Data validation, caching, reshaping, and source documentation
- Exploratory visualization for highly seasonal tourism data
- Park-by-month and year fixed-effects regression
- Park-clustered standard errors and leave-one-park-out robustness checks
- Clear distinction between statistical association and causal inference

## Run The Project

Create an environment and install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

Then open and run:

```text
tourism_weather_econometrics.ipynb
```

The first run downloads the NPS CSV (about 70 MB) and one small NASA POWER
response per park. Raw files are cached under `data/raw/` and ignored by Git.
The cleaned analysis panel is written to `data/processed/`.

## Data Sources

- [NPS Visitor Use Statistics Data Package, 2024](https://catalog.data.gov/dataset/nps-visitor-use-statistics-data-package-2024)
  - Monthly visitor-use records from 1979–2024
  - Public domain (CC0)
- [NASA POWER Monthly API](https://power.larc.nasa.gov/docs/services/api/temporal/monthly/)
  - Monthly 2-meter temperature and corrected precipitation

## Repository Structure

```text
.
|-- tourism_weather_econometrics.ipynb
|-- src/
|   |-- __init__.py
|   `-- tourism_pipeline.py
|-- data/
|   `-- processed/
|-- requirements.txt
`-- README.md
```

## Interpretation

The model is designed to estimate conditional associations, not a complete
causal effect of weather on travel. Weather may also affect road access,
wildfire conditions, park operations, and visit-count quality. The notebook
discusses these limitations and frames the results as inputs to staffing,
visitor communication, and demand-monitoring decisions.
