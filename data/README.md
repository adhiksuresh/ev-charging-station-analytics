# Data

## `processed/` (included)

| File | Contents |
|---|---|
| `daily_energy_march2023.csv` | Total kWh per day, 1–31 March 2023, for Stations 1–4 |
| `hour_of_day_energy_march2023.csv` | Total kWh for each hour of the day (0–23), summed over March 2023, for Stations 1–4 |

## `raw/` (not included)

Raw exports from the 5G smart meters at the IIT Hyderabad EV charging stations:

| Station | Meter ID | File expected at |
|---|---|---|
| 1 | IHM10050174 | `data/raw/IHM10050174.csv` |
| 2 | IHM10050175 | `data/raw/IHM10050175.csv` |
| 3 | IHM10050178 | `data/raw/IHM10050178.csv` |
| 4 | IHM10050180 | `data/raw/IHM10050180.csv` (not yet available) |

Format: semicolon-separated, one row per register reading, with columns `Reading values`, `Profile`, `Reading time`, `OBIS code`, `Unit`, and others. The pipeline uses the `InstantaneousProfile` rows with OBIS code `1-0:1.8.0*255` (cumulative imported energy, kWh).

The raw data belongs to the Smart Grid Lab, IIT Hyderabad, and is available on reasonable request.
