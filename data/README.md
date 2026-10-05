# NOAA GHCN-Hourly (GHCNh) Data Ingestion Directory

This directory is the designated landing folder for raw NOAA Global Historical Climatology Network - Hourly (GHCNh) data files.

**IMPORTANT:** Keep this directory empty of actual data files in the repository. Drop your uncompressed or gzip-compressed NOAA GHCNh files here to run the ingestion pipeline.

---

## 1. Expected File Naming Convention

NOAA NCEI distributes GHCNh files in Pipe-Separated Values (PSV) or Parquet format with the following naming patterns:

### Primary Expected Pattern (Single Year per Station)
```
GHCNh_<Station_ID>_<Year>.psv
```
*Example:* `GHCNh_USW00093037_2023.psv` (Station ID: `USW00093037`, Year: `2023`)

### Period of Record (POR) Pattern
```
GHCNh_<Station_ID>_por.psv
```
*Example:* `GHCNh_USW00023188_por.psv`

### Compressed Files
The ingestion engine also transparently reads gzip-compressed archives:
```
GHCNh_<Station_ID>_<Year>.psv.gz
GHCNh_<Station_ID>_por.psv.gz
```

*Note on Station IDs:*
Station IDs in GHCNh follow the standard 11-character GHCN convention:
- First 2 characters: Country code (e.g., `US` for United States, `MX` for Mexico, `CA` for Canada)
- 3rd character: Network type (`W` = WBAN/NWS, `C` = CoCoRaHS, `M` = WMO/Synoptic, `R` = RAWS)
- Remaining 8 characters: Station identification number (e.g., `00093037`)

---

## 2. Published NOAA GHCNh PSV File Schema

GHCNh files are pipe-delimited (`|`) text files with a header line as the very first row.

### Header Line (Exact NOAA Column Layout)
```psv
Station_ID|Station_name|Latitude|Longitude|Elevation|Year|Month|Day|Hour|Minute|temperature|dew_point_temperature|relative_humidity|wind_direction|wind_speed|wind_gust|station_level_pressure|sea_level_pressure|precipitation|precipitation_1_hour|precipitation_3_hour|precipitation_6_hour|precipitation_24_hour|present_weather|quality_flag
```

*(Note: The ingestion pipeline also accepts uppercase variants e.g. `STATION_ID|LATITUDE|LONGITUDE` or `DATE` timestamps).*

### Column Definitions and Units

| Column Name | Type | Unit / Format | Description & Valid Ranges | Missing Value Sentinels |
|---|---|---|---|---|
| `Station_ID` | String | 11 alphanumeric | Station identifier (e.g., `USW00093037`) | None (required) |
| `Station_name` | String | UTF-8 text | Geographic / airport name of station | None |
| `Latitude` | Float | Decimal degrees | Station latitude (-90.0000 to +90.0000) | `NaN`, `-999.9` |
| `Longitude` | Float | Decimal degrees | Station longitude (-180.0000 to +180.0000) | `NaN`, `-999.9` |
| `Elevation` | Float | Meters (m) | Station elevation above mean sea level | `-999.9`, `9999` |
| `Year` | Integer | YYYY | Observation year (e.g., 2023) | Required |
| `Month` | Integer | 1–12 | Observation month | Required |
| `Day` | Integer | 1–31 | Observation day of month | Required |
| `Hour` | Integer | 0–23 | Observation hour in UTC | Required |
| `Minute` | Integer | 0–59 | Observation minute in UTC | Required (often 00 or 50–55) |
| `temperature` | Float | °C | Dry bulb air temperature | `-999.9`, `-9999`, `NA` |
| `dew_point_temperature`| Float | °C | Dew point temperature | `-999.9`, `-9999`, `NA` |
| `relative_humidity` | Float | % (0–100) | Relative humidity | `-999.9`, `999.0` |
| `wind_direction` | Integer | Degrees (0–360) | Wind direction (0=Calm, 360=North) | `999`, `-999` |
| `wind_speed` | Float | m/s | Wind speed | `-999.9`, `999.9` |
| `wind_gust` | Float | m/s | Peak gust speed | `-999.9`, `999.9`, `NA` |
| `station_level_pressure` | Float | hPa / mbar | Local atmospheric pressure at station elevation | `-9999.9`, `9999.9` |
| `sea_level_pressure` | Float | hPa / mbar | Barometric pressure adjusted to sea level | `-9999.9`, `9999.9` |
| `precipitation` | Float | mm | Reported precipitation accumulation | `-999.9`, `999.9` |
| `precipitation_1_hour`| Float | mm | 1-hour liquid precipitation accumulation | `-999.9`, `NA` |
| `precipitation_3_hour`| Float | mm | 3-hour precipitation accumulation | `-999.9`, `NA` |
| `precipitation_6_hour`| Float | mm | 6-hour precipitation accumulation | `-999.9`, `NA` |
| `precipitation_24_hour`| Float | mm | 24-hour precipitation accumulation | `-999.9`, `NA` |
| `present_weather` | String | WMO 4677 code | Present weather code (e.g. RA, TS, DZ, HZ) | `""`, `None` |
| `quality_flag` | String | Character code | NOAA quality control flag (`V`=Passed, `M`=Suspect, `Q`=Failed) | Empty |

---

## 3. Example Sample Record (Pipe-Separated)

```psv
Station_ID|Station_name|Latitude|Longitude|Elevation|Year|Month|Day|Hour|Minute|temperature|dew_point_temperature|relative_humidity|wind_direction|wind_speed|wind_gust|station_level_pressure|sea_level_pressure|precipitation|precipitation_1_hour|precipitation_3_hour|precipitation_6_hour|precipitation_24_hour|present_weather|quality_flag
USW00093037|ALBUQUERQUE INTL AP|35.0433|-106.6128|1619.1|2023|08|15|14|00|26.7|12.2|40|180|4.1|7.2|841.2|1012.4|0.0|0.0|0.0|0.0|0.0||V
USW00093037|ALBUQUERQUE INTL AP|35.0433|-106.6128|1619.1|2023|08|15|15|00|28.1|13.0|39|190|5.5|8.8|840.1|1011.0|0.0|0.0|0.0|0.0|0.0||V
USW00093037|ALBUQUERQUE INTL AP|35.0433|-106.6128|1619.1|2023|08|15|16|00|22.4|18.5|78|240|12.8|19.5|836.4|1006.2|34.8|34.8|34.8|34.8|34.8|TSRA|V
```

---

## 4. Ingestion Engine Behavior

When you run the ingestion pipeline (`src/ingestion/ghcnh_parser.py` or `run_pipeline.py`):
1. **Directory Scan:** It scans `data/` for all `GHCNh_*.psv` and `GHCNh_*.psv.gz` files.
2. **Fallback / Mock Mode:** If `data/` contains no external files, the pipeline automatically detects this and generates verified, synthetic GHCNh records matching this exact NOAA schema so tests and downstream models can run end-to-end without breaking.
3. **Cleaning & Deduplication (FR1.2):**
   - Observations within the same hour are cleaned (e.g. synoptic vs METAR specials).
   - Missing value sentinels (`-999.9`, `-9999`, `999.9`, `9999`) are normalized to `None`/`NaN`.
   - Quality control flags (`Q`, `X`) are rejected; verified observations (`V` or blank standard) are preserved.
   - Interpolates minor missing values up to 2 hours while preserving gaps larger than 2 hours.
