# OceanEmbed v2 — Individual Dataset Download Scripts

Each script is **standalone** — just run it and it will:
1. Auto-install its pip dependency
2. Start downloading immediately

All scripts use the **Bay of Bengal** region by default:
- Longitude: 75°E – 100°E
- Latitude: 5°N – 25°N
- Time period: 2010-01-01 → 2021-12-31

---

## One-time Setup

### For CMEMS datasets (SST, SLA, Wind, Currents, GLORYS, Chl, Kd490)
```bash
pip install copernicusmarine
copernicusmarine login      # ← run this ONCE, it saves your credentials
```

### For NASA datasets (SSS, Precipitation)
```bash
pip install earthaccess
# Register at: https://urs.earthdata.nasa.gov/
# Add to ~/.netrc:
#   machine urs.earthdata.nasa.gov
#   login YOUR_USERNAME
#   password YOUR_PASSWORD
```

### For ERA5 + GloFAS datasets (Heat Flux, SLP, River Discharge)
```bash
pip install cdsapi
# Register at: https://cds.climate.copernicus.eu/
# Create ~/.cdsapirc:
#   url: https://cds.climate.copernicus.eu/api/v2
#   key: UID:API-KEY
```

---

## Scripts

| Script | Dataset | Source | Tier | Size (est.) |
|--------|---------|--------|------|-------------|
| `dl_sst.py` | Sea Surface Temperature | CMEMS OSTIA | 1 | ~5 GB |
| `dl_sla.py` | Sea Level Anomaly + ADT | CMEMS DUACS | 1 | ~3 GB |
| `dl_wind.py` | Wind U & V | CMEMS CERSAT | 1 | ~4 GB |
| `dl_heatflux.py` | Solar/LW/Latent/Sensible Heat Flux | ERA5 / CDS | 2 | ~8 GB |
| `dl_slp.py` | Sea Level Pressure | ERA5 / CDS | 2 | ~2 GB |
| `dl_currents.py` | Surface Currents U & V | CMEMS GlobCurrent | 3 | ~3 GB |
| `dl_glorys.py` | 3-D Temp + Salinity + MLD | CMEMS GLORYS12 | 3 | ~150 GB ⚠️ |
| `dl_sss.py` | Sea Surface Salinity | NASA SMAP | 3 | ~2 GB |
| `dl_precip.py` | Precipitation | NASA GPM IMERG | 2 | ~10 GB |
| `dl_chl.py` | Chlorophyll-a | CMEMS GlobColour | 4 | ~4 GB |
| `dl_kd490.py` | Diffuse Attenuation Kd490 | CMEMS GlobColour | 4 | ~4 GB |
| `dl_bathymetry.py` | Bathymetry (static) | GEBCO 2023 | 5 | ~7 GB |
| `dl_geothermal.py` | Geothermal Heat Flux (static) | IHFC | 5 | ~50 MB |
| `dl_glofas.py` | River Discharge | GloFAS / CDS | 5 | ~5 GB |

---

## Run order (priority from v2 plan)

```bash
# 🥇 Priority 1 — biggest accuracy gains
python3 scripts/download/dl_sss.py          # Sea Surface Salinity
python3 scripts/download/dl_glorys.py       # MLD + 3-D Temp + Salinity ⚠️ Large

# 🥈 Priority 2 — energy budget
python3 scripts/download/dl_heatflux.py    # Heat Flux
python3 scripts/download/dl_precip.py      # Precipitation

# 🥉 Priority 3 — eddies + biology
python3 scripts/download/dl_currents.py    # Ocean Currents
python3 scripts/download/dl_chl.py         # Chlorophyll-a
python3 scripts/download/dl_kd490.py       # Kd490

# Static (run once)
python3 scripts/download/dl_bathymetry.py  # GEBCO Bathymetry
python3 scripts/download/dl_geothermal.py  # Geothermal
python3 scripts/download/dl_glofas.py      # River Discharge

# Core (already in v1 but upgrading)
python3 scripts/download/dl_sst.py         # SST
python3 scripts/download/dl_sla.py         # SLA
python3 scripts/download/dl_wind.py        # Wind
python3 scripts/download/dl_slp.py         # Sea Level Pressure
```

---

## Data will be saved to

```
data/raw/
├── sst/          ← dl_sst.py
├── sla/          ← dl_sla.py
├── wind/         ← dl_wind.py
├── heatflux/     ← dl_heatflux.py
├── slp/          ← dl_slp.py
├── currents/     ← dl_currents.py
├── glorys/       ← dl_glorys.py  (3-D: thetao, so, mld)
├── sss/          ← dl_sss.py
├── precip/       ← dl_precip.py
├── chl/          ← dl_chl.py
├── kd490/        ← dl_kd490.py
├── bathymetry/   ← dl_bathymetry.py
├── geothermal/   ← dl_geothermal.py
└── glofas/       ← dl_glofas.py
```

After all downloads:
```bash
python3 scripts/preprocess.py --config configs/v2_bob_full.yaml
```
