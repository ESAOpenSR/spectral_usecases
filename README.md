# Spectral Use Case Validation Repo

This repository accompanies the paper on spectrally faithful super-resolution (SR) for remote-sensing imagery. It provides the supplementary material, including code and ready-to-share maps hosted on the project website (`usecases.opensr.eu`). The experiments reproduce the Valencia 2024 flood and the 2025 Palisades wildfire case studies, highlighting how physics-constrained latent diffusion SR improves spectral indices for water and burn-severity detection.

## Getting started
1. **Install dependencies**: create a virtual environment and install requirements.
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Fetch data**: download the prepared Sentinel-2 stacks (includes native, interpolated, and SR products) via the helper script, which populates `data_fire/` and `data_flood/` under each workflow directory.
   ```bash
   ./fetch_data.sh
   ```
   Update `ZIP_URL` in the script to point at your storage location if you host the archive elsewhere.

## Running the workflows
The workflows are organized by hazard; each step is a numbered script. Run them from their directory after data download.

### Flood (Valencia 2024)
```bash
cd flood_workflow
python 01_after_create_SR_flood.py      # visualize SR vs baseline inputs
python 02_create_flood_mask.py          # compute spectral mask
python 03_calc_mndwi.py                 # derive MNDWI index
python 04_create_thresh_mask.py         # apply thresholds
python 05_histograms_and_mndwi_thresh.py# explore histogram-based cuts
python 06_compute_metrics.py            # evaluate detection metrics
```

### Fire (Palisades 2025)
```bash
cd fire_workflow
python 01a_before_create_SR_fire.py     # visualize native inputs
python 01b_after_create_SR_fire.py      # visualize SR outputs
python 02_create_fire_mask.py           # compute spectral mask
python 03_calc_dnbr.py                  # derive dNBR index
python 04_create_thresh_mask.py         # apply thresholds
python 05_histograms_and_dnbr_thresh.py # explore histogram-based cuts
python 06_compute_metrics.py            # evaluate detection metrics
```

## Notes
- Both workflows assume GPU access for LDSR-S2 super-resolution (`torch.cuda.is_available()` must be true).
- The `resources/` folder stores ancillary assets used by the notebooks and visualizations.
- Replace paths or thresholds in the scripts as needed to test additional events or alternative SR baselines.
