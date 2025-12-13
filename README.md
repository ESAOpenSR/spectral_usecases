<p align="center">
  <a href="https://usecases.opensr.eu" target="_blank" rel="noreferrer" style="text-decoration:none;">
    <button style="padding: 10px 18px; font-size: 16px; font-weight: 600; color: white; background-color: #1f6feb; border: none; border-radius: 6px; cursor: pointer;">Interactive Maps</button>
  </a>
</p>



# Spectral Use Case Validation - Flood 🌊 and Fire 🔥

<p align="center">
  🌍 <strong><a href="https://usecases.opensr.eu">Explore the Interactive Maps</a></strong> 🛰️
</p

This repository supports the paper’s LR vs. SR spectral use case validation, comparing native Sentinel-2 imagery against physics-constrained latent diffusion SR outputs for water (flood) and burn-severity (fire) detection. It provides reproducible code, data download helpers, and links to the accompanying walkthroughs on the project site [usecases.opensr.eu](https://usecases.opensr.eu).


<p align="center">
  <img
    src="https://raw.githubusercontent.com/ESAOpenSR/spectral_usecases/15370fdf93cd2965b76abd2bec848cede6da156a/resources/flood_map_example.png"
    width="600"
    alt="Flood map example"
  />
</p>



## Run the LR vs SR validations locally
The steps below mirror the instructions shown in the notebooks; follow them once to enable both workflows.

1. **Install dependencies**: create a virtual environment and install requirements.
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Fetch data**: download the prepared Sentinel-2 stacks (native, interpolated, and SR products) with the helper script. It populates `data_fire/` and `data_flood/` under each workflow directory.
   ```bash
   ./fetch_data.sh
   ```
   Update `ZIP_URL` in the script to point at your storage location if you host the archive elsewhere.
3. **Open the walkthroughs**: the rendered notebooks are available on the project site at https://usecases.opensr.eu/notebooks; they reference the same commands below for local execution.

## Running the workflows
The workflows are organized by hazard; each step is a numbered script. Run them from their directory after data download.

### Flood (Valencia 2024)
```bash
cd flood_workflow
python 01_after_create_SR_flood.py       # visualize SR vs baseline inputs
python 02_create_flood_mask.py           # compute spectral mask
python 03_calc_mndwi.py                  # derive MNDWI index
python 04_create_thresh_mask.py          # apply thresholds
python 05_histograms_and_mndwi_thresh.py # explore histogram-based cuts
python 06_compute_metrics.py             # evaluate detection metrics
```

### Fire (Palisades 2025)
```bash
cd fire_workflow
python 01a_before_create_SR_fire.py      # visualize native inputs
python 01b_after_create_SR_fire.py       # visualize SR outputs
python 02_create_fire_mask.py            # compute spectral mask
python 03_calc_dnbr.py                   # derive dNBR index
python 04_create_thresh_mask.py          # apply thresholds
python 05_histograms_and_dnbr_thresh.py  # explore histogram-based cuts
python 06_compute_metrics.py             # evaluate detection metrics
```

## Notes
- Both workflows assume GPU access for LDSR-S2 super-resolution (`torch.cuda.is_available()` must be true).
- Replace paths or thresholds in the scripts as needed to test additional events or alternative SR baselines.
