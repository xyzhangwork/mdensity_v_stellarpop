# mdensity_v_stellarpop

This repository contains the scripts to conduct the analysis and generate the figures in Zhang et al. 2025 (arxiv: 2507.22602)

## Analysis Pipeline:
1. Cross-match between MaNGA and SGA catalogs: match_manga.py; 
	- _product: ./data/manga_sga_z.fits_
2.  Measure stellar mass and effective radii: nsa_v_sga.py; 
	- _product: ./data/sga_mass_new_28mag2.fits_
3. Construct sample mask: masks.py; 
	- _product: sample_mask_new2.fits_
4. obtain additional masking after visual inspection: masking.py
	- _product: ./data/masking.fits_
6. Collect kinematic data and stellar mass of the final sample: kinematics.py 
	- _product: ./data/parameters_reff2_2.fits_
7. split sample using different parameter spaces: sample_split.py 
	- _product: ./data/sample_split_outliers_total2.fits_
8.  Integrate spectra in 3 radial bins for each galaxy in the final sample: stack_gal.py 
	- _product: ./data/stacked_spec.fits_ 
9.  correct for telluric lines: sigma_unite.py (broaden the spectra to extract line features)+ fix_masks.py (compare with telluric line list and mask out pixels with large residuals around known telluric line features)
	- _product: ./data/stacked_sigfix_mask.fits_
10. smooth masked spectra to uniform resolution: smooth_spec.py
	- _product: stacked_smooth_sigma300_mask_new2.fits_
11. stack spectra for different sample-split methods and measure LICK indices (& uncertainties) and estimate measurement uncertainties for stacked spectra: uncertainty.py
	- _product: ./data/index/*; ./data/std*.fits_
12. measure LICK index values for sMILES SSPs 
	- _product: ./data/smiles_ssp/*_  
13. construct input spectra for alf fitting: make_spec.py  
	- _product: ./data/alf/*_
----------------------------------------------------------

### other scripts: 
- mclrs.py: collection of several Mass-to-Light-ratio-to-Color Relations 
- calc_kcor.py: k-correction code
- smiles_index.py: convolve sMILES SSP to data resolution and measure their LICK indices
	- _product_: 
		- _./data/smiles_ssp/Bimodal_2.8_ssp.csv_
		- _./data/smiles_ssp/Universal_Kroupa_ssp.csv_ 
- ./plot_scripts/read_alf.py: reading alf output data by Charlie Conroy, repository: [alf](https://github.com/cconroy20/alf)
- ./plot_scripts/ks-error.py: perform k-s test and estimate p value errors by bootstrapping
## Plotting Scripts
___path: ./plot_scripts___
- ./plot_scripts/plot_split.ipynb: visualize sample-split methods and the corresponding physical parameter distributions, __Figure 4 & 5__
- ./plot_scripts/plot_index.ipynb: plot radial profiles of LICK indices for different sample-split methods, __Figure 6 & 7__
- ./plot_scripts/simle_plot.ipynb: plot data index values against sMILES SSP grids, __Figure 8__
- ./plot_scripts/plot_alf_spectra.ipynb: compare data with output spectra from alf-fitting, __Figure 9__
- ./plot_scripts/plot_alf_corner.ipynb: generate corner plots for alf posteriors, __Figure 10__
- ./plot_scripts/plot_alf_gradient.ipynb: plot radial profiles of elemental abundances from alf fitting, __Figure 11 & 12 & 18 & 19__
- ./plot_scripts/plot_alf_gradient_imf.ipynb: compare alf results between Kroupa IMF and IMF-variable fitting, __Figure 13__
