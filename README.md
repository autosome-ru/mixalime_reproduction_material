# MIXALIME: Reproduction material

Scripts that were used to produce simulation studies and various numerical experiments presented in the MIXALIME paper.

## Requirements
To run most scripts, it should suffice installing all the packages that are imported in the beginning of the script. However, to run benchmark, you should also install MIXALIME as it is a standalone tool not used in imports directly. To this end, makre sure to install MIXALIME v. 2.23.3 (although it is not the latest version as of today, it is the version that was used to perform simulation studies):
```
pip3 install mixalime==2.23.3
```

## Supplementary Methods 1

+ Python script for plotting heatmap and density plots can be found at `slices/main.py`;
+ Python script for plotting 3D surface of the reparametrization function can be found at `r_reparam/main.py`;
+ Python scripts for plotting images at error surfaces of the incomplete beta function and their corresponding animated versions can be found at `error_surfaces/plot_pb.py` and at `error_surfaces/plot_ab.py` for the first and the second image respectively;
+ Python scripts for plotting error surfaces of the 3F2 hypergeometric functions can be found at `error_surfaces/plot_hyp_px.py`,  `error_surfaces/plot_hyp_rx.py` and `error_surfaces/plot_hyp_kx.py` for the first, the second image and the third image respectively;
+ Python script that reproduces results showcased at the "On the linearity of the reference bias* appendix can be found at `circle_fit/main.py`;

## Supplementary Methods 2
To reproduce simulation studies, run the following scripts in a consequitive order:

1. `benchmark/run_benchmark.py` - to generate synthetic datasets and run MIXALIME models;
2. `benchmark/compute_metrics.py` - to compute performance metrics and their standard deviations across repetitions for each dataset;
3. `benchmark/build_plots_and_tables.py` - to draw plots and build tables.

Then, tables in `.tex` format as well as figures with evaluated performance metrics can be found in the `results/benchmark_tabulars.tex`, `results/benchmark_figures.tex` and `results/benchmark_figures_full.tex` respectively.
