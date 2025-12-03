# [ACM TOPML under review] Physical Simulator-Based Neural Networks for Real-Time Fouling Tomography
This repository implements our fouling localization method.

## Folder Structure
<pre>
.
└── empirical_data_evaluation/
    ├── data/
    │   └── model_name.pth 
    │   └── clean_pipe_measurement.tsv <ins>(will be available upon request)</ins>  
    │   └── fouled_pipe_measurement.tsv <ins>(will be available upon request)</ins>          
    ├── outputs/
    ├── define_model.py
    ├── fouling_localization.py


└── nn_training/
    ├── dataset/
    ├── training_output/
    │    ├── models/
    │    └── results/
    ├── define_model.py
    ├── test_model.py
    ├── train_model.py
    ├── utility.py


└── simulate_data/
    ├── B_Dx_Dy/
    ├── dataset/
    ├── create_B_Dx_Dy.py
    ├── create_maps.py
    ├── create_ratios.py
    ├── tool.py
    ├── utility.py


└── stan_code/
    ├── dataset/
    ├── results/
    ├── gp_inversion.py


└── uncertainty_estimate/
    ├── dataset/
    ├── training_output/
    │    ├── models/
    │    └── results/
    ├── define_model.py
    ├── test_uq_model.py
    ├── uq_plot.py
    ├── uq_training.py
</pre>

## Citation
<pre>
Here will be updated upon publication.  
</pre>
