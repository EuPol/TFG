#!/bin/bash

source "${HOME}/miniconda3/etc/profile.d/conda.sh" activate &&  
conda activate 'comparativa' # cambiar por nombre del entorno


mkdir -p 'FaceCOX/GMM'
mkdir -p 'FaceCOX/CBL'
mkdir -p 'FaceCOX/NNO'

# Test with FaceCOX

python3 main_custom.py GMM 0 FaceCOX
python3 main_custom.py GMM 1 FaceCOX
python3 main_custom.py GMM 2 FaceCOX

python3 main_custom.py CBL 0 FaceCOX
python3 main_custom.py CBL 1 FaceCOX
python3 main_custom.py CBL 2 FaceCOX

python3 main_custom.py NNO 0 FaceCOX
python3 main_custom.py NNO 1 FaceCOX
python3 main_custom.py NNO 2 FaceCOX

