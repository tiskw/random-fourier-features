#!/bin/sh
#
# Shell script
#
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# Date  : Oct 20, 2018
#################################### SOURCE START ###################################

### Regression with RFF
python3 main_reg_rff.py

### MNIST classification with RFF and ORF (uncomment the redirection if you wait)
python3 main_svm_kernel.py    # > ../etc/output_main_ksvm.txt
python3 main_svm_rff.py       # > ../etc/output_main_rff.txt
python3 main_svm_orf.py       # > ../etc/output_main_orf.txt
python3 main_svm_rff_batch.py # > ../etc/output_main_rff_batch.txt

#################################### SOURCE FINISH ##################################
# Ganerated by grasp version 0.0
