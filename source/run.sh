#!/bin/sh
#
# Shell script
#
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# Date  : Oct  6, 2018
#################################### SOURCE START ###################################

python3 main_rff.py  > ../etc/output_main_rff.txt
python3 main_orf.py  > ../etc/output_main_orf.txt
# python3 main_ksvm.py > ../etc/output_main_ksvm.txt

#################################### SOURCE FINISH ##################################
# Ganerated by grasp version 0.0
