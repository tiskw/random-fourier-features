#!/usr/bin/env python3
#
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# Date  : September 04, 2020
##################################################### SOURCE START #####################################################

import gzip
import os
import subprocess
import numpy as np

### List of download targets (key = ID string on GDrive, value = filename).
TARGETS = {
    "1bKOfH1tvMSY05DoMpT42abHIAurFKAVS": "7259_2000-2019",
}

### Parse line. This function assume that the first element of the given line
### is 'Date' which is replresentable as np.datetime64, and others are integer.
def parse(line):

    tokens = line.strip().split(",")
    return [np.datetime64(tokens[0])] + list(map(int, tokens[1:]))


def download_and_save_as_npz(id_gdrive, basename):

    filename_npz = basename + ".npz"
    filename_csv = basename + ".csv.gz"

    if os.path.exists(filename_npz):
        print("File '%s' already exists. Skip it." % filename_npz)
        return

    if not os.path.exists(filename_csv):
        command = "wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=%s' -O %s" % (id_gdrive, filename_csv)
        subprocess.run(command, shell = True)

    ### Read downloaded zipped csv file.
    lines = [line.decode().strip() for line in gzip.open(filename_csv)]

    ### Parse 1st line (header line) of the csv file.
    headers = [token.strip() for token in lines[0].split(",")]

    ### Read values and store to dictionaly.
    dataset = {header:[] for header in headers}
    for line in lines[1:]:
        for header, value in zip(headers, parse(line)):
            dataset[header].append(value)

    ### Convert list to np.array.
    for header in dataset:
        dataset[header] = np.array(dataset[header])

    ### Save arrays.
    np.savez(filename_npz, **dataset)


if __name__ == "__main__":

    ### Download all stock price files (.csv.gz) and convert it to .npy file.
    for id_gdrive, filename_npy in TARGETS.items():
        download_and_save_as_npz(id_gdrive, filename_npy)

##################################################### SOURCE FINISH ####################################################
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
