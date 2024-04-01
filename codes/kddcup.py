#!/usr/bin/env python3

"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Plot deviations between 1995 and 1996 for the veterans org's direct mailing.

This script offers command-line options "--interactive" and "--no-interactive"
for plotting interactively and non-interactively, respectively. Interactive is
the default. The interactive setting does the same as the non-interactive, but
without saving to disk any plots, and without plotting any classical
reliability diagrams or any scatterplots of the covariates used as controls.
When run non-interactively (i.e., with command-line option "--no-iteractive"),
this script creates a directory, "unweighted", in the working directory if the
directory does not already exist, then creates five subdirectories there, named
"02", "12", "20", "21", and "012", corresponding to controlling for covariates
with the corresponding numbers in the specified order, where "0" labels the
covariate for the normalized age of the recipient, "1" labels the normalized
fraction married in the Census block where the recipient lives, and "2" labels
the normalized average household income in the Census block where the recipient
lives. These three covariates are all integer-valued in the data set --
the fraction married is given by the nearest integer-valued percentage.
Subdirectories "02", "12", "20", and "21" get filled with eight files, while
"012" gets filled with six (the first six in the following list):
1. metrics.txt -- metrics about the plots
2. cumulative.pdf -- plot of cumulative differences
3. equiscores10.pdf -- reliability diagram with 10 bins, equispaced in scores
4. equiscores100.pdf -- reliability diagram with 100 bins, equispaced in scores
5. equierrs10.pdf -- reliability diagram with 10 bins, where the error bar is
                     about the same for every bin
6. equierrs100.pdf -- reliability diagram with 100 bins, where the error bar is
                      about the same for every bin
7. covars.pdf -- PDF scatterplot of the covariates used as controls;
                 shading corresponds to arc length along the Hilbert curve
8. covars.jpg -- compressed scatterplot of the covariates used as controls;
                 shading corresponds to arc length along the Hilbert curve
The script also creates 12 files in the directory, "unweighted", namely
"01.pdf", "01.jpg", "02.pdf", "02.jpg", "10.pdf", "10.jpg",
"12.pdf", "12.jpg", "20.pdf", "20.jpg", "21.pdf", "21.jpg".
These files scatterplot the associated covariates against each other
(in the order given by the name of the file).
The data comes from a direct mailing campaign of a national veterans org.

This source code is licensed under the MIT license found in the LICENSE file in
the root directory of this source tree.
"""

import argparse
import math
import numpy as np
import os
import subprocess
from numpy.random import default_rng

import matplotlib
from matplotlib import get_backend
default_backend = get_backend()
matplotlib.use('agg')
import matplotlib.pyplot as plt

from hilbertcurve.hilbertcurve import HilbertCurve
import paired_weighted


# Specify whether to randomize the scores in order to ensure their uniqueness.
RANDOMIZE = True

# Specify the name of the file of comma-separated values
# for the training data about the direct-mail marketing campaign.
filename = 'cup98lrn.txt'

# Parse the command-line arguments (if any).
parser = argparse.ArgumentParser()
parser.add_argument('--interactive', dest='interactive', action='store_true')
parser.add_argument(
    '--no-interactive', dest='interactive', action='store_false')
parser.add_argument(
    '--non-interactive', dest='interactive', action='store_false')
parser.set_defaults(interactive=True)
clargs = parser.parse_args()

# Make matplotlib interactive if clargs.interactive is True.
if clargs.interactive:
    plt.switch_backend(default_backend)

# Count the number of lines in the file for filename.
lines = 0
with open(filename, 'r') as f:
    for line in f:
        lines += 1
print(f'reading and filtering all {lines} lines from {filename}....')

# Determine the number of columns in the file for filename.
with open(filename, 'r') as f:
    line = f.readline()
    num_cols = line.count(',') + 1

# Read and store the data in the file named filename.
with open(filename, 'r') as f:
    for line_num, line in enumerate(f):
        # Eliminate the newline character at the end of line with line[:-1].
        parsed = line[:-1].split(',')
        if line_num == 0:
            # The initial line is a header ... save its column labels.
            header = parsed.copy()
        else:
            # All but the initial line consist of data ... save the data.
            data = []
            for s in parsed:
                if s == '':
                    # Label missing data with None.
                    data.append(None)
                elif s.isdecimal():
                    # Convert a whole number to an int.
                    data.append(int(s))
                else:
                    try:
                        # Convert a floating-point number to a float.
                        data.append(float(s))
                    except ValueError:
                        if(s[-1] == '-'):
                            # The zipcode includes an extra hyphen ... omit it.
                            data.append(int(s[:-1]))
                        else:
                            # Save the entire string.
                            data.append(s)
            # Initialize raw if line_num == 1.
            if line_num == 1:
                raw = []
            # Discard entries whose ages, marital fractions,
            # or average household incomes are missing.
            if data[header.index('AGE')] not in [None, 0]:
                if data[header.index('MARR1')] not in [None, 0]:
                    if data[header.index('IC3')] not in [None, 0]:
                        raw.append(data)

# Rename especially interesting columns with easier-to-understand phrases.
header[header.index('AGE')] = 'normalized age'
header[header.index('MARR1')] = 'normalized fraction married'
header[header.index('IC3')] = 'normalized average household income'

# Tabulate the total numbers of mailings of each type.
count = [0, 0]
for k in range(2, 25):
    count.append(0)
    for rawent in raw:
        if rawent[header.index('ADATE_' + str(k))] is not None:
            count[k] += 1

# Retain only those who received mailings in both 1995 and 1996 and donated.
newraw = []
for rawent in raw:
    in1996 = False
    for k in range(3, 13):
        if rawent[header.index('ADATE_' + str(k))] is not None:
            in1996 = True
    in1995 = False
    for k in range(13, 23):
        if rawent[header.index('ADATE_' + str(k))] is not None:
            in1995 = True
    if in1995 and in1996:
        # Total the contributions made in 1995 and 1996.
        ss = 0
        for k in range(3, 23):
            if rawent[header.index('RAMNT_' + str(k))] is not None:
                ss += rawent[header.index('RAMNT_' + str(k))]
        if ss > 0:
            newraw.append(rawent)
raw = newraw
print()
print(f'mailed both in 1995 and in 1996 and donated = len(raw) = {len(raw)}')

# Set up the random number generator.
rng = default_rng(seed=543216789)

# Tabulate all covariates of possible interest.
vars = [
    'normalized age', 'normalized fraction married',
    'normalized average household income']
covars = np.zeros((len(raw), len(vars)))
for k in range(len(raw)):
    for j in range(len(vars)):
        covars[k, j] = raw[k][header.index(vars[j])]
        if vars[j] == 'normalized average household income':
            covars[k, j] *= 100
            if RANDOMIZE:
                covars[k, j] *= 1 + 1e-8 * rng.standard_normal()
        if vars[j] == 'normalized fraction married':
            covars[k, j] /= 100

# Store processes for converting from pdf to jpeg in procs.
procs = []

# Normalize every covariate, saving the unnormalized versions in covs.
covs = covars.copy()
cmin = np.min(covars, axis=0)
cmax = np.max(covars, axis=0)
covars = (covars - cmin) / (cmax - cmin)

if not clargs.interactive:
    # Create a directory, "unweighted", if none exists.
    dir = 'unweighted'
    try:
        os.mkdir(dir)
    except FileExistsError:
        pass
    dir += '/'
    # Scatterplot pairs of covariates.
    for j in range(covars.shape[1]):
        for k in list(range(0, j)) + list(range(j + 1, covars.shape[1])):
            plt.figure()
            plt.scatter(covars[:, j], covars[:, k], s=.1, c='k')
            plt.xlabel(vars[j])
            plt.ylabel(vars[k])
            plt.tight_layout()
            # Save the figure to disk and queue up a process for converting
            # from pdf to jpg.
            filename = dir + str(j) + str(k)
            filepdf = filename + '.pdf'
            filejpg = filename + '.jpg'
            plt.savefig(filepdf, bbox_inches='tight')
            plt.close()
            args = ['convert', '-density', '600', filepdf, filejpg]
            procs.append(subprocess.Popen(args))

# Set the number of bits in the discretization (mantissa).
precision = 64
# Determine the data type from precision.
if precision == 8:
    dtype = np.uint8
elif precision == 16:
    dtype = np.uint16
elif precision == 32:
    dtype = np.uint32
elif precision == 64:
    dtype = np.uint64
else:
    raise TypeError(f'There is no support for precision = {precision}.')

# Specify tuples of covariates for controls.
controls = [(0, 2), (2, 0), (1, 2), (2, 1), (0, 1, 2)]

# Construct plots for each set of controls.
for control in controls:
    p = len(control)
    t = covars[:, control]
    # Round the inputs.
    it = np.rint((2**precision - 1) * t.astype(np.longdouble)).astype(
        dtype=dtype)
    # Perform the Hilbert mapping from p dimensions to one dimension.
    hc = HilbertCurve(precision, p)
    ints = hc.distances_from_points(it)
    if RANDOMIZE:
        assert np.unique(ints).size == it.shape[0]
    # Sort according to the scores.
    perm = np.argsort(ints)
    t = t[perm, :]
    u = covs[:, control]
    u = u[perm, :]
    # Construct scores for plotting.
    imin = np.min(ints)
    imax = np.max(ints)
    s = (np.sort(ints) - imin) / (imax - imin)
    if RANDOMIZE:
        # Ensure uniqueness even after roundoff errors.
        eps = np.finfo(np.float64).eps
        s = s + np.arange(0, s.size * eps, eps)
        s = s.astype(np.float64)

    # Compare 1995 and 1996.
    r0 = []
    r1 = []
    for k in range(len(raw)):
        rawent = raw[perm[k]]
        # Total the contributions made in 1996.
        ss = 0
        for j in range(3, 13):
            if rawent[header.index('RAMNT_' + str(j))] is not None:
                ss += rawent[header.index('RAMNT_' + str(j))]
        r0.append(ss)
        # Total the contributions made in 1995.
        ss = 0
        for j in range(13, 23):
            if rawent[header.index('RAMNT_' + str(j))] is not None:
                ss += rawent[header.index('RAMNT_' + str(j))]
        r1.append(ss)
    # Normalize both 1995 and 1996 such that their means are 1.
    r0 = [entry * len(r0) / sum(r0) for entry in r0]
    r1 = [entry * len(r1) / sum(r1) for entry in r1]
    if not clargs.interactive:
        # Set a directory for the controls.
        dir = 'unweighted'
        try:
            os.mkdir(dir)
        except FileExistsError:
            pass
        dir += '/'
        for k in control:
            dir += str(k)
        try:
            os.mkdir(dir)
        except FileExistsError:
            pass
        dir += '/'
        print(f'./{dir} is under construction....')
    if len(control) == 2 and not clargs.interactive:
        # Plot the covariates.
        plt.figure()
        plt.xlabel(vars[control[0]])
        plt.ylabel(vars[control[1]])
        gray = .8 * np.vstack((np.array(s), np.array(s), np.array(s))).T
        plt.scatter(
            covars[perm, control[0]], covars[perm, control[1]],
            s=2, c=gray, marker='o', linewidths=0)
        plt.title('covariates')
        # Save the figure to disk and queue up a process for converting
        # from pdf to jpg.
        filename = dir + 'covars'
        filepdf = filename + '.pdf'
        filejpg = filename + '.jpg'
        plt.savefig(filepdf, bbox_inches='tight')
        plt.close()
        args = ['convert', '-density', '600', filepdf, filejpg]
        procs.append(subprocess.Popen(args))
    if clargs.interactive:
        # Plot the cumulative differences interactively.
        majorticks = 5
        minorticks = 300
        covariates = []
        for j in range(len(control)):
            offset = len('normalized ')
            covariates.append(vars[control[j]][offset:])
        window = 'Donations made in 1995 versus in 1996'
        window += ' (click the plot to continue)'
        paired_weighted.icumulative(
            r1, r0, s, t, u, covariates, majorticks, minorticks, window=window)
    else:
        # Plot reliability diagrams and the cumulative graph.
        filename = dir + 'cumulative.pdf'
        majorticks = 10
        minorticks = 300
        kuiper, kolmogorov_smirnov, lenscale = paired_weighted.cumulative(
            r1, r0, s, majorticks, minorticks, filename)
        filename = dir + 'metrics.txt'
        with open(filename, 'w') as f:
            if not RANDOMIZE:
                f.write('len(np.unique(s)):\n')
                f.write(f'{len(np.unique(s))}\n')
            f.write('len(s):\n')
            f.write(f'{len(s)}\n')
            f.write('lenscale:\n')
            f.write(f'{lenscale}\n')
            f.write('Kuiper:\n')
            f.write(f'{kuiper:.4}\n')
            f.write('Kolmogorov-Smirnov:\n')
            f.write(f'{kolmogorov_smirnov:.4}\n')
            f.write('Kuiper / lenscale:\n')
            f.write(f'{(kuiper / lenscale):.4}\n')
            f.write('Kolmogorov-Smirnov / lenscale:\n')
            f.write(f'{(kolmogorov_smirnov / lenscale):.4}\n')
        nin = [10, 100]
        for nbins in nin:
            filename = dir + 'equiscores' + str(nbins) + '.pdf'
            paired_weighted.equiscores(r1, r0, s, nbins, filename)
            filename = dir + 'equierrs' + str(nbins) + '.pdf'
            paired_weighted.equierrs(r1, r0, s, nbins, filename)
if not clargs.interactive:
    print()
    print('waiting for conversion from pdf to jpg to finish....')
    for iproc, proc in enumerate(procs):
        proc.wait()
        print(f'{iproc + 1} of {len(procs)} conversions are done....')
