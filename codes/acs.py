#!/usr/bin/env python3

"""
Plot paired differences for the American Community Survey of the Census Bureau.

Copyright (c) Meta Platforms, Inc. and affiliates.

This script offers command-line options "--interactive" and "--no-interactive"
for plotting interactively and non-interactively, respectively. Interactive is
the default. The interactive setting does the same as the non-interactive, but
without saving to disk any plots, and without plotting any classical
reliability diagrams or any scatterplots of the covariates used as controls.
When run non-interactively (i.e., with command-line option "--no-interactive"),
this script creates a directory, "weighted", in the working directory if the
directory does not already exist, then creates a subdirectory there for one
of the four supported combinations of covariates used for conditioning: "NOC",
"NP", "NOC+NP", or "NP+NOC", where "NOC" refers to "number of own children",
"NP" refers to "number of people in the household", "NOC+NP" refers to both
"number of own children" & "number of people in the household" (in that order),
and "NP+NOC" refers to "number of people in the household" & "number of own
children" (in that order). In all cases, there is also an additional covariate
appended, namely the logarithm (base 10) of the adjusted household personal
income. The command line flag "--var" specifies which of the four possibilities
to use, defaulting to "NOC" if unspecified. The script fills each subdirectory
of the main directory, "weighted", with subdirectories corresponding to several
counties in California. The script fills each of these subdirectories (one for
each county) with 8 or 10 files (only 8 for "NOC+NP" or "NP+NOC"), with all
differences considered pertaining to the presence of a smartphone versus the
presence of a laptop or desktop computer (the difference is 0 if both are
present or both are absent, 1 if a smartphone is present but there is no laptop
or desktop computer, and -1 if there is no smartphone but a laptop or desktop
computer is present):
1. metrics.txt -- metrics about the plots
2. cumulative.pdf -- plot of cumulative differences
3. equiscores10.pdf -- reliability diagram with 10 bins (equispaced in scores)
4. equiscores20.pdf -- reliability diagram with 20 bins (equispaced in scores)
5. equiscores100.pdf -- reliability diagram with 100 bins (equispaced in score)
6. equierrs10.pdf -- reliability diagram with 10 bins (the error bar is about
                     the same for every bin)
7. equierrs20.pdf -- reliability diagram with 20 bins (the error bar is about
                     the same for every bin)
8. equierrs100.pdf -- reliability diagram with 100 bins (the error bar is about
                      the same for every bin)
9. covars.pdf -- PDF scatterplot of the covariates used as controls;
                 shading corresponds to the arc length along the Hilbert curve
10. covars.jpg -- compressed scatterplot of the covariates used as controls;
                  shading corresponds to the arc length along the Hilbert curve
The data comes from the American Community Survey of the U.S. Census Bureau,
specifically the household data from the counties in the state of California.
The results/responses are given by the variates specified in the list "comp"
defined below (together with the value of the variate to be considered
"success" in the sense of Bernoulli trials). The first response variate is an
indicator (either 1 or 0) of the presence of a smartphone in the household. The
second response variate is an indicator (either 1 or 0) of the presence of a
laptop or desktop computer in the household.

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
RANDOMIZE = False

# Specify which counties and variates to process, as well as the coded value
# of interest for each variate (or None if the values of interest are
# nonnegative integer counts). "LAPTOP" refers to a laptop or a desktop.
counties = [
    'Napa', 'Santa Cruz', 'San Mateo', 'Santa Clara', 'San Diego', 'Orange',
    'Los Angeles'
]
comp = [
    {'var': 'SMARTPHONE', 'val': 1},
    {'var': 'LAPTOP', 'val': 1},
]

# Specify the name of the file of comma-separated values
# for the household data in the American Community Survey.
filename = 'psam_h06.csv'

# Parse the command-line arguments (if any).
parser = argparse.ArgumentParser()
parser.add_argument(
    '--var', default='NOC', choices=['NOC', 'NP', 'NOC+NP', 'NP+NOC'])
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

# Read and store all but the first two columns in the file for filename.
raw = np.zeros((lines, num_cols - 2))
with open(filename, 'r') as f:
    for line_num, line in enumerate(f):
        parsed = line.split(',')[2:]
        if line_num == 0:
            # The initial line is a header ... save its column labels.
            header = parsed.copy()
            # Eliminate the newline character at the end of the line.
            header[-1] = header[-1][:-1]
        else:
            # All but the initial line consist of data ... extract the ints.
            raw[line_num - 1, :] = np.array(
                [int(s if s != '' else -1) for s in parsed])

# Rename especially interesting columns with easier-to-understand phrases.
header[header.index('NP')] = 'number of people in the household'
header[header.index('NOC')] = 'number of householder\'s own children'

# Filter out undesirable observations -- keep only strictly positive weights,
# strictly positive household personal incomes, and strictly positive factors
# for adjusting the income.
keep = np.logical_and.reduce([
    raw[:, header.index('WGTP')] > 0,
    raw[:, header.index('HINCP')] > 0,
    raw[:, header.index('ADJINC')] > 0])
raw = raw[keep, :]
print(f'm = raw.shape[0] = {raw.shape[0]}')

# Form a dictionary of the lower- and upper-bounds on the ranges of numbers
# of the public-use microdata areas (PUMAs) for the counties in California.
puma = {
    'Alameda': (101, 110),
    'Alpine, Amador, Calaveras, Inyo, Mariposa, Mono and Tuolumne': (300, 300),
    'Butte': (701, 702),
    'Colusa, Glenn, Tehama and Trinity': (1100, 1100),
    'Contra Costa': (1301, 1309),
    'Del Norte, Lassen, Modoc, Plumas and Siskiyou': (1500, 1500),
    'El Dorado': (1700, 1700),
    'Fresno': (1901, 1907),
    'Humboldt': (2300, 2300),
    'Imperial': (2500, 2500),
    'Kern': (2901, 2905),
    'Kings': (3100, 3100),
    'Lake and Mendocino': (3300, 3300),
    'Los Angeles': (3701, 3769),
    'Madera': (3900, 3900),
    'Marin': (4101, 4102),
    'Merced': (4701, 4702),
    'Monterey': (5301, 5303),
    'Napa': (5500, 5500),
    'Nevada and Sierra': (5700, 5700),
    'Orange': (5901, 5918),
    'Placer': (6101, 6103),
    'Riverside': (6501, 6515),
    'Sacramento': (6701, 6712),
    'San Bernardino': (7101, 7115),
    'San Diego': (7301, 7322),
    'San Francisco': (7501, 7507),
    'San Joaquin': (7701, 7704),
    'San Luis Obispo': (7901, 7902),
    'San Mateo': (8101, 8106),
    'Santa Barbara': (8301, 8303),
    'Santa Clara': (8501, 8514),
    'Santa Cruz': (8701, 8702),
    'Shasta': (8900, 8900),
    'Solano': (9501, 9503),
    'Sonoma': (9701, 9703),
    'Stanislaus': (9901, 9904),
    'Sutter and Yuba': (10100, 10100),
    'Tulare': (10701, 10703),
    'Ventura': (11101, 11106),
    'Yolo': (11300, 11300),
}

# Read the weights.
w = raw[:, header.index('WGTP')]

# Read the input covariates.
# Adjust the household personal income by the relevant factor.
var0 = '$\\log_{10}$ of the adjusted household personal income'
s0 = raw[:, header.index('HINCP')] * raw[:, header.index('ADJINC')] / 1e6
# Convert the adjusted incomes to a log (base-10) scale.
s0 = np.log(s0) / math.log(10)
if RANDOMIZE:
    # Dither in order to ensure the uniqueness of the scores.
    rng = default_rng(seed=543216789)
    s0 = s0 * (np.ones(s0.shape) + rng.standard_normal(size=s0.shape) * 1e-8)
# Consider the number of the household's own children for var 'NOC'
# or the number of people in the household for var 'NP'.
if clargs.var == 'NOC+NP':
    var1 = 'number of householder\'s own children'
    var2 = 'number of people in the household'
    s1 = raw[:, header.index(var1)].astype(np.float64)
    s1 = np.clip(s1, 0, 8)
    s2 = raw[:, header.index(var2)].astype(np.float64)
    s2 = np.clip(s2, 0, 10)
    t = np.vstack((s0, s1, s2)).T
elif clargs.var == 'NP+NOC':
    var1 = 'number of people in the household'
    var2 = 'number of householder\'s own children'
    s1 = raw[:, header.index(var1)].astype(np.float64)
    s1 = np.clip(s1, 0, 10)
    s2 = raw[:, header.index(var2)].astype(np.float64)
    s2 = np.clip(s2, 0, 8)
    t = np.vstack((s0, s1, s2)).T
else:
    if clargs.var == 'NOC':
        var1 = 'number of householder\'s own children'
    elif clargs.var == 'NP':
        var1 = 'number of people in the household'
    else:
        raise NotImplementedError(
            clargs.var + ' is not an implemented option.')
    var2 = None
    s1 = raw[:, header.index(var1)].astype(np.float64)
    if var1 == 'number of householder\'s own children':
        s1 = np.clip(s1, 0, 8)
    else:
        s1 = np.clip(s1, 0, 10)
    t = np.vstack((s0, s1)).T

# Proprocess and order the inputs.
# Set the number of covariates.
p = t.shape[1]
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
# Normalize and round the inputs.
it = t.copy()
for k in range(p):
    it[:, k] /= np.max(it[:, k])
it = np.rint((2**precision - 1) * it.astype(np.longdouble)).astype(dtype=dtype)
# Perform the Hilbert mapping from p dimensions to one dimension.
hc = HilbertCurve(precision, p)
ints = hc.distances_from_points(it)
if RANDOMIZE:
    assert np.unique(ints).size == it.shape[0]
# Sort according to the scores.
perm = np.argsort(ints)
t = t[perm, :]
u = t.copy()
for k in range(p):
    t[:, k] /= np.max(t[:, k])
w = w[perm]
# Construct scores for plotting.
imin = np.min(ints)
imax = np.max(ints)
s = (np.sort(ints) - imin) / (imax - imin)
if RANDOMIZE:
    # Ensure uniqueness even after roundoff errors.
    eps = np.finfo(np.float64).eps
    s = s + np.arange(0, s.size * eps, eps)
    s = s.astype(np.float64)
# Process the two records in comp.
r = []
for record in comp:
    # Read the result (raw integer count if the specified value is None,
    # Bernoulli indicator of success otherwise).
    if record['val'] is None:
        r.append(raw[:, header.index(record['var'])])
    else:
        r.append(raw[:, header.index(record['var'])] == record['val'])
    # Sort according to the scores.
    r[-1] = r[-1][perm]

if not clargs.interactive:
    # Create directories as needed.
    dir = 'weighted'
    try:
        os.mkdir(dir)
    except FileExistsError:
        pass
    dir = 'weighted/' + clargs.var
    try:
        os.mkdir(dir)
    except FileExistsError:
        pass

# Process all the counties.
procs = []
for county in counties:
    # Identify the indices of the subset corresponding to the county.
    slice = raw[perm, header.index('PUMA')]
    inds = slice >= (puma[county][0] * np.ones(raw.shape[0]))
    inds = inds & (slice <= (puma[county][1] * np.ones(raw.shape[0])))
    inds = np.nonzero(inds)[0]
    inds = np.unique(inds)
    if not clargs.interactive:
        # Set a directory for the county (creating the directory if necessary).
        dir = 'weighted/' + clargs.var + '/County_of_'
        dir += county.replace(' ', '_').replace(',', '')
        try:
            os.mkdir(dir)
        except FileExistsError:
            pass
        dir += '/'
        print(f'./{dir} is under construction....')
    if var2 is None and not clargs.interactive:
        # Plot the covariates.
        plt.figure()
        plt.xlabel(var0)
        plt.ylabel(var1)
        gray = 1 - .8 * np.vstack(
            (np.array(s[inds]), np.array(s[inds]), np.array(s[inds]))).T
        plt.scatter(
            u[inds, 0], u[inds, 1], s=10, c=gray, marker='o', linewidths=0)
        plt.title('covariates')
        filename = dir + 'covars'
        plt.savefig(filename + '.pdf', bbox_inches='tight')
        args = [
            'convert', '-density', '600', filename + '.pdf',
            filename + '.jpg']
        procs.append(subprocess.Popen(args))
    if clargs.interactive:
        # Plot the cumulative differences interactively.
        covariates = [var0, var1]
        if var2 is not None:
            covariates.append(var2)
        majorticks = 5
        minorticks = 300
        window = county + ' County: '
        window += comp[0]['var'].lower() + ' vs. ' + comp[1]['var'].lower()
        window += ' (click the plot to continue)'
        paired_weighted.icumulative(
            r[0][inds], r[1][inds], s[inds], t[inds], u[inds], covariates,
            majorticks, minorticks, weights=w[inds], window=window)
    else:
        # Plot reliability diagrams and the cumulative graph.
        nin = [10, 20, 100]
        nout = {}
        for nbins in nin:
            filename = dir + 'equiscores' + str(nbins) + '.pdf'
            paired_weighted.equiscores(
                r[0][inds], r[1][inds], s[inds], nbins, filename=filename,
                weights=w[inds], left=0)
            filename = dir + 'equierrs' + str(nbins) + '.pdf'
            rng2 = default_rng(seed=987654321)
            nout[str(nbins)] = paired_weighted.equierrs(
                r[0][inds], r[1][inds], s[inds], nbins, rng2,
                filename=filename, weights=w[inds])
        majorticks = 10
        minorticks = 300
        filename = dir + 'cumulative.pdf'
        kuiper, kolmogorov_smirnov, lenscale = paired_weighted.cumulative(
            r[0][inds], r[1][inds], s[inds], majorticks, minorticks,
            filename=filename, weights=w[inds])
        # Save metrics in a text file.
        filename = dir + 'metrics.txt'
        with open(filename, 'w') as f:
            f.write('len(s):\n')
            f.write(f'{len(s)}\n')
            if not RANDOMIZE:
                f.write('len(np.unique(s[inds])):\n')
                f.write(f'{len(np.unique(s[inds]))}\n')
            f.write('len(inds):\n')
            f.write(f'{len(inds)}\n')
            f.write('lenscale:\n')
            f.write(f'{lenscale}\n')
            for nbins in nin:
                f.write("nout['" + str(nbins) + "']:\n")
                f.write(f'{nout[str(nbins)]}\n')
            f.write('Kuiper:\n')
            f.write(f'{kuiper:.4}\n')
            f.write('Kolmogorov-Smirnov:\n')
            f.write(f'{kolmogorov_smirnov:.4}\n')
            f.write('Kuiper / lenscale:\n')
            f.write(f'{(kuiper / lenscale):.4}\n')
            f.write('Kolmogorov-Smirnov / lenscale:\n')
            f.write(f'{(kolmogorov_smirnov / lenscale):.4}\n')
if var2 is None and not clargs.interactive:
    print('waiting for conversion from pdf to jpg to finish....')
    for iproc, proc in enumerate(procs):
        proc.wait()
        print(f'{iproc + 1} of {len(procs)} conversions are done....')
