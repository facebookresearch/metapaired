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
from matplotlib.backend_bases import MouseButton
from matplotlib.ticker import FixedFormatter
from matplotlib import get_backend
default_backend = get_backend()
matplotlib.use('agg')
import matplotlib.pyplot as plt

from hilbertcurve.hilbertcurve import HilbertCurve
import paired_weighted


def icumulative(q, r, s, t, u, covariates, majorticks, minorticks,
                title='deviation is the slope as a function of $A_j$',
                fraction=1, weights=None, expected_vals=False,
                window='Figure'):
    """
    Cumulative weighted differences between paired samples' responses

    Plots the difference between the normalized cumulative weighted sums of q
    and the normalized cumulative weighted sums of r, with majorticks major
    ticks and minorticks minor ticks on the lower axis, where the labels
    on the major ticks are the corresponding values from s.
    This is an interactive version of paired_weighted.cumulative (probably not
    suitable in general for all data sets, however).

    _N.B._: The outputs when expected_vals=True will be correct only
    if the responses are Bernoulli variates, taking values 0 or 1 only.

    Parameters
    ----------
    q : array_like
        random outcomes from one population
    r : array_like
        random outcomes from another population
    s : array_like
        scores (must be in non-decreasing order)
    t : array_like
        normalized values of the covariates
    u : array_like
        unnormalized values of the covariates
    covariates : array_like
        strings labeling the covariates
    majorticks : int
        number of major ticks on each of the horizontal axes
    minorticks : int
        number of minor ticks on the lower axis
    title : string, optional
        title of the plot
    fraction : float, optional
        proportion of the full horizontal axis to display
    weights : array_like, optional
        weights of the observations
        (the default None results in equal weighting)
    expected_vals : bool, optional
        set to True if the entries of the inputs q and r are expected values;
        set to False (the default) if the entries are random observations
    window : string, optional
        title of the window displayed in the title bar

    Returns
    -------
    None
    """

    def histcounts(nbins, a):
        # Counts the number of entries of a
        # falling into each of nbins equispaced bins.
        j = 0
        nbin = np.zeros(nbins, dtype=np.int64)
        for k in range(len(a)):
            if a[k] > a[-1] * (j + 1) / nbins:
                j += 1
            if j == nbins:
                break
            nbin[j] += 1
        return nbin

    def on_move(event):
        if event.inaxes:
            ax = event.inaxes
            k = round(event.xdata * (len(s) - 1))
            toptxt = ''
            bottomtxt = '\n'
            for j in range(len(covariates)):
                toptxt += covariates[j]
                if(np.allclose(
                        np.round(u[k, j]), u[k, j], rtol=1e-5)):
                    toptxt += ' = {}'.format(round(u[k, j]))
                else:
                    toptxt += ' = {:.2f}'.format(u[k, j])
                toptxt += '\n'
                bottomtxt += 'normalized ' + covariates[j]
                bottomtxt += ' = {:.2f}'.format(t[k, j])
                bottomtxt += '\n'
            toptxt += '$S_j$' + ' = {:.2f}'.format(s[k])
            bottomtxt += '$S_j$' + ' = {:.2f}'.format(s[k])
            toptext.set_text(toptxt)
            bottomtext.set_text(bottomtxt)
            plt.draw()

    def on_click(event):
        if event.button is MouseButton.LEFT:
            plt.disconnect(binding_id)
            plt.close()

    assert all(s[k] <= s[k + 1] for k in range(len(s) - 1))
    # Determine the weighting scheme.
    if weights is None:
        w = np.ones((len(s)))
    else:
        w = weights.copy()
    assert np.all(w > 0)
    w /= w.sum()
    # Create the figure.
    plt.figure(window)
    ax = plt.axes()
    # Average the results and total the weights for repeated scores.
    # Also calculate factors for adjusting the variances of responses
    # to account for the responses being averages of other responses
    # (when the scores need not be unique).
    slist = []
    rlist = []
    qlist = []
    wlist = []
    flist = []
    rsum = 0
    qsum = 0
    wsum = 0
    wsos = 0
    for k in range(len(s)):
        rsum += r[k] * w[k]
        qsum += q[k] * w[k]
        wsum += w[k]
        wsos += w[k]**2
        if k == len(s) - 1 or s[k] != s[k + 1]:
            slist.append(s[k])
            rlist.append(rsum / wsum)
            qlist.append(qsum / wsum)
            wlist.append(wsum)
            flist.append(wsos / wsum**2)
            rsum = 0
            qsum = 0
            wsum = 0
            wsos = 0
    s = np.asarray(slist)
    r = np.asarray(rlist)
    q = np.asarray(qlist)
    w = np.asarray(wlist)
    f = np.asarray(flist)
    # Normalize the weights.
    w /= w[:int(len(w) * fraction)].sum()
    # Accumulate the weighted q and r, as well as w.
    qa = np.insert(np.cumsum(w * q), 0, [0])
    ra = np.insert(np.cumsum(w * r), 0, [0])
    x = np.insert(np.cumsum(w), 0, [0])
    # Plot the difference.
    plt.plot(
        x[:int(len(x) * fraction)], (qa - ra)[:int(len(x) * fraction)], 'k')
    # Make sure the plot includes the origin.
    plt.plot(0, 'k')
    # Add an indicator of the scale of 1/sqrt(n) to the vertical axis.
    rsub = np.insert(r, 0, [0])[:(int(len(r) * fraction) + 1)]
    qsub = np.insert(q, 0, [0])[:(int(len(q) * fraction) + 1)]
    if expected_vals:
        lenscale = np.sum(w**2 * qsub[1:] * (1 - qsub[1:]) * f)
        lenscale += np.sum(w**2 * rsub[1:] * (1 - rsub[1:]) * f)
    else:
        lenscale = np.sum(w**2 * (qsub[1:] - rsub[1:])**2)
    lenscale = np.sqrt(lenscale)
    plt.plot(2 * lenscale, 'k')
    plt.plot(-2 * lenscale, 'k')
    kwargs = {
        'head_length': 2 * lenscale, 'head_width': fraction / 20, 'width': 0,
        'linewidth': 0, 'length_includes_head': True, 'color': 'k'}
    plt.arrow(.1e-100, -2 * lenscale, 0, 4 * lenscale, shape='left', **kwargs)
    plt.arrow(.1e-100, 2 * lenscale, 0, -4 * lenscale, shape='right', **kwargs)
    plt.margins(x=0, y=.6)
    # Label the major ticks of the lower axis with the values of s.
    lenxf = int(len(x) * fraction)
    sl = ['{:.2f}'.format(a) for a in
          np.insert(s, 0, [0])[:lenxf:(lenxf // majorticks)].tolist()]
    plt.xticks(x[:lenxf:(lenxf // majorticks)], sl)
    if len(rsub) >= 300 and minorticks >= 50:
        # Indicate the distribution of s via unlabeled minor ticks.
        plt.minorticks_on()
        ax.tick_params(which='minor', axis='x')
        ax.tick_params(which='minor', axis='y', left=False)
        ax.set_xticks(x[np.cumsum(histcounts(minorticks,
                      s[:int((len(x) - 1) * fraction)]))], minor=True)
    # Label the axes.
    plt.xlabel('$S_j$')
    plt.ylabel('$C_j$')
    ax2 = plt.twiny()
    plt.xlabel(
        '$j/m$ (together with minor ticks at equispaced values of $A_j$)')
    ax2.tick_params(which='minor', axis='x', top=True, direction='in', pad=-17)
    ax2.set_xticks(np.arange(0, 1 + 1 / majorticks, 1 / majorticks),
                   minor=True)
    ks = ['{:.2f}'.format(a) for a in
          np.arange(0, 1 + 1 / majorticks, 1 / majorticks).tolist()]
    alist = (lenxf - 1) * np.arange(0, 1 + 1 / majorticks, 1 / majorticks)
    alist = alist.tolist()
    # Jitter minor ticks that overlap with major ticks lest Pyplot omit them.
    alabs = []
    for a in alist:
        multiple = x[int(a)] * majorticks
        if abs(multiple - round(multiple)) > 1e-4:
            alabs.append(x[int(a)])
        else:
            alabs.append(x[int(a)] * (1 - 1e-4))
    plt.xticks(alabs, ks)
    ax2.xaxis.set_minor_formatter(FixedFormatter(
        [r'$A_j\!=\!{:.2f}$'.format(1 / majorticks)]
        + [r'${:.2f}$'.format(k / majorticks) for k in range(2, majorticks)]))
    # Title the plot.
    plt.title(title)
    # Clean up the whitespace in the plot.
    plt.tight_layout()
    # Set the locations (in the plot) of the covariate values.
    xmid = s[:(len(s) * fraction)][-1] / 2
    toptext = plt.text(
        xmid, max(2 * lenscale, np.max((qa - ra)[:(len(qa) * fraction)])), '',
        ha='center', va='bottom')
    bottomtext = plt.text(
        xmid, min(-2 * lenscale, np.min((qa - ra)[:(len(qa) * fraction)])), '',
        ha='center', va='top')
    # Set up interactivity.
    binding_id = plt.connect('motion_notify_event', on_move)
    plt.connect('button_press_event', on_click)
    # Show the plot.
    plt.show()
    plt.close()


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
        window = 'donations made in 1995 versus in 1996'
        window += ' (click the plot to continue)'
        icumulative(r0, r1, s, t, u, covariates, majorticks, minorticks,
                    window=window)
    else:
        # Plot reliability diagrams and the cumulative graph.
        filename = dir + 'cumulative.pdf'
        majorticks = 10
        minorticks = 300
        kuiper, kolmogorov_smirnov, lenscale = paired_weighted.cumulative(
            r0, r1, s, majorticks, minorticks, filename)
        filename = dir + 'metrics.txt'
        with open(filename, 'w') as f:
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
            paired_weighted.equiscores(r0, r1, s, nbins, filename)
            filename = dir + 'equierrs' + str(nbins) + '.pdf'
            paired_weighted.equierrs(r0, r1, s, nbins, filename)
if not clargs.interactive:
    print()
    print('waiting for conversion from pdf to jpg to finish....')
    for iproc, proc in enumerate(procs):
        proc.wait()
        print(f'{iproc + 1} of {len(procs)} conversions are done....')
