#!/usr/bin/env python3

"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Plots of cumulative differences between paired samples' responses, with weights

*
This implementation assumes that the responses can take only the values 0 or 1,
at least when expected_vals=True as the argument to functions cumulative and
icumulative.
*

This script offers command-line options "--interactive" and "--no-interactive"
for plotting interactively and non-interactively, respectively. Interactive is
the default. The interactive setting does the same as the non-interactive, but
without saving to disk any plots, and without plotting any of the traditional
reliability diagrams. When run non-interactively (that is, with the option
"--no-interactive" given on the command-line), this script creates a directory,
"weighted", in the working directory if the directory does not already exist,
then creates six subdirectories there, named "10_0", "100_0", "10_1", "100_1",
"10_2", and "100_2", with each subdirectory corresponding to 10 or 100 bins in
the reliability diagrams for each of the three examples, where each example has
the indicated number (0, 1, or 2). Each of these six subdirectories gets filled
with seven files:
1. metrics.txt -- metrics about the plots
2. cumulative.pdf -- plot of cumulative differences
3. cumulative_exact.pdf -- PDF plot of exact expectations for cumulative diffs.
4. equiscores.pdf -- reliability diagram with bins equispaced in scores
5. equierrs.pdf -- reliability diagram where the error bar is about the same
                   for every bin
6. exact.pdf -- PDF plot of the exact expectations for the reliability diagrams
7. exact.jpg -- compressed plot of the exact expectations for the reliability
                diagrams

Functions
---------
cumulative
    Cumulative weighted differences between paired samples' responses
icumulative
    Interactive cumulative differences between paired samples' responses
equiscores
    Reliability diagram with roughly equispaced average scores over bins
equierrs
    Reliability diagram with similar ratio L2-norm / L1-norm of weights by bin
exactplot
    Reliability diagram with exact values plotted

This source code is licensed under the MIT license found in the LICENSE file in
the root directory of this source tree.
"""

import argparse
import math
import os
import subprocess
import numpy as np
from numpy.random import default_rng

import matplotlib
from matplotlib.backend_bases import MouseButton
from matplotlib.ticker import FixedFormatter
from matplotlib import get_backend
default_backend = get_backend()
matplotlib.use('agg')
import matplotlib.pyplot as plt


def cumulative(q, r, s, majorticks, minorticks, filename='cumulative.pdf',
               title='deviation is the slope as a function of $A_j$',
               fraction=1, weights=None, expected_vals=False):
    """
    Cumulative weighted differences between paired samples' responses

    Saves a plot of the difference between the normalized cumulative weighted
    sums of q and the normalized cumulative weighted sums of r, with majorticks
    major ticks and minorticks minor ticks on the lower axis, where the labels
    on the major ticks are the corresponding values from s.

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
    majorticks : int
        number of major ticks on each of the horizontal axes
    minorticks : int
        number of minor ticks on the lower axis
    filename : string, optional
        name of the file in which to save the plot
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

    Returns
    -------
    float
        Kuiper statistic
    float
        Kolmogorov-Smirnov statistic
    float
        quarter of the full height of the isosceles triangle
        at the origin in the plot
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

    assert all(s[k] <= s[k + 1] for k in range(len(s) - 1))
    # Determine the weighting scheme.
    if weights is None:
        w = np.ones((len(s)))
    else:
        w = weights.copy()
    assert np.all(w > 0)
    w /= w.sum()
    # Create the figure.
    plt.figure()
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
    plt.margins(x=0, y=.1)
    # Label the major ticks of the lower axis with the values of s.
    lenxf = int(len(x) * fraction)
    sl = ['{:.2f}'.format(a) for a in
          np.insert(s, 0, [0])[:lenxf:(lenxf // majorticks)].tolist()]
    plt.xticks(
        x[:lenxf:(lenxf // majorticks)], sl,
        bbox=dict(boxstyle='Round', fc='w'))
    if len(rsub) >= 300 and minorticks >= 50:
        # Indicate the distribution of s via unlabeled minor ticks.
        plt.minorticks_on()
        ax.tick_params(which='minor', axis='x')
        ax.tick_params(which='minor', axis='y', left=False)
        ax.set_xticks(x[np.cumsum(histcounts(minorticks,
                      s[:int((len(x) - 1) * fraction)]))], minor=True)
    # Label the axes.
    plt.xlabel('$S_j$', labelpad=6)
    plt.ylabel('$C_j$')
    ax2 = plt.twiny()
    plt.xlabel(
        '$j/m$ (together with minor ticks at equispaced values of $A_j$)',
        labelpad=8)
    ax2.tick_params(which='minor', axis='x', top=True, direction='in', pad=-17)
    ax2.set_xticks(np.arange(1 / majorticks, 1, 1 / majorticks), minor=True)
    ks = ['{:.2f}'.format(a) for a in
          np.arange(0, 1 + 1 / majorticks, 1 / majorticks).tolist()]
    alist = (lenxf - 1) * np.arange(0, 1 + 1 / majorticks, 1 / majorticks)
    alist = alist.tolist()
    # Jitter minor ticks that overlap with major ticks lest Pyplot omit them.
    alabs = []
    for a in alist:
        multiple = x[int(a)] * majorticks
        if abs(multiple - round(multiple)) > multiple * 1e-3 / 2:
            alabs.append(x[int(a)])
        else:
            alabs.append(x[int(a)] * (1 + 1e-3))
    plt.xticks(alabs, ks, bbox=dict(boxstyle='Round', fc='w'))
    ax2.xaxis.set_minor_formatter(FixedFormatter(
        [r'$A_j\!=\!{:.2f}$'.format(1 / majorticks)]
        + [r'${:.2f}$'.format(k / majorticks) for k in range(2, majorticks)]))
    # Title the plot.
    plt.title(title)
    # Clean up the whitespace in the plot.
    plt.tight_layout()
    # Save the plot.
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    # Calculate summary statistics.
    qr = (qa - ra)[:int(len(qa) * fraction)]
    kuiper = np.max(qr) - np.min(qr)
    kolmogorov_smirnov = np.max(np.abs(qr))
    return kuiper, kolmogorov_smirnov, lenscale


def icumulative(q, r, s, t, u, covariates, majorticks, minorticks,
                title='deviation is the slope as a function of $A_j$',
                fraction=1, weights=None, expected_vals=False,
                window='Figure'):
    """
    Interactive cumulative differences between paired samples' responses

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
    plt.xticks(
        x[:lenxf:(lenxf // majorticks)], sl,
        bbox=dict(boxstyle='Round', fc='w'))
    if len(rsub) >= 300 and minorticks >= 50:
        # Indicate the distribution of s via unlabeled minor ticks.
        plt.minorticks_on()
        ax.tick_params(which='minor', axis='x')
        ax.tick_params(which='minor', axis='y', left=False)
        ax.set_xticks(x[np.cumsum(histcounts(minorticks,
                      s[:int((len(x) - 1) * fraction)]))], minor=True)
    # Label the axes.
    plt.xlabel('$S_j$', labelpad=6)
    plt.ylabel('$C_j$')
    ax2 = plt.twiny()
    plt.xlabel(
        '$j/m$ (together with minor ticks at equispaced values of $A_j$)',
        labelpad=8)
    ax2.tick_params(which='minor', axis='x', top=True, direction='in', pad=-17)
    ax2.set_xticks(np.arange(1 / majorticks, 1, 1 / majorticks), minor=True)
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
    plt.xticks(alabs, ks, bbox=dict(boxstyle='Round', fc='w'))
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


def equiscores(q, r, s, nbins, filename='equiscore.pdf', weights=None,
               top=None, left=None, right=None):
    """
    Reliability diagram with roughly equispaced average scores over bins

    Plots a reliability diagram with roughly equispaced average scores
    for the bins.

    Parameters
    ----------
    q : array_like
        random outcomes for one population
    r : array_like
        random outcomes for another population
    s : array_like
        scores (must be in non-decreasing order)
    nbins : int
        number of bins
    filename : string, optional
        name of the file in which to save the plot
    weights : array_like, optional
        weights of the observations
        (the default None results in equal weighting)
    top : float, optional
        top of the range of the vertical axis (the default None is adaptive)
    left : float, optional
        leftmost value of the horizontal axis (the default None is adaptive)
    right : float, optional
        rightmost value of the horizontal axis (the default None is adaptive)

    Returns
    -------
    None
    """

    def bintwo(nbins, a, b, q, qmax, w):
        # Determines the total weight of entries of q falling into each
        # of nbins equispaced bins, and calculates the weighted average per bin
        # of the arrays a and b, returning np.nan as the "average"
        # for any bin that is empty.
        j = 0
        bina = np.zeros(nbins)
        binb = np.zeros(nbins)
        wbin = np.zeros(nbins)
        for k in range(len(q)):
            if q[k] > qmax * (j + 1) / nbins:
                j += 1
            if j == nbins:
                break
            bina[j] += w[k] * a[k]
            binb[j] += w[k] * b[k]
            wbin[j] += w[k]
        # Normalize the sum for each bin to compute the arithmetic average.
        bina = np.divide(bina, wbin, where=wbin != 0)
        bina[np.where(wbin == 0)] = np.nan
        binb = np.divide(binb, wbin, where=wbin != 0)
        binb[np.where(wbin == 0)] = np.nan
        return wbin, bina, binb

    assert all(s[k] <= s[k + 1] for k in range(len(s) - 1))
    # Determine the weighting scheme.
    if weights is None:
        w = np.ones((len(s)))
    else:
        w = weights.copy()
    assert np.all(w > 0)
    w /= w.sum()
    # Create the figure.
    plt.figure()
    _, binr, binsr = bintwo(nbins, r, s, s, s[-1], w)
    _, binq, binsq = bintwo(nbins, q, s, s, s[-1], w)
    assert np.allclose(binsq, binsr, equal_nan=True)
    plt.plot(binsr, binr, '*:', color='gray')
    plt.plot(binsq, binq, '*:', color='black')
    xmin = s[0] if left is None else left
    xmax = s[-1] if right is None else right
    plt.xlim((xmin, xmax))
    plt.ylim(bottom=0)
    plt.ylim(top=top)
    plt.xlabel('weighted average of $S_j$ for $j$ in the bin')
    plt.ylabel(
        'weighted average of $Q_j^{(k)}$ (black) or $R_j^{(k)}$ (gray)'
        + ' for $j$ in the bin')
    plt.title('reliability diagram')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def equierrs(q, r, s, nbins, rng, filename='equibins.pdf', weights=None,
             top=None, left=None, right=None):
    """
    Reliability diagram with similar ratio L2-norm / L1-norm of weights by bin

    Plots a reliability diagram with the ratio of the L2 norm of the weights
    to the L1 norm of the weights being roughly the same for every bin.
    The L2 norm is the square root of the sum of the squares, while the L1 norm
    is the sum of the absolute values.

    Parameters
    ----------
    q : array_like
        random outcomes for one population
    r : array_like
        random outcomes for another population
    s : array_like
        scores (must be in non-decreasing order)
    nbins : int
        rough number of bins to construct
    rng : Generator
        fully initialized random number generator from NumPy
    filename : string, optional
        name of the file in which to save the plot
    weights : array_like, optional
        weights of the observations
        (the default None results in equal weighting)
    top : float, optional
        top of the range of the vertical axis (the default None is adaptive)
    left : float, optional
        leftmost value of the horizontal axis (the default None is adaptive)
    right : float, optional
        rightmost value of the horizontal axis (the default None is adaptive)

    Returns
    -------
    int
        number of bins constructed
    """

    def inbintwo(a, b, inbin, w):
        # Determines the total weight falling into the bins given by inbin,
        # and calculates the weighted average per bin of the arrays a and b,
        # returning np.nan as the "average" for any bin that is empty.
        wbin = [w[inbin[k]:inbin[k + 1]].sum() for k in range(len(inbin) - 1)]
        bina = [(w[inbin[k]:inbin[k + 1]] * a[inbin[k]:inbin[k + 1]]).sum()
                for k in range(len(inbin) - 1)]
        binb = [(w[inbin[k]:inbin[k + 1]] * b[inbin[k]:inbin[k + 1]]).sum()
                for k in range(len(inbin) - 1)]
        # Normalize the sum for each bin to compute the weighted average.
        bina = np.divide(bina, wbin, where=wbin != 0)
        bina[np.where(wbin == 0)] = np.nan
        binb = np.divide(binb, wbin, where=wbin != 0)
        binb[np.where(wbin == 0)] = np.nan
        return wbin, bina, binb

    def binbounds(nbins, w):
        # Partitions w into around nbins bins, each with roughly equal ratio
        # of the L2 norm of w in the bin to the L1 norm of w in the bin,
        # returning the indices defining the bins in the list inbin.
        proxy = len(w) // nbins
        v = w[np.sort(rng.permutation(len(w))[:proxy])]
        # t is a heuristic threshold.
        t = np.square(v).sum() / v.sum()**2
        inbin = []
        k = 0
        while k < len(w) - 1:
            inbin.append(k)
            k += 1
            s = w[k]
            ss = w[k]**2
            while ss / s**2 > t and k < len(w) - 1:
                k += 1
                s += w[k]
                ss += w[k]**2
        if len(w) - inbin[-1] < (inbin[-1] - inbin[-2]) / 2:
            inbin[-1] = len(w)
        else:
            inbin.append(len(w))
        return inbin

    assert all(s[k] <= s[k + 1] for k in range(len(s) - 1))
    # Determine the weighting scheme.
    if weights is None:
        w = np.ones((len(s)))
    else:
        w = weights.copy()
    assert np.all(w > 0)
    w /= w.sum()
    inbin = binbounds(nbins, w)
    # Create the figure.
    plt.figure()
    _, binr, binsr = inbintwo(r, s, inbin, w)
    _, binq, binsq = inbintwo(q, s, inbin, w)
    assert np.allclose(binsr, binsq, equal_nan=True)
    plt.plot(binsr, binr, '*:', color='gray')
    plt.plot(binsq, binq, '*:', color='black')
    xmin = binsr[0] if left is None else left
    xmax = binsr[-1] if right is None else right
    plt.xlim((xmin, xmax))
    plt.ylim(bottom=0)
    plt.ylim(top=top)
    plt.xlabel('weighted average of $S_j$ for $j$ in the bin')
    plt.ylabel(
        'weighted average of $Q_j^{(k)}$ (black) or $R_j^{(k)}$ (gray)'
        + ' for $j$ in the bin')
    title = r'reliability diagram'
    title += r' ($\Vert W \Vert_2 / \Vert W \Vert_1$ is similar for every bin)'
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

    return len(inbin) - 1


def exactplot(q, r, s, filename='exact.pdf', title='exact expectations',
              top=None, left=None, right=None):
    """
    Reliability diagram with exact values plotted

    Plots a reliability diagram at full resolution with fractional numbers.
    The entries of q and r should be the expected values of outcomes.

    Parameters
    ----------
    q : array_like
        expected value of outcomes for one population
    r : array_like
        expected value of outcomes for another population
    s : array_like
        scores (must be in non-decreasing order)
    filename : string, optional
        name of the file in which to save the plot
    title : string, optional
        title of the plot
    top : float, optional
        top of the range of the vertical axis (the default None is adaptive)
    left : float, optional
        leftmost value of the horizontal axis (the default None is adaptive)
    right : float, optional
        rightmost value of the horizontal axis (the default None is adaptive)

    Returns
    -------
    None
    """
    assert all(s[k] <= s[k + 1] for k in range(len(s) - 1))
    plt.figure()
    plt.plot(s, r, '*', color='gray')
    plt.plot(s, q, '*', color='black')
    plt.xlim((left, right))
    plt.ylim(bottom=0)
    plt.ylim(top=top)
    plt.xlabel('score $S_j$')
    plt.ylabel(
        'expected value of weighted average outcome'
        + r' $\tilde{Q}_j$ or $\tilde{R}_j$')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    # Parse the command-line arguments (if any).
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--interactive', dest='interactive', action='store_true')
    parser.add_argument(
        '--no-interactive', dest='interactive', action='store_false')
    parser.add_argument(
        '--non-interactive', dest='interactive', action='store_false')
    parser.set_defaults(interactive=True)
    clargs = parser.parse_args()
    # Make matplotlib interactive if clargs.interactive is True.
    if clargs.interactive:
        plt.switch_backend(default_backend)
    #
    # Generate directories with plots as specified via the code below,
    # with each directory named "nbins_inds," where nbins is the number of bins
    # for the reliability diagrams and inds is an index -- 0, 1, or 2 --
    # for the 3 examples considered.
    #
    # Set parameters.
    # minorticks is the number of minor ticks on the lower axis.
    minorticks = 100
    # majorticks is the number of major ticks on the lower axis.
    majorticks = 10
    # n is the number of observations.
    n = 4000
    if not clargs.interactive:
        # Store processes for converting from pdf to jpeg in procs.
        procs = []
    # Consider 3 examples.
    for iex in range(3):
        # nbins is the number of bins for the reliability diagrams.
        nbins_vals = [10, 100]
        for nbins in nbins_vals:
            # nbins must divide n evenly.
            assert n % nbins == 0

            if iex == 0:
                # Construct scores.
                sl = np.arange(0, 1, 4 / n) + 2 / n
                s = np.square(sl)
                s = np.concatenate([s] * 4)
                # The scores must be in non-decreasing order.
                s = np.sort(s)

                # Construct perturbations to the scores for sampling rates.
                d = .25
                tl = -np.arange(-d, d, 2 * d / n) - d / n
                t = d - 2.5 * np.square(np.square(tl)) / d**3
                e = .7
                ul = -np.arange(-e, e, 2 * e / n) - e / n
                u = e - np.abs(ul)
                u[n // 2:] = (2 * t - u)[n // 2:]
                ins = np.arange(n // 2 - n // 50, n // 2 + n // 50)
                u[ins] = t[ins]

            if iex == 1:
                # Construct scores.
                s = np.arange(0, 1, 1 / n) + 1 / (2 * n)
                s = np.sqrt(s)
                # The scores must be in non-decreasing order.
                s = np.sort(s)

                # Construct perturbations to the scores for sampling rates.
                d = math.sqrt(1 / 2)
                tl = np.arange(-d, d, 2 * d / n) - d / n
                t = np.square(tl) - d**2
                u = np.square(tl) - d**2
                u *= .75 * np.round(1 + np.cos(8 * np.arange((n)) / n))

            if iex == 2:
                # Construct scores.
                s = np.arange(0, 1, 10 / n) + 5 / n
                s = np.concatenate([s] * 10)
                # The scores must be in non-decreasing order.
                s = np.sort(s)

                # Construct perturbations to the scores for sampling rates.
                tl = np.arange(0, 1, 1 / n) + 1 / (2 * n)
                t = (np.power(tl, 1 / 4) - tl) * (n + tl) / (2 * n)
                u = np.power(tl, 1 / 4) - tl
                u *= .6 * (.9 + np.sin(
                    25 * np.power(np.arange(0, n**4, n**3), 1 / 4) / n))

            # Construct the exact sampling probabilities.
            qexact = s + t
            rexact = s + u

            # Construct weights.
            weights = 4 - np.cos(9 * np.arange(n) / n)

            if not clargs.interactive:
                # Set a unique directory for each collection of experiments
                # (creating the directory if necessary).
                dir = 'weighted'
                try:
                    os.mkdir(dir)
                except FileExistsError:
                    pass
                dir = 'weighted/' + str(nbins) + '_' + str(iex)
                try:
                    os.mkdir(dir)
                except FileExistsError:
                    pass
                dir = dir + '/'
                print(f'./{dir} is under construction....')

            # Generate a sample of classifications into two classes,
            # correct (class 1) and incorrect (class 0),
            # avoiding numpy's random number generators
            # that are based on random bits --
            # they yield strange results for many seeds.
            rng = default_rng(seed=12345432123454321)
            uniform = np.asarray([rng.uniform() for _ in range(n)])
            q = (uniform <= qexact).astype(float)
            uniform = np.asarray([rng.uniform() for _ in range(n)])
            r = (uniform <= rexact).astype(float)

            if clargs.interactive:
                if nbins == nbins_vals[0]:
                    window = 'Example ' + str(iex)
                    window += ' (click the plot to continue)'
                    unnormalized = s[..., np.newaxis]
                    normalized = unnormalized / np.max(s)
                    covariates = ['score']
                    icumulative(
                        q, r, s, normalized, unnormalized, covariates,
                        majorticks, minorticks, weights=weights,
                        expected_vals=True, window=window)
            else:
                # Generate five plots and a text file reporting metrics.
                filename = dir + 'cumulative.pdf'
                kuiper, kolmogorov_smirnov, lenscale = cumulative(
                    q, r, s, majorticks, minorticks, filename, weights=weights)
                filename = dir + 'metrics.txt'
                with open(filename, 'w') as f:
                    f.write('n:\n')
                    f.write(f'{n}\n')
                    f.write('number of unique scores in the subset:\n')
                    f.write(f'{len(np.unique(s))}\n')
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
                filename = dir + 'cumulative_exact.pdf'
                _, _, _ = cumulative(
                    qexact, rexact, s, majorticks, minorticks, filename,
                    title='exact expectations', weights=weights,
                    expected_vals=True)
                filename = dir + 'equiscores.pdf'
                equiscores(q, r, s, nbins, filename, weights, top=1, left=0,
                           right=1)
                filename = dir + 'equierrs.pdf'
                rng = default_rng(seed=987654321)
                equierrs(q, r, s, nbins, rng, filename, weights, top=1, left=0,
                         right=1)
                filepdf = dir + 'exact.pdf'
                filejpg = dir + 'exact.jpg'
                exactplot(qexact, rexact, s, filepdf, top=1, left=0, right=1)
                args = ['convert', '-density', '1200', filepdf, filejpg]
                procs.append(subprocess.Popen(args))
    if not clargs.interactive:
        print('waiting for conversion from pdf to jpg to finish....')
        for iproc, proc in enumerate(procs):
            proc.wait()
            print(f'{iproc + 1} of {len(procs)} conversions are done....')
