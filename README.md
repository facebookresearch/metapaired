The accompanying codes reproduce all figures and statistics presented in
"Cumulative differences between paired samples." This repository also provides
LaTeX and BibTeX sources for replicating the paper.

The main files in the repository are the following:

``tex/paper.pdf``
PDF version of the paper

``tex/paper.tex``
LaTeX source for the paper

``tex/paper.bib``
BibTeX source for the paper

``tex/hilbert.pdf``
PDF plot of an approximation to the Hilbert curve

``codes/paired_weighted.py``
Functions for plotting cumulative differences between paired samples' responses

``codes/kddcup.py``
Python script for processing the KDD Cup 1998 data

``codes/cup98lrn.txt.gz``
Data from the 1998 KDD Cup

Be sure to run ``gunzip codes/cup98lrn.txt.gz`` to decompress the microdata.
Regenerating all the figures requires running in the subdirectory ``codes``
all Python files there, after having decompressed ``codes/cup98lrn.txt.gz``
(being sure to include the command-line option ``--no-interactive`` when
running ``codes/kddcup.py``).

The unit tests invoke [ImageMagick](https://imagemagick.org)'s ``convert``.

********************************************************************************

License

This metapaired software is licensed under the LICENSE file (the MIT license)
in the root directory of this source tree.
