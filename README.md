The accompanying codes reproduce all figures and statistics presented in
"Cumulative differences between paired samples." This repository also provides
LaTeX and BibTeX sources for replicating the paper.

Be sure to ``pip install hilbertcurve`` prior to running any of this software
(the codes depend on [HilbertCurve](https://github.com/galtay/hilbertcurve)).
Also be sure to run ``gunzip codes/cup98lrn.txt.gz`` to decompress the data.
Regenerating all the figures requires running in the subdirectory ``codes``
all Python files there, after having decompressed ``codes/cup98lrn.txt.gz``
and installed hilbertcurve via ``pip`` (being sure to include the option,
``--no-interactive``, when executing the Python scripts on the command-line).

The main files in the repository are the following:

``tex/paper.pdf``
PDF version of the paper

``tex/paper.tex``
LaTeX source for the paper

``tex/paper.bib``
BibTeX source for the paper

``tex/fairmeta.cls``
LaTeX class for the paper

``tex/logometa.pdf``
PDF logo for Meta

``tex/hilbert.pdf``
PDF plot of an approximation to the Hilbert curve

``codes/paired_weighted.py``
Functions for plotting cumulative differences between paired samples' responses

``codes/acs.py``
Python script for processing the 2019 American Community Survey

``codes/psam_h06.csv``
Microdata from the 2019 American Community Survey of the U.S. Census Bureau

``codes/kddcup.py``
Python script for processing the 1998 KDD Cup data

``codes/cup98lrn.txt.gz``
Microdata from the 1998 KDD Cup

The unit tests invoke [ImageMagick](https://imagemagick.org)'s ``convert``.

********************************************************************************

License

This metapaired software is licensed under the LICENSE file (the MIT license)
in the root directory of this source tree.
