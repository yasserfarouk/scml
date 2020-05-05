#!/usr/bin/env bash
./update.sh
make html

cp ../notebooks/tutorials/run.gif _build/html/tutorials/run.gif
cp ../notebooks/tutorials/run.gif _build/html/_images/run.gif

cp ../notebooks/tutorials/anatomy.pdf _build/html/tutorials/anatomy.pdf
cp ../notebooks/tutorials/anatomy.pdf _build/html/_images/anatomy.pdf

cp ../notebooks/tutorials/anatomy.png _build/html/tutorials/anatomy.png
cp ../notebooks/tutorials/anatomy.png _build/html/_images/anatomy.png

cp ../notebooks/tutorials/negotiate.png _build/html/tutorials/negotiate.png
cp ../notebooks/tutorials/negotiate.png _build/html/_images/negotiate.png

cp ../notebooks/tutorials/trading.png _build/html/tutorials/trading.png
cp ../notebooks/tutorials/trading.png _build/html/_images/trading.png


# open _build/html/index.html
