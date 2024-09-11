#!/bin/bash

# This script downloads and installs the Acoustics Toolbox software.

URL="http://oalib.hlsresearch.com/AcousticsToolbox/at_2023_5_18.zip"
DLDEST=src/tmp
BINDIR=src/bin
SETUPDIR=setup/at

if [ ! -d $DLDEST ]; then
    mkdir -p $DLDEST
fi
if [ ! -d $BINDIR ]; then
    mkdir -p $BINDIR
fi

curl -o "$DLDEST/at.zip" $URL
tar -xvf "$DLDEST/at.zip" -C $BINDIR
rm "$DLDEST/at.zip"

if [ "$(uname)" == "Darwin" ] && [ "$(uname -m)" == "arm64" ]; then
    echo "Using Makefile.mac-arm64"
    cp $SETUPDIR/Makefile.mac-arm64 $BINDIR/at/Makefile
else
    echo "Acoustics Toolbox is not supported on this architecture."
fi

cd $BINDIR/at
make clean
make
