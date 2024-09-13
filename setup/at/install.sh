#!/bin/sh

# This script downloads and installs the Acoustics Toolbox software.

URL="http://oalib.hlsresearch.com/AcousticsToolbox/at_2023_5_18.zip"
DLDEST=src/tmp
LIBDIR=src/lib
SETUPDIR=setup/at

if [ ! -d $DLDEST ]; then
    mkdir -p $DLDEST
fi
if [ ! -d $LIBDIR ]; then
    mkdir -p $LIBDIR
fi

curl -o "$DLDEST/at.zip" $URL
tar -xvf "$DLDEST/at.zip" -C $LIBDIR
rm "$DLDEST/at.zip"

if [ "$(uname)" == "Darwin" ] && [ "$(uname -m)" == "arm64" ]; then
    echo "Using Makefile.mac-arm64"
    cp $SETUPDIR/Makefile.mac-arm64 $LIBDIR/at/Makefile
else
    echo "Acoustics Toolbox is not supported on this architecture."
fi

cd $LIBDIR/at
make clean
make

mkdir bin
cp Bellhop/bellhop.exe bin/
cp Bellhop/bellhop3d.exe bin/
cp Kraken/bounce.exe bin/
cp Kraken/kraken.exe bin/
cp Kraken/krakenc.exe bin/
cp KrakenField/field.exe bin/
cp KrakenField/field3d.exe bin/
cp Scooter/scooter.exe bin/
cp Scooter/sparc.exe bin/

find . -mindepth 1 \
    -path './bin' -prune -o \
    -path './doc' -prune -o \
    -not -name 'index.html' \
    -not -name 'LICENSE' \
    -not -name 'Makefile' \
    -exec rm -rf {} +
