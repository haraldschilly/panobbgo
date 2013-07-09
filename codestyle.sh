#!/usr/bin/env bash

FILES="panobbgo*/*.py *.py doc/source/*.py"
ARGS='--max-line-length=110'
AUTOPEP="autopep8 -i --aggressive $ARGS"

pep8 $ARGS $FILES

for DIR in panobbgo panobbgo_lib; do
  find $DIR -name "*.py"  | xargs $AUTOPEP
done

$AUTOPEP setup*.py
