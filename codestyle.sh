#!/usr/bin/env bash

FILES="panobbgo*/*.py *.py doc/source/*.py"
ARGS='--max-line-length=110'
AUTOPEP="autopep8 -i --aggressive $ARGS"

pep8 $ARGS $FILES

for DIR in panobbgo panobbgo.lib sketchpad; do
  find $DIR -name "*.py" -print0 | xargs -0 $AUTOPEP
done

$AUTOPEP setup*.py fabfile.py
