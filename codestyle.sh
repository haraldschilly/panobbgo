#!/usr/bin/env bash

FILES="panobbgo*/*.py *.py doc/source/*.py"
ARGS='--max-line-length=110'

pep8 $ARGS $FILES

for DIR in panobbgo panobbgo_lib; do
  find $DIR -name "*.py"  | xargs autopep8 -i --aggressive $ARGS
done
