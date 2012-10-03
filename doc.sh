#!/usr/bin/env bash

set -e

cd `dirname "$0"`

cd doc

COMMIT=`git rev-parse HEAD`

make clean
make html
make latexpdf

cd build
git add -A
git commit --amend -m "doc for $COMMIT"
git push origin gh-pages -f
git gc
