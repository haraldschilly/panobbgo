# Panobbgo Solver

Parallel Noisy Black-Box Global Optimization.

It minimizes a function over a box in $R^n$ (n = dimension of the problem).
   
<a href="https://docs.google.com/presentation/pub?id=10fyYYtti5B-rdVE9gaJ-H4LWLMCOlBCVUC8B_gk3wXo&start=false&loop=false&delayms=3000">Short introduction talk.</a>

## DOWNLOAD & INSTALL

This program is work in progress. Only do `python setup.py build|install` if you know what you are doing.

Installation: Basically, you have to install the panobbgo_problem module across the cluster. 
It contains the problem definitions you want to use.
Then, you have to create a script to execute everything; in particular, deserializing the
problem definition needs to be possible on the remote machine.

## Dependencies

* IPython &ge; 0.12.1

  * and you have to start your cluster via `ipcluster start ...` and tell Panobbgo 
    about it :-)

* NumPy &ge; 1.5.0

* SciPy &ge; 0.8.0

## License

<a href="http://www.apache.org/licenses/LICENSE-2.0">Apache 2.0</a>

## Credits

Based on ideas by

* http://reflectometry.org/danse/docs/snobfit/

* http://www.mat.univie.ac.at/~neum/software/snobfit/

