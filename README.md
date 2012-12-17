# Panobbgo: Parallel Noisy Black-Box Global Optimization.

It minimizes a function over a box in $R^n$ (n = dimension of the problem)
while respecting a vector of constraint violations.

## Documentation

* [HTML](http://haraldschilly.github.com/panobbgo/html/)
* [PDF](http://haraldschilly.github.com/panobbgo/pdf/panobbgo.pdf)

## DOWNLOAD & INSTALL

This program is work in progress. Only do `python setup.py build|install` if you know what you are doing.

To get it running, you have to install the reqired dependencies.
I installed `IPython` 0.13 (or higher) from their git sources,
`git checkout v0.13`, and installed it locally: `python setup.py install --user`
(which required to install the `python-zmq` Debian package, too).

To meet the other dependencies, I used [virtualenv](http://www.virtualenv.org/en/latest/)

    $ virtualenv --system-site-packages .
    $ . bin/activate #remember, you have to source this *always*
    $ pip install numpy
    $ pip install scipy
    $ pip install matplotlib

to have up to date versions without having to rely on the packages of Debian.

For the user-interface, you also have to install `python-gtk2`, which
also provides the `pygtk` module, right?

[![Build Status](https://secure.travis-ci.org/haraldschilly/panobbgo.png?branch=master)](https://travis-ci.org/haraldschilly/panobbgo)

## Dependencies

* IPython &ge; 0.13

  * and you have to start your cluster via `ipcluster start ...` and tell Panobbgo 
    about it :-)

* NumPy &ge; 1.5.0

* SciPy &ge; 0.8.0

* matplotlib &ge; 1.1.0 (at least)

* pyGTK &ge; 2.0: `python-gtk2` in Debian/Ubuntu

* nose &ge; 1.1

* coverage &ge; 3.4

* It also calls `Git` to get the ref of the HEAD for logging.

## Running

### One time

1. Setup your cluster according to the IPython documentation (you have to 
   know the profile name, default is `default`)
1. `panobbgo_lib` must be available on all nodes.
   It contains the problem definitions you want to use.
   In particular, you have to create a script to execute everything -
   while especially the problem definition needs to be available for deserialization
   on the remote machine.
1. After running it the first time, it will create a configuration file.
   There you have to enter the profile name, if it is not `default`.

### Every time

1. Start the cluster.
1. If you use `virtualenv`, do `$ . bin/activate` in another terminal.
1. Run the script, examples are included.

## License

<a href="http://www.apache.org/licenses/LICENSE-2.0">Apache 2.0</a>

## Credits

Based on ideas of Snobfit:

* http://reflectometry.org/danse/docs/snobfit/

* http://www.mat.univie.ac.at/~neum/software/snobfit/

