from fabric.api import local, run


def info():
    run("uname -a")
    run("lsb_release -a")


# def freeze():
#    local("pip freeze > requirements.txt")
#    local("git add requirements.txt")
#    local("git commit -v")


def pip_upgrade():
    import pip

    for dist in pip.get_installed_distributions():
        local("pip install --upgrade {0}".format(dist.project_name))


def clean_pyc():
    local("find -name '*.pyc' -delete")
