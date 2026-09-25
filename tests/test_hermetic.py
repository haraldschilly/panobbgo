# -*- coding: utf8 -*-
# Copyright 2012 -- 2026 Harald Schilly <harald.schilly@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The autouse conftest fixture keeps tests away from the user's config files."""

import os
import pwd
from pathlib import Path

from panobbgo.config import Config

REPO = Path(__file__).resolve().parent.parent
#: The account's home directory, whatever ``HOME`` says.
REAL_HOME = pwd.getpwuid(os.getuid()).pw_dir


def test_config_ignores_the_real_home_and_the_repo_config_yaml():
    config = Config(parse_args=False, testing_mode=True)

    real_ini = os.path.join(REAL_HOME, ".panobbgo", "config.ini")
    assert os.path.abspath(config.config_fn) != real_ini
    assert os.path.exists(config.config_fn)  # the default ini went into the private HOME
    assert Path.cwd().resolve() != REPO
    assert not config.yaml_config  # ./config.yaml is not the repository's


def test_each_test_gets_a_fresh_home_and_cwd():
    # Nothing a previous test wrote is visible.
    assert os.listdir(Path.home()) == [".panobbgo"]
    assert os.listdir(Path.home() / ".panobbgo") == []
    assert os.listdir(Path.cwd()) == []
