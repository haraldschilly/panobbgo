# -*- coding: utf8 -*-
# Copyright 2012 - 2026 Harald Schilly <harald.schilly@univie.ac.at>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pin the floating-point kernels on import: ``import panobbgo.fp_pin`` before numpy.

The benchmark entry points import this module first, so OpenBLAS and numpy
load with fixed kernels (:func:`panobbgo.fp_env.pin_fp_env`; opt out with
``PANOBBGO_FP_PIN=0``).  ``import panobbgo`` pins too; this module makes
the order explicit in a script and records :data:`PINNED`.

``python -m panobbgo.fp_pin`` prints this machine's FP record
(:func:`panobbgo.fp_env.collect`) as JSON, pinned like the entry points.
"""

import json

from panobbgo.fp_env import collect, pin_fp_env

#: Whether the pin is in effect for this process.
PINNED = pin_fp_env()


if __name__ == "__main__":
    print(json.dumps(collect(), indent=2))
