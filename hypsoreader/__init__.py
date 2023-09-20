# -*- coding: utf-8 -*-

# This file is part of hypsoreader.
#
# hypsoreader is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# hypsoreader is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with hypsoreader.  If not, see <http://www.gnu.org/licenses/>.

"""A python package for satellite image processing
"""

__author__ = "Alvaro Flores <alvaro.f.romero@ntnu.no>"
__credits__ = "Norwegian University of Science and Technology"

from .device import Satellite

try:
    from ._version import __version__  # noqa
except:
    pass

from importlib.resources import files
from importlib.metadata import version

all_files_here = files(
    'hypsoreader').iterdir()
for k in all_files_here:
    print(k)
print("============================")
cool_file = files('hypsoreader').joinpath('_version.py')
print(cool_file)
with open(cool_file, 'r') as fin:
    print(fin.read())
