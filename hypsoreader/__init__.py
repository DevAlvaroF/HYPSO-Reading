# -*- coding: utf-8 -*-
#
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

from .plot import *
from .spectra import *

try:
    from ._version import version as __version__
except ModuleNotFoundError:  # development mode
    __version__ = '0.2.0'
