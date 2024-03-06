"""
Subpackage to generate galaxy images with GalSim
"""

from pkgutil import extend_path

from . import params_eu_gen
from . import run_constimg_eu

__path__ = extend_path(__path__, __name__)
