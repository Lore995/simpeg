from __future__ import absolute_import
from __future__ import print_function

from .io_utils_general import read_GOCAD_ts, surface2inds, download
from .io_utils_pf import (
    read_mag3d_ubc,
    write_mag3d_ubc,
    read_grav3d_ubc,
    write_grav3d_ubc,
    read_gg3d_ubc,
    write_gg3d_ubc,
)

from .io_utils_electromagnetics import (
    read_dcip3d_ubc,
    read_dcipoctree_ubc,
    write_dcip3d_ubc,
    write_dcipoctree_ubc
)
