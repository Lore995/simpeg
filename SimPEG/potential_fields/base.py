from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import properties
import numpy as np
import multiprocessing
from ..simulation import LinearSimulation
from scipy.sparse import csr_matrix as csr
from SimPEG.utils import mkvc, sdiag
from .. import props
from dask import delayed, array, config
from dask.diagnostics import ProgressBar

try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver

###############################################################################
#                                                                             #
#                             Base Potential Fields Problem                   #
#                                                                             #
###############################################################################


class BasePFSimulation(LinearSimulation):

    store_sensitivity = properties.Bool(
        "Store the sensitivity to disk",
        default=True
    )

    actInd = properties.Array(
        "Array of active cells (ground)",
        dtype=(bool, int),
        default=None
    )

    n_cpu = properties.Integer(
        "Number of processors used for the forward simulation",
        default=int(multiprocessing.cpu_count())
    )

    store_sensitivities = properties.StringChoice(
        "Compute and store G",
        choices=['disk', 'ram', 'forward_only'],
        default='disk'
    )

    max_chunk_size = properties.Float(
        "Largest chunk size (Mb) used by Dask",
        default=128
    )

    chunk_format = properties.StringChoice(
        "Apply memory chunks along rows of G",
        choices=['equal', 'row', 'auto'],
        default='equal'
    )

    max_ram = properties.Float(
        "Target maximum memory (Gb) usage",
        default=128
    )

    sensitivity_path = properties.String(
        "Directory used to store the sensitivity matrix on disk",
        default="./Inversion/sensitivity.zarr"
    )

    def __init__(self, mesh, **kwargs):

        LinearSimulation.__init__(self, mesh, **kwargs)

        # Find non-zero cells
        if getattr(self, 'actInd', None) is not None:
            if self.actInd.dtype == 'bool':
                indices = np.where(self.actInd)[0]
            else:
                indices = self.actInd

        else:

            indices = np.asarray(range(self.mesh.nC))

        self.nC = len(indices)

        # Create active cell projector
        projection = csr(
            (np.ones(self.nC), (indices, range(self.nC))),
            shape=(self.mesh.nC, self.nC)
        )

        # Create vectors of nodal location for the lower and upper corners
        bsw = (self.mesh.gridCC - self.mesh.h_gridded/2.)
        tne = (self.mesh.gridCC + self.mesh.h_gridded/2.)

        xn1, xn2 = bsw[:, 0], tne[:, 0]
        yn1, yn2 = bsw[:, 1], tne[:, 1]

        self.Yn = projection.T*np.c_[mkvc(yn1), mkvc(yn2)]
        self.Xn = projection.T*np.c_[mkvc(xn1), mkvc(xn2)]

        # Allows for 2D mesh where Zn is defined by user
        if self.mesh.dim > 2:
            zn1, zn2 = bsw[:, 2], tne[:, 2]
            self.Zn = projection.T*np.c_[mkvc(zn1), mkvc(zn2)]

    def linear_operator(self):

        self.nC = self.modelMap.shape[1]

        n_data_comp = len(self.survey.components)

        components = np.array(list(self.survey.components.keys()))
        active_components = np.hstack([np.c_[values] for values in self.survey.components.values()]).tolist()

        if self.store_sensitivities != 'ram':

            row = delayed(self.evaluate_integral, pure=True)

            rows = [
                array.from_delayed(
                    row(receiver_location, components[component]), dtype=np.float32, shape=(n_data_comp,  self.nC)
                )
                for receiver_location, component in zip(self.survey.receiver_locations.tolist(), active_components)
            ]
            stack = array.vstack(rows)

            # Chunking options
            if self.chunk_format == 'row' or self.store_sensitivities == 'forward_only':
                config.set({'array.chunk-size': f'{self.max_chunk_size}MiB'})
                # Autochunking by rows is faster and more memory efficient for
                # very large problems sensitivty and forward calculations
                stack = stack.rechunk({0: 'auto', 1: -1})

            elif self.chunk_format == 'equal':
                # Manual chunks for equal number of blocks along rows and columns.
                # Optimal for Jvec and Jtvec operations
                n_chunks_col = 1
                n_chunks_row = 1
                row_chunk = int(np.ceil(stack.shape[0]/n_chunks_row))
                col_chunk = int(np.ceil(stack.shape[1]/n_chunks_col))
                chunk_size = row_chunk*col_chunk*8*1e-6  # in Mb

                # Add more chunks along either dimensions until memory falls below target
                while chunk_size >= self.max_chunk_size:

                    if row_chunk > col_chunk:
                        n_chunks_row += 1
                    else:
                        n_chunks_col += 1

                    row_chunk = int(np.ceil(stack.shape[0]/n_chunks_row))
                    col_chunk = int(np.ceil(stack.shape[1]/n_chunks_col))
                    chunk_size = row_chunk*col_chunk*8*1e-6  # in Mb

                stack = stack.rechunk((row_chunk, col_chunk))
            else:
                # Auto chunking by columns is faster for Inversions
                config.set({'array.chunk-size': f'{self.max_chunk_size}MiB'})
                stack = stack.rechunk({0: -1, 1: 'auto'})

            if self.store_sensitivities == 'forward_only':

                with ProgressBar():
                    print("Forward calculation: ")
                    pred = array.dot(stack, self.model).compute()

                return pred

            else:
                if os.path.exists(self.sensitivity_path):

                    kernel = array.from_zarr(self.sensitivity_path)

                    if np.all(np.r_[
                            np.any(np.r_[kernel.chunks[0]] == stack.chunks[0]),
                            np.any(np.r_[kernel.chunks[1]] == stack.chunks[1]),
                            np.r_[kernel.shape] == np.r_[stack.shape]]):
                        # Check that loaded kernel matches supplied data and mesh
                        print("Zarr file detected with same shape and chunksize ... re-loading")

                        return kernel
                    else:
                        print("Zarr file detected with wrong shape and chunksize ... over-writing")

                with ProgressBar():
                    print("Saving kernel to zarr: " + self.sensitivity_path)
                    kernel = array.to_zarr(stack, self.sensitivity_path, compute=True, return_stored=True, overwrite=True)

        else:
            # TODO
            # Process in parallel using multiprocessing
            # pool = multiprocessing.Pool(self.n_cpu)
            # kernel = pool.map(
            #   self.evaluate_integral, [
            #       receiver for receiver in self.survey.receiver_locations.tolist()
            # ])
            # pool.close()
            # pool.join()

            # Single threaded
            kernel = np.vstack([
                self.evaluate_integral(receiver, components[component])
                for receiver, component in zip(self.survey.receiver_locations.tolist(), active_components)
            ])

        return kernel

    def evaluate_integral(self):
        """
        evaluate_integral

        Compute the forward linear relationship between the model and the physics at a point.
        :param self:
        :return:
        """

        raise RuntimeError(f"Integral calculations must implemented by the subclass {self}.")
