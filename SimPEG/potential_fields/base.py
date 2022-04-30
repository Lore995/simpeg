import os
# import properties
import numpy as np
import multiprocessing
from ..simulation import LinearSimulation
from scipy.sparse import csr_matrix as csr
from SimPEG.utils import mkvc

###############################################################################
#                                                                             #
#                             Base Potential Fields Simulation                #
#                                                                             #
###############################################################################


class BasePFSimulation(LinearSimulation):
    r"""Base class for potential field simulations that use integral formulations.

    For integral formulations, the forward simulation for a set of voxel cells
    can be defined as a linear operation of the form:

    .. math::
        \mathbf{d} = \mathbf{Am}

    where :math:`\mathbf{d}` are the data, :math:`\mathbf{m}` are the model paramters
    and :math:`\mathbf{A}` is a linear operator defining the sensitivities. The primary
    difference between child simulation classes is the kernel function used to create
    the rows of :math:`\mathbf{A}`.

    Parameters
    ----------
    mesh : discretize.TensorMesh or discretize.TreeMesh
        A 3D tensor or tree mesh.
    ind_active : np.ndarray of int or bool
        Indices array denoting the active topography cells.
    store_sensitivities : str {'ram', 'disk', 'forward_only'}, Default='ram'
        Options for storing sensitivities. There are 3 options

        - 'ram': sensitivities are stored in the computer's RAM
        - 'disk': sensitivities are written to a directory
        - 'forward_only': you intend only do perform a forward simulation and sensitivities do no need to be stored

    """
    # actInd = properties.Array(
    #     "Array of active cells (ground)", dtype=(bool, int), default=None
    # )

    def __init__(self, mesh, ind_active=None, store_sensitivities='ram', **kwargs):

        # If deprecated property set with kwargs
        if "actInd" in kwargs:
            ind_active = kwargs.pop("actInd")
        if ind_active is not None:
            self._ind_active = ind_active

        if "forwardOnly" in kwargs:
            store_sensitivities = kwargs.pop("forwardOnly")
            if store_sensitivities:
                self.store_sensitivities = 'forward_only'
        else:
            self.store_sensitivities = store_sensitivities

        if "n_cpu" in kwargs:
            del kwargs["n_cpu"]
            warnings.warn(
                "n_cpu has been deprecated. If interested, try out "
                "loading dask for parallelism by doing ``import SimPEG.dask``. "
                "This will be removed in version 0.16.0 of SimPEG",
                FutureWarning,
            )

        LinearSimulation.__init__(self, mesh, **kwargs)
        
        # Find non-zero cells indices
        if getattr(self, "ind_active", None) is not None:
            if self.ind_active.dtype == "bool":
                indices = np.where(self.ind_active)[0]
            else:
                indices = self.ind_active
        else:
            indices = np.asarray(range(self.mesh.nC))

        self.nC = len(indices)

        # Create active cell projector
        projection = csr(
            (np.ones(self.nC), (indices, range(self.nC))), shape=(self.mesh.nC, self.nC)
        )

        # Create vectors of nodal location for the lower and upper corners
        bsw = self.mesh.cell_centers - self.mesh.h_gridded / 2.0
        tne = self.mesh.cell_centers + self.mesh.h_gridded / 2.0

        xn1, xn2 = bsw[:, 0], tne[:, 0]
        yn1, yn2 = bsw[:, 1], tne[:, 1]

        self.Yn = projection.T * np.c_[mkvc(yn1), mkvc(yn2)]
        self.Xn = projection.T * np.c_[mkvc(xn1), mkvc(xn2)]

        # Allows for 2D mesh where Zn is defined by user
        if self.mesh.dim > 2:
            zn1, zn2 = bsw[:, 2], tne[:, 2]
            self.Zn = projection.T * np.c_[mkvc(zn1), mkvc(zn2)]

    @property
    def store_sensitivities(self):
        """Options for storing sensitivities.

        There are 3 options:

        - 'ram': sensitivity matrix stored in RAM
        - 'disk': sensitivities written and stored to disk
        - 'forward_only': sensitivities are not store (only use for forward simulation)

        Returns
        -------
        str
            A string defining the model type for the simulation.
            One of {'disk', 'ram', 'forward_only'}.
        """
        if self._store_sensitivities is None:
            self._store_sensitivities = 'ram'
        return self._store_sensitivities

    @store_sensitivities.setter
    def store_sensitivities(self, value):
        choices = ["disk", "ram", "forward_only"]
        value = value.lower()
        if value not in choices:
            raise ValueError(
                "Store sensitivities option ({}) unrecognized. ",
                "Choose one of ['disk', 'ram', 'forward_only']".format(value)
            )
        self._store_sensitivities = value

    @property
    def ind_active(self):
        """Active topography cells

        Returns
        -------
        (n_cell) numpy.ndarray of bool
            Returns the active topography cells
        """
        if self._ind_active is None:
            self._ind_active = np.asarray(range(self.mesh.nC))
        return self._ind_active

    @ind_active.setter
    def ind_active(self, input_array):
        if not isinstance(input_array, (tuple, list, np.ndarray)):
            raise TypeError(
                "'ind_active' must be set using a tuple, list or numpy.ndarray"
            )
        if isinstance(input_array, (tuple, list)):
            input_array = np.array(input_array)
        if not isinstance(input_array.dtype, (int, bool)):
            raise TypeError("'ind_active' must be an array of int or bool")

        self._ind_active = input_array

    @property
    def actInd(self):
        """'actInd' is deprecated. Use 'ind_active' instead."""
        warnings.warn(
            "The 'actInd' property has been deprecated. "
            "Please use 'ind_active'. This will be removed in version 0.17.0 of SimPEG.",
            FutureWarning,
        )
        return self._ind_active

    @actInd.setter
    def actInd(self, value):
        warnings.warn(
            "The 'actInd' property has been deprecated. "
            "Please use 'ind_active'. This will be removed in version 0.17.0 of SimPEG.",
            FutureWarning,
        )
        self.ind_active(value)

    

    def linear_operator(self):
        """Return linear operator

        Returns
        -------
        numpy.ndarray
            Linear operator
        """
        self.nC = self.modelMap.shape[0]

        components = np.array(list(self.survey.components.keys()))
        active_components = np.hstack(
            [np.c_[values] for values in self.survey.components.values()]
        ).tolist()
        nD = self.survey.nD

        if self.store_sensitivities == "disk":
            sens_name = self.sensitivity_path + "sensitivity.npy"
            if os.path.exists(sens_name):
                # do not pull array completely into ram, just need to check the size
                kernel = np.load(sens_name, mmap_mode="r")
                if kernel.shape == (nD, self.nC):
                    print(f"Found sensitivity file at {sens_name} with expected shape")
                    kernel = np.asarray(kernel)
                    return kernel
        # Single threaded
        if self.store_sensitivities != "forward_only":
            kernel = np.vstack(
                [
                    self.evaluate_integral(receiver, components[component])
                    for receiver, component in zip(
                        self.survey.receiver_locations.tolist(), active_components
                    )
                ]
            )
        else:
            kernel = np.hstack(
                [
                    self.evaluate_integral(receiver, components[component]).dot(
                        self.model
                    )
                    for receiver, component in zip(
                        self.survey.receiver_locations.tolist(), active_components
                    )
                ]
            )
        if self.store_sensitivities == "disk":
            print(f"writing sensitivity to {sens_name}")
            os.makedirs(self.sensitivity_path, exist_ok=True)
            np.save(sens_name, kernel)
        return kernel

    def evaluate_integral(self):
        """'evaluate_integral' method no longer implemented for *BaseSimulation* class."""

        raise RuntimeError(
            f"Integral calculations must implemented by the subclass {self}."
        )

    @property
    def forwardOnly(self):
        """The forwardOnly property has been removed. Please set the store_sensitivites
        property instead.
        """
        raise TypeError(
            "The forwardOnly property has been removed. Please set the store_sensitivites "
            "property instead."
        )

    @forwardOnly.setter
    def forwardOnly(self, other):
        raise TypeError(
            "The forwardOnly property has been removed. Please set the store_sensitivites "
            "property instead."
        )

    @property
    def parallelized(self):
        """The parallelized property has been removed. If interested, try out
        loading dask for parallelism by doing ``import SimPEG.dask``.
        """
        raise TypeError(
            "parallelized has been removed. If interested, try out "
            "loading dask for parallelism by doing ``import SimPEG.dask``. "
        )

    @parallelized.setter
    def parallelized(self, other):
        raise TypeError(
            "Do not set parallelized. If interested, try out "
            "loading dask for parallelism by doing ``import SimPEG.dask``."
        )

    @property
    def n_cpu(self):
        """The parallelized property has been removed. If interested, try out
        loading dask for parallelism by doing ``import SimPEG.dask``.
        """
        raise TypeError(
            "n_cpu has been removed. If interested, try out "
            "loading dask for parallelism by doing ``import SimPEG.dask``."
        )

    @parallelized.setter
    def n_cpu(self, other):
        raise TypeError(
            "Do not set n_cpu. If interested, try out "
            "loading dask for parallelism by doing ``import SimPEG.dask``."
        )


def progress(iter, prog, final):
    """Progress (% complete) for constructing sensitivity matrix

    Parameters
    ----------
    iter : int
        Current rows
    prog : float
        Progress
    final : int
        Number of rows (= number of receivers)
    
    Returns
    -------
    float
        % completed
    """
    arg = np.floor(float(iter) / float(final) * 10.0)

    if arg > prog:

        print("Done " + str(arg * 10) + " %")
        prog = arg

    return prog


def get_dist_wgt(mesh, receiver_locations, actv, R, R0):
    """Compute distance weights for potential field simulations

    Parameters
    ----------
    mesh : discretize.BaseMesh
        A discretize mesh
    receiver_locations : (n, 3) numpy.ndarray
        Observation locations [x, y, z]
    actv : (n_cell) numpy.ndarray of bool
        Active cells vector [0:air , 1: ground]
    R : float
        Decay factor (mag=3, grav =2)
    R0 : float
        Stabilization factor. Usually a fraction of the minimum cell size

    Returns
    wr : (n_cell) numpy.ndarray
        Distance weighting model; 0 for all inactive cells
    """

    # Find non-zero cells
    if actv.dtype == "bool":
        inds = (
            np.asarray([inds for inds, elem in enumerate(actv, 1) if elem], dtype=int)
            - 1
        )
    else:
        inds = actv

    nC = len(inds)

    # Create active cell projector
    P = csr((np.ones(nC), (inds, range(nC))), shape=(mesh.nC, nC))

    # Geometrical constant
    p = 1 / np.sqrt(3)

    # Create cell center location
    Ym, Xm, Zm = np.meshgrid(mesh.vectorCCy, mesh.vectorCCx, mesh.vectorCCz)
    hY, hX, hZ = np.meshgrid(mesh.hy, mesh.hx, mesh.hz)

    # Remove air cells
    Xm = P.T * mkvc(Xm)
    Ym = P.T * mkvc(Ym)
    Zm = P.T * mkvc(Zm)

    hX = P.T * mkvc(hX)
    hY = P.T * mkvc(hY)
    hZ = P.T * mkvc(hZ)

    V = P.T * mkvc(mesh.vol)
    wr = np.zeros(nC)

    ndata = receiver_locations.shape[0]
    count = -1
    print("Begin calculation of distance weighting for R= " + str(R))

    for dd in range(ndata):

        nx1 = (Xm - hX * p - receiver_locations[dd, 0]) ** 2
        nx2 = (Xm + hX * p - receiver_locations[dd, 0]) ** 2

        ny1 = (Ym - hY * p - receiver_locations[dd, 1]) ** 2
        ny2 = (Ym + hY * p - receiver_locations[dd, 1]) ** 2

        nz1 = (Zm - hZ * p - receiver_locations[dd, 2]) ** 2
        nz2 = (Zm + hZ * p - receiver_locations[dd, 2]) ** 2

        R1 = np.sqrt(nx1 + ny1 + nz1)
        R2 = np.sqrt(nx1 + ny1 + nz2)
        R3 = np.sqrt(nx2 + ny1 + nz1)
        R4 = np.sqrt(nx2 + ny1 + nz2)
        R5 = np.sqrt(nx1 + ny2 + nz1)
        R6 = np.sqrt(nx1 + ny2 + nz2)
        R7 = np.sqrt(nx2 + ny2 + nz1)
        R8 = np.sqrt(nx2 + ny2 + nz2)

        temp = (
            (R1 + R0) ** -R
            + (R2 + R0) ** -R
            + (R3 + R0) ** -R
            + (R4 + R0) ** -R
            + (R5 + R0) ** -R
            + (R6 + R0) ** -R
            + (R7 + R0) ** -R
            + (R8 + R0) ** -R
        )

        wr = wr + (V * temp / 8.0) ** 2.0

        count = progress(dd, count, ndata)

    wr = np.sqrt(wr) / V
    wr = mkvc(wr)
    wr = np.sqrt(wr / (np.max(wr)))

    print("Done 100% ...distance weighting completed!!\n")

    return wr
