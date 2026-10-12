import warnings
from scipy.sparse import csc_array, csr_array, coo_array
from scipy.sparse._sputils import isshape, check_shape
from scipy.sparse import spmatrix, sparray
from numpy import (
    repeat,
    tile,
    atleast_1d,
    atleast_2d,
    arange,
    asarray,
    concatenate,
    empty,
    int32,
    ndarray,
)


class CooMatrix:
    """Small container storing the sparse matrix shape and three lists for
    accumulating the entries for row, column and data Wiki/COO.

    Parameters
    ----------
    shape : tuple, 2D
        tuple defining the shape of the matrix

    References
    ----------
    Wiki/COO: https://en.wikipedia.org/wiki/Sparse_matrix#Coordinate_list_(COO)
    """

    def __init__(self, shape):
        # check shape input
        if (
            type(shape) is tuple
            and len(shape) == 2
            and type(shape[0]) is int
            and type(shape[1]) is int
            and shape[0] >= 0
            and shape[1] >= 0
        ):
            # fast path for the common case of a tuple of two non-negative ints
            self.shape = shape
            self.__data = []
            self.__row = []
            self.__col = []
            return
        if isinstance(shape, tuple):
            pass
        else:
            try:
                shape = tuple(shape)
            except Exception:
                raise ValueError(
                    "input argument shape is not tuple or cannot be interpreted as tuple"
                )

        # see https://github.com/scipy/scipy/blob/adc4f4f7bab120ccfab9383aba272954a0a12fb0/scipy/sparse/sputils.py#L210
        if isshape(shape, nonneg=True):
            M, N = shape
            # see https://github.com/scipy/scipy/blob/adc4f4f7bab120ccfab9383aba272954a0a12fb0/scipy/sparse/sputils.py#L267
            self.shape = check_shape((M, N))
        else:
            raise TypeError(
                "input argument shape cannot be interpreted as correct shape"
            )

        # Entries are stored as lists of numpy chunks (one per assigned block)
        # that are concatenated lazily. This is much cheaper than extending
        # python arrays element by element for many small blocks.
        self.__data = []
        self.__row = []
        self.__col = []

    @staticmethod
    def _consolidate(chunks, dtype=None):
        if len(chunks) == 0:
            return empty(0, dtype=dtype if dtype is not None else float)
        if len(chunks) == 1:
            return chunks[0]
        chunks[:] = [concatenate(chunks)]
        return chunks[0]

    @property
    def data(self):
        return self._consolidate(self.__data)

    @data.setter
    def data(self, value):
        self.__data = [asarray(value)]

    @property
    def row(self):
        return self._consolidate(self.__row, int32)

    @row.setter
    def row(self, value):
        self.__row = [asarray(value)]

    @property
    def col(self):
        return self._consolidate(self.__col, int32)

    @col.setter
    def col(self, value):
        self.__col = [asarray(value)]

    def __setitem__(self, key, value):
        # None is returned by every function that does not return. Hence, we
        # can use this to add no contribution to the matrix.
        if value is not None:
            # extract rows and columns to assign
            rows, cols = key
            if isinstance(rows, slice):
                rows = arange(*rows.indices(self.shape[0]))
            if isinstance(cols, slice):
                cols = arange(*cols.indices(self.shape[1]))
            rows = atleast_1d(rows)
            cols = atleast_1d(cols)

            if isinstance(value, ndarray):
                if value.ndim != 2:
                    value = atleast_2d(value)
                assert value.shape == (len(rows), len(cols)), "inconsistent assignment"

                # 2D array, flatten() always returns a copy
                self.__data.append(value.flatten(order="C"))
                self.__row.append(repeat(rows, len(cols)))
                self.__col.append(tile(cols, len(rows)))
            elif isinstance(value, CooMatrix):
                assert value.shape == (len(rows), len(cols)), "inconsistent assignment"

                # append chunks from given CooMatrix
                self.__data.append(value.data.copy())
                self.__row.append(rows[value.row])
                self.__col.append(cols[value.col])
            elif isinstance(value, sparray):
                assert value.shape == (len(rows), len(cols)), "inconsistent assignment"

                # all scipy sparse matrices are converted to coo_array, their
                # data, row and column lists are subsequently appended
                coo = value.tocoo()
                self.__data.append(coo.data.copy())
                self.__row.append(rows[coo.row])
                self.__col.append(cols[coo.col])
            elif isinstance(value, spmatrix):
                raise RuntimeError("Do not use sparse matrices, move to sparse array.")
            else:
                # convert everything else (scalars, lists, ...) to 2D numpy arrays
                value = atleast_2d(value)
                assert value.shape == (len(rows), len(cols)), "inconsistent assignment"

                self.__data.append(value.flatten(order="C"))
                self.__row.append(repeat(rows, len(cols)))
                self.__col.append(tile(cols, len(rows)))

    def extend(self, matrix, DOF):
        warnings.warn(
            "Usage of `CooMatrix.extend` is deprecated. "
            "You can simply index the object, e.g., coo[rows, cols] = value",
            category=DeprecationWarning,
        )
        self[DOF[0], DOF[1]] = matrix

    def asformat(self, format, copy=False):
        """Return this matrix in the passed format.
        Parameters
        ----------
        format : {str, None}
            The desired matrix format ("csr", "csc", "lil", "dok", "array", ...)
            or None for no conversion.
        copy : bool, optional
            If True, the result is guaranteed to not share data with self.
        Returns
        -------
        A : This matrix in the passed format.
        """
        try:
            convert_method = getattr(self, "to" + format)
        except AttributeError as e:
            raise ValueError("Format {} is unknown.".format(format)) from e

        # Forward the copy kwarg, if it's accepted.
        try:
            return convert_method(copy=copy)
        except TypeError:
            return convert_method()

    def tosparse(self, scipy_matrix, copy=False):
        """Convert container to scipy sparse matrix.

        Parameters
        ----------
        scipy_matrix: scipy.sparse.spmatrix
            scipy sparse matrix format that should be returned
        """
        if not any(len(chunk) for chunk in self.__data):
            # constructing an empty matrix from its shape is much cheaper
            return scipy_matrix(self.shape, dtype=float)
        return scipy_matrix(
            (self.data, (self.row, self.col)), shape=self.shape, copy=copy
        )

    def tocoo(self, copy=False):
        """Convert container to scipy coo_array."""
        return self.tosparse(coo_array, copy=copy)

    def tocsc(self, copy=False):
        """Convert container to scipy csc_array."""
        return self.tosparse(csc_array, copy=copy)

    def tocsr(self, copy=False):
        """Convert container to scipy csr_array."""
        return self.tosparse(csr_array, copy=copy)

    def toarray(self, copy=False):
        """Convert container to 2D numpy array."""
        return self.tocoo(copy).toarray()


if __name__ == "__main__":
    from profilehooks import profile
    import numpy as np
    from scipy.sparse import random

    entries = 1
    density = 1
    local_size = 10
    nlocal = 100

    @profile(entries=entries)
    def run_dense_matrix():
        global_size = nlocal * local_size
        coo = CooMatrix((global_size, global_size))

        for i in range(nlocal):
            dense = np.random.rand(local_size, local_size)
            idx = np.arange(i * local_size, (i + 1) * local_size)
            coo[idx, idx] = dense

        return coo.tocsr()

    @profile(entries=entries)
    def run_dense_vector():
        global_size = nlocal * local_size
        coo = CooMatrix((global_size, global_size))

        for i in range(nlocal):
            dense = np.random.rand(local_size)
            idx = np.arange(i * local_size, (i + 1) * local_size)
            coo[idx, idx] = dense

        return coo.tocsr()

    @profile(entries=entries)
    def run_scipy_sparse():
        global_size = nlocal * local_size
        coo = CooMatrix((global_size, global_size))

        for i in range(nlocal):
            dense = random(local_size, local_size, density=density)
            idx = np.arange(i * local_size, (i + 1) * local_size)
            coo[idx, idx] = dense

        return coo.tocsr()

    @profile(entries=entries)
    def run_coo_sparse():
        global_size = nlocal * local_size
        coo = CooMatrix((global_size, global_size))

        for i in range(nlocal):
            dense = random(local_size, local_size, density=density)
            dense_coo = CooMatrix((local_size, local_size))
            dense_coo.data = dense.data
            dense_coo.row = dense.row
            dense_coo.col = dense.col
            idx = np.arange(i * local_size, (i + 1) * local_size)
            coo[idx, idx] = dense_coo

        return coo.tocsr()

    run_dense_matrix()
    run_dense_vector()
    run_scipy_sparse()
    run_coo_sparse()
