import warnings
from scipy.sparse import csc_array, csr_array, coo_array
from scipy.sparse._sputils import isshape, check_shape
from scipy.sparse import spmatrix, sparray
import numpy as np
from numpy import repeat, tile, atleast_1d, atleast_2d, arange, ndarray
from array import array


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

        # python array as efficient container for numerical data,
        # see https://docs.python.org/3/library/array.html
        self.data = np.empty(0, dtype=float)  # double
        self.row = np.empty(0, dtype=int)  # unsigned int
        self.col = np.empty(0, dtype=int)  # unsigned int

        self._data_index = {}
        self._value_type = {}
        self._scipy_coo = None

    def __setitem__(self, key, value):
        # None is returned by every function that does not return. Hence, we
        # can use this to add no contribution to the matrix.
        if value is not None:
            if len(key) == 3:
                # extract rows and columns to assign
                name, rows, cols = key
                pre_allocate = name in self._data_index.keys()
            elif len(key) == 2:
                # extract rows and columns to assign
                rows, cols = key
                pre_allocate = False
            else:
                raise NotImplementedError

            if pre_allocate:
                value_type = self._value_type[name]
            else:
                if isinstance(rows, slice):
                    rows = arange(*rows.indices(self.shape[0]))
                if isinstance(cols, slice):
                    cols = arange(*cols.indices(self.shape[1]))
                rows = atleast_1d(rows)
                cols = atleast_1d(cols)

                if isinstance(value, CooMatrix):
                    value_type = "Coo"
                elif isinstance(value, sparray):
                    value_type = "sparse"
                elif isinstance(value, spmatrix):
                    raise RuntimeError(
                        "Do not use sparse matrices, move to sparse array."
                    )
                elif isinstance(value, ndarray):
                    value_type = "ndarray"
                elif isinstance(value, (int, float)):
                    value_type = "digit"
                else:
                    raise NotImplementedError
                if len(key) == 3:
                    self._value_type[name] = value_type

            if value_type == "Coo":
                # assert value.shape == (len(rows), len(cols)), "inconsistent assignment"

                # extend arrays from given CooMatrix
                new_data = value.data
                if not pre_allocate:
                    new_rows = rows[value.row]
                    new_cols = cols[value.col]
                # TODO: benchmark
                # self.data.fromlist(value.data.tolist())
                # self.row.fromlist(rows[value.row].tolist())
                # self.col.fromlist(cols[value.col].tolist())
            elif value_type == "sparse":
                # assert value.shape == (len(rows), len(cols)), "inconsistent assignment"

                # all scipy sparse matrices are converted to coo_array, their
                # data, row and column lists are subsequently appended
                coo = value.tocoo()
                new_data = coo.data
                if not pre_allocate:
                    new_rows = rows[coo.row]
                    new_cols = cols[coo.col]
                # TODO: benchmark
                # self.data.fromlist(coo.data.tolist())
                # self.row.fromlist(rows[coo.row].tolist())
                # self.col.fromlist(cols[coo.col].tolist())
            elif value_type == "ndarray":
                # convert to 2D numpy arrays
                # value = atleast_2d(value)
                # assert value.shape == (len(rows), len(cols)), "inconsistent assignment"

                # 2D array
                new_data = value.ravel(order="C")
                if not pre_allocate:
                    new_rows = rows.repeat(len(cols))
                    new_cols = tile(cols, len(rows))
            elif value_type == "digit":
                new_rows = rows
                new_cols = cols
                new_data = np.array([value])
            else:
                raise NotImplementedError

            if pre_allocate:
                id0, id1 = self._data_index[name]
                self.data[id0:id1] = new_data
            else:
                self.data = np.concatenate([self.data, new_data])
                self.col = np.concatenate([self.col, new_cols])
                self.row = np.concatenate([self.row, new_rows])
                if len(key) == 3:
                    self._data_index[name] = (
                        len(self.data) - len(new_data),
                        len(self.data),
                    )

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
        if format == "Coo":
            return self
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
        return scipy_matrix(
            (self.data, (self.row, self.col)), shape=self.shape, copy=copy
        )

    def tocoo(self, copy=False):
        """Convert container to scipy coo_array."""
        if copy:
            coo = self.tosparse(coo_array, copy=True)
        elif self._scipy_coo is None:
            coo = self.tosparse(coo_array, copy=False)
            self._scipy_coo = coo
        else:
            coo = self._scipy_coo
        return coo

    def tocsc(self, copy=False):
        """Convert container to scipy csc_array."""
        return self.tosparse(csc_array, copy=copy)

    def tocsr(self, copy=False):
        """Convert container to scipy csr_array."""
        return self.tosparse(csr_array, copy=copy)

    def toarray(self, copy=False):
        """Convert container to 2D numpy array."""
        return self.tocoo(copy).toarray()

    def transpose(self, copy=False):
        ret = CooMatrix((self.shape[1], self.shape[0]))
        if copy:
            ret.row = self.col.copy()
            ret.col = self.row.copy()
            ret.data = self.data.copy()
        else:
            ret.row = self.col
            ret.col = self.row
            ret.data = self.data
        return ret

    @property
    def T(self):
        return self.transpose(copy=False)

    def __neg__(self):
        ret = CooMatrix(self.shape)
        ret.row = self.row.copy()
        ret.col = self.col.copy()
        ret.data = -self.data
        return ret


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
            dense_coo.data = array("d", dense.data)
            dense_coo.row = array("I", dense.row)
            dense_coo.col = array("I", dense.col)
            idx = np.arange(i * local_size, (i + 1) * local_size)
            coo[idx, idx] = dense_coo

        return coo.tocsr()

    run_dense_matrix()
    run_dense_vector()
    run_scipy_sparse()
    run_coo_sparse()
