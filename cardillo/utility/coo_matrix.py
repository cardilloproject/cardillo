import warnings
from scipy.sparse import csc_array, csr_array, coo_array
from scipy.sparse.sputils import isshape, check_shape
from numpy import repeat, tile
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
        self.__data = array("d", [])  # double
        self.__row = array("I", [])  # unsigned int
        self.__col = array("I", [])  # unsigned int

    def __setitem__(self, key, value):
        # None is returned by every function that does not return. Hence, we
        # can use this to add no contribution to the matrix.
        if value is not None:
            rows, cols = key

            # extend arrays from given CooMatrix
            if isinstance(value, CooMatrix):
                self.__data.extend(value.__data)
                self.__row.extend(value.__row)
                self.__col.extend(value.__col)
            else:
                ndim = value.ndim
                if ndim > 1:
                    # 2D array
                    # - fast version
                    self.__data.fromlist(value.ravel(order="C").tolist())
                    self.__row.fromlist(repeat(rows, len(cols)).tolist())
                    self.__col.fromlist(tile(cols, len(rows)).tolist())
                    # - slow version
                    # self.__data.extend(value.ravel(order="C"))
                    # self.__row.extend(repeat(rows, len(cols)))
                    # self.__col.extend(tile(cols, len(rows)))
                else:
                    # 1D array
                    self.__data.fromlist(value.tolist())
                    self.__row.fromlist(rows.tolist())
                    self.__col.fromlist(cols.tolist())

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
        return scipy_matrix(
            (self.__data, (self.__row, self.__col)), shape=self.shape, copy=copy
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

    @profile(entries=50)
    def run(local_size, nlocal):
        global_size = nlocal * local_size
        coo = CooMatrix((global_size, global_size))

        for i in range(nlocal):
            dense = np.random.rand(local_size, local_size)
            idx = np.arange(i * local_size, (i + 1) * local_size)
            coo[idx, idx] = dense

        return coo.tocsr()
        # return coo.asformat("coo")
        # return coo.asformat("csr")

    run(100, 1000)
