from scipy.sparse import csc_matrix, csr_matrix, coo_matrix, lil_array, coo_array
from scipy.sparse.sputils import isshape, check_shape
from numpy import repeat, tile, asarray, append
from array import array


class OldCoo:
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

        # initialize empty lists for row, columns and data
        self.row = []
        self.col = []
        self.data = []

    def extend(self, matrix, DOF):
        """Extend COO matrix with data given by `matrix` and indices stored
        in the tuple `DOF` containing two arrays (row indices and column
        indices).
        """
        array = matrix.ravel(order="C")
        row = repeat(DOF[0], DOF[1].size)
        col = tile(DOF[1], DOF[0].size)

        # do not assemble zero elements
        nnz_mask = array.nonzero()[0]
        self.data.extend(array[nnz_mask].tolist())
        self.row.extend(row[nnz_mask].tolist())
        self.col.extend(col[nnz_mask].tolist())

    def __setitem__(self, key, value):
        rows, cols = key
        self.extend(value, key)

    def extend_diag(self, array, DOF):
        """Extend COO matrix with diagonal matrix (diagonal elements stored
        in the input `array` and indices stored in the `DOF` array).
        """
        self.data.extend(array.tolist())
        self.row.extend(DOF[0].tolist())
        self.col.extend(DOF[1].tolist())

    def extend_sparse(self, coo):
        """Extend COO matrix with other COO matrix."""
        self.data.extend(coo.data)
        self.row.extend(coo.row)
        self.col.extend(coo.col)

    def tosparse(self, scipy_matrix):
        """Convert container to scipy sparse matrix.
        Parameters
        ----------
        scipy_matrix: scipy.sparse.spmatrix
            scipy sparse matrix format that should be returned
        """
        return scipy_matrix((self.data, (self.row, self.col)), shape=self.shape)

    def tocsc(self):
        """Convert container to scipy csc_matrix."""
        return self.tosparse(csc_matrix)

    def tocsr(self):
        """Convert container to scipy csr_matrix."""
        return self.tosparse(csr_matrix)

    def tocoo(self):
        """Convert container to scipy coo_matrix."""
        return self.tosparse(coo_matrix)

    def toarray(self):
        """Convert container to 2D numpy array."""
        return self.tocoo().toarray()


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
        rows, cols = key
        ndim = value.ndim
        if ndim > 1:
            if isinstance(value, CooMatrix):
                self.__data.extend(value.data)
                self.__row.extend(value.row)
                self.__col.extend(value.col)
            else:
                # dense matrix
                # - fast version
                self.__data.fromlist(value.ravel(order="C").tolist())
                self.__row.fromlist(repeat(rows, len(cols)).tolist())
                self.__col.fromlist(tile(cols, len(rows)).tolist())
                # - slow version
                # self.__data.extend(value.ravel(order="C"))
                # self.__row.extend(repeat(rows, len(cols)))
                # self.__col.extend(tile(cols, len(rows)))
        else:
            # array (we assume diagonal matrix)
            self.__data.fromlist(value.tolist())
            self.__row.fromlist(rows.tolist())
            self.__col.fromlist(cols.tolist())

    def extend(self, matrix, DOF):
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
        """Convert container to scipy coo_matrix."""
        return self.tosparse(coo_matrix, copy=copy)

    def tocsc(self, copy=False):
        """Convert container to scipy csc_matrix."""
        return self.tosparse(csc_matrix, copy=copy)

    def tocsr(self, copy=False):
        """Convert container to scipy csr_matrix."""
        return self.tosparse(csr_matrix, copy=copy)

    def toarray(self, copy=False):
        """Convert container to 2D numpy array."""
        return self.tocoo(copy).toarray()


class Lil:
    def __init__(self, shape) -> None:
        self.lil = lil_array(shape)

    def extend(self, matrix, DOF):
        self.lil[DOF[0], tile(DOF[1], (len(DOF[0]), 1)).T] = matrix

    def __setitem__(self, key, value):
        rows, cols = key
        self.lil[rows, tile(cols, (len(rows), 1)).T] = value
        # ndim = value.ndim
        # if ndim > 1:
        #     if isinstance(value, CooMatrix):
        #         self.__data.extend(value.data)
        #         self.__row.extend(value.row)
        #         self.__col.extend(value.col)
        #     else:
        #         # dense matrix
        #         self.__data.extend(value.ravel(order="C"))
        #         self.__row.extend(repeat(rows, len(cols)))
        #         self.__col.extend(tile(cols, len(rows)))
        # else:
        #     # array (we assume diagonal matrix)
        #     self.__data.extend(value.tolist())
        #     self.__row.extend(rows.tolist())
        #     self.__col.extend(cols.tolist())

    def tosparse(self, scipy_matrix: str):
        """Convert container to scipy sparse matrix.
        Parameters
        ----------
        scipy_matrix: scipy.sparse.spmatrix
            scipy sparse matrix format that should be returned
        """
        if not isinstance(scipy_matrix, str):
            scipy_matrix = scipy_matrix.__name__.strip("matrix").strip("_")
        return self.lil.asformat(scipy_matrix, copy=False)

    def tocsc(self):
        """Convert container to scipy csc_matrix."""
        return self.tosparse("csc")

    def tocsr(self):
        """Convert container to scipy csr_matrix."""
        return self.tosparse("csr")

    def tocoo(self):
        """Convert container to scipy coo_matrix."""
        return self.tosparse("coo")

    def toarray(self):
        """Convert container to 2D numpy array."""
        return self.tosparse("array")


class ScipyCoo(coo_matrix):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def extend(self, matrix, DOF):
        """Extend coo_matrix with data given in `matrix` and indices stored in the tuple `DOF` containing two arrays.

        Parameters
        ----------
        matrix: numpy.ndarray, 2D
            dense matrix which has to be stored
        DOF : tuple, 2D
            tuple defining the global row and column indices of the dense matrix
        """
        self.data = append(self.data, matrix.ravel(order="C"))
        self.row = append(self.row, repeat(DOF[0], DOF[1].size))
        self.col = append(self.col, tile(DOF[1], DOF[0].size))

        # TODO: slow in python
        # data = []
        # row = []
        # col = []
        # for i, row_i in enumerate(DOF[0]):
        #     for j, col_j in enumerate(DOF[1]):
        #         data.append(matrix[i, j])
        #         row.append(row_i)
        #         col.append(col_j)

        # self.data = append( self.data, data )
        # self.row = append( self.row, row )
        # self.col = append( self.col, col )

    def __setitem__(self, key, value):
        self.extend(value, key)

    def extend_diag(self, array, DOF):
        """Extend container with diagonal matrix (diagonal elements stored in the input `array` and indices stored in the `DOF` array).

        Parameters
        ----------
        matrix: numpy.ndarray, 2D
            dense matrix which has to be stored
        DOF : tuple, 2D
            tuple defining the global row and column indices of the dense matrix
        """
        self.data = append(self.data, array)
        self.row = append(self.row, DOF[0])
        self.col = append(self.col, DOF[1])

    def extend_sparse(self, sparse_matrix):
        """Extend container with sparse matrix defined by three lists `data`, `row` and `col`.

        Parameters
        ----------
        sparse_matrix: coo_matrix
            scipy coo_matrix
        """
        self.data = append(self.data, sparse_matrix.data)
        self.row = append(self.row, sparse_matrix.row)
        self.col = append(self.col, sparse_matrix.col)

    def tosparse(self, scipy_matrix):
        """Convert extended coo_matrix to another arbitrary scipy sparse matrix.

        Parameters
        ----------
        scipy_matrix: scipy.sparse.spmatrix
            scipy sparse matrix format that should be returned
        """
        return scipy_matrix(self)


class OurCoo2(coo_array):
    def __init__(self, arg1, shape=None, dtype=None, copy=False):
        super().__init__(arg1, shape, dtype, copy)
        # self.__data = self.data.tolist()
        # self.__row = self.row.tolist()
        # self.__col = self.col.tolist()
        self.__data = array("d", [])
        self.__row = array("i", [])
        self.__col = array("i", [])

    @property
    def data(self):
        return asarray(self.__data, dtype=float)

    @data.setter
    def data(self, value):
        # self.__data = value.tolist()
        # self.__data = array("d", value.tolist())
        self.__data = array("d", value)

    @property
    def row(self):
        return asarray(self.__row, dtype=int)

    @row.setter
    def row(self, value):
        # self.__row = value.tolist()
        self.__row = array("i", value)

    @property
    def col(self):
        return asarray(self.__col, dtype=int)

    @col.setter
    def col(self, value):
        # self.__col= value.tolist()
        self.__col = array("i", value)

    def __extend(self, matrix, DOF):
        """Extend COO matrix with data given by `matrix` and indices stored
        in the tuple `DOF` containing two arrays (row indices and column
        indices).
        """
        # flat_matrix = matrix.ravel(order="C")
        # row = repeat(DOF[0], len(DOF[1]))
        # col = tile(DOF[1], len(DOF[0]))

        # self.__data.extend(flat_matrix.tolist())
        # self.__row.extend(row.tolist())
        # self.__col.extend(col.tolist())

        self.__data.extend(matrix.ravel(order="C"))
        self.__row.extend(repeat(DOF[0], len(DOF[1])))
        self.__col.extend(tile(DOF[1], len(DOF[0])))

    def __setitem__(self, key, value):
        self.__extend(value, key)


# CooMatrix = OldCoo
# CooMatrix = CooMatrix
# CooMatrix = Lil
# CooMatrix = ScipyCoo
# CooMatrix = OurCoo2


if __name__ == "__main__":
    from profilehooks import profile
    import numpy as np

    @profile(entries=10)
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
