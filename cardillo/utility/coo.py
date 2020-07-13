from scipy.sparse import csc_matrix, csr_matrix, coo_matrix
from scipy.sparse.sputils import isshape, check_shape
from numpy import repeat, tile
from numpy import append

# class Coo(object):
#     """Small container storing the sparse matrix shape and three lists for accumulating the entries for row, column and data [1]_.

#         Parameters
#         ----------
#         shape : tuple, 2D
#             tuple defining the shape of the matrix

#         References
#         ----------
#         .. [1] https://en.wikipedia.org/wiki/Sparse_matrix#Coordinate_list_(COO)
#     """

#     def __init__(self, shape):
#         # check shape input
#         if isinstance(shape, tuple):
#             pass
#         else:
#             try:
#                 shape = tuple(shape)
#             except Exception:
#                 raise ValueError('input argument shape is not tuple or cannot be interpreted as tuple')

#         # see https://github.com/scipy/scipy/blob/adc4f4f7bab120ccfab9383aba272954a0a12fb0/scipy/sparse/sputils.py#L210
#         if isshape(shape, nonneg=True):
#             M, N = shape
#             # see https://github.com/scipy/scipy/blob/adc4f4f7bab120ccfab9383aba272954a0a12fb0/scipy/sparse/sputils.py#L267
#             self.__shape = check_shape((M, N))
#         else:
#             raise TypeError('input argument shape cannot be interpreted as correct shape')
        
#         # initialice empty lists for row, columns and data
#         self.__row = []
#         self.__col = []
#         self.__data = []

#     @property
#     def shape(self):
#         return self.__shape

#     @property
#     def row(self):
#         return self.__row

#     @property
#     def col(self):
#         return self.__col
        
#     @property
#     def data(self):
#         return self.__data

#     def extend(self, matrix, DOF):
#         """Extend container with data given in `matrix` and indices stored in the tuple `DOF` containing two arrays.

#         Parameters
#         ----------
#         matrix: numpy.ndarray, 2D
#             dense matrix which has to be stored
#         DOF : tuple, 2D
#             tuple defining the global row and column indices of the dense matrix
#         """
#         # TODO: row and column indices can be calculated in the assembler (see old sparse assembler)
#         self.data.extend( matrix.reshape(-1, order='C').tolist() )
#         self.row.extend( repeat(DOF[0], DOF[1].size).tolist() )
#         self.col.extend( tile(DOF[1], DOF[0].size).tolist() )

#         # TODO: slow in python
#         # array = matrix.reshape(-1, order='C')
#         # nnz_mask = array != 0
#         # row = repeat(DOF[0], DOF[1].size)
#         # col = tile(DOF[1], DOF[0].size)
#         # self.data.extend( array[nnz_mask].tolist() )
#         # self.row.extend( row[nnz_mask].tolist() )
#         # self.col.extend( col[nnz_mask].tolist() )

#         # for i, row_i in enumerate(DOF[0]):
#         #     for j, col_j in enumerate(DOF[1]):
#         #         self.data.append(matrix[i, j])
#         #         self.row.append(row_i)
#         #         self.col.append(col_j)

#     def extend_diag(self, array, DOF):
#         """Extend container with diagonal matrix (diagonal elements stored in the input `array` and indices stored in the `DOF` array).

#         Parameters
#         ----------
#         matrix: numpy.ndarray, 2D
#             dense matrix which has to be stored
#         DOF : tuple, 2D
#             tuple defining the global row and column indices of the dense matrix
#         """
#         self.data.extend( array.tolist() )
#         self.row.extend( DOF[0].tolist() )
#         self.col.extend( DOF[1].tolist() )

#     def extend_sparse(self, coo):
#         """Extend container with sparse matrix defined by three lists `data`, `row` and `col`.

#         Parameters
#         ----------
#         coo: Coo
#             Coo container
#         """
#         self.data.extend( coo.data )
#         self.row.extend( coo.row )
#         self.col.extend( coo.col )

#     def tosparse(self, scipy_matrix):
#         """Convert container to scipy sparse matrix.

#         Parameters
#         ----------
#         scipy_matrix: scipy.sparse.spmatrix
#             scipy sparse matrix format that should be returned
#         """
#         return scipy_matrix((self.data, (self.row, self.col)), shape=self.shape)

#     def tocsc(self):
#         """Convert container to scipy csc_matrix.
#         """
#         return self.tosparse(csc_matrix)

#     def tocsr(self):
#         """Convert container to scipy csr_matrix.
#         """
#         return self.tosparse(csr_matrix)

#     def tocoo(self):
#         """Convert container to scipy coo_matrix.
#         """
#         return self.tosparse(coo_matrix)

#     def toarray(self):
#         """Convert container to 2D numpy array.
#         """
#         return self.tocoo().toarray()

class Coo(coo_matrix):
    def __init__(self, shape):
        super().__init__(shape)

    def extend(self, matrix, DOF):
        """Extend coo_matrix with data given in `matrix` and indices stored in the tuple `DOF` containing two arrays.

        Parameters
        ----------
        matrix: numpy.ndarray, 2D
            dense matrix which has to be stored
        DOF : tuple, 2D
            tuple defining the global row and column indices of the dense matrix
        """
        self.data = append( self.data, matrix.reshape(-1, order='C') )
        self.row = append( self.row, repeat(DOF[0], DOF[1].size) )
        self.col = append( self.col, tile(DOF[1], DOF[0].size) )
        
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

    def extend_diag(self, array, DOF):
        """Extend container with diagonal matrix (diagonal elements stored in the input `array` and indices stored in the `DOF` array).

        Parameters
        ----------
        matrix: numpy.ndarray, 2D
            dense matrix which has to be stored
        DOF : tuple, 2D
            tuple defining the global row and column indices of the dense matrix
        """
        self.data = append( self.data, array )
        self.row = append( self.row, DOF[0] )
        self.col = append( self.col, DOF[1] )

    def extend_sparse(self, sparse_matrix):
        """Extend container with sparse matrix defined by three lists `data`, `row` and `col`.

        Parameters
        ----------
        sparse_matrix: coo_matrix
            scipy coo_matrix
        """
        self.data = append( self.data, sparse_matrix.data )
        self.row = append( self.row, sparse_matrix.row )
        self.col = append( self.col, sparse_matrix.col )

    def tosparse(self, scipy_matrix):
        """Convert extended coo_matrix to another arbitrary scipy sparse matrix.

        Parameters
        ----------
        scipy_matrix: scipy.sparse.spmatrix
            scipy sparse matrix format that should be returned
        """
        return scipy_matrix(self)