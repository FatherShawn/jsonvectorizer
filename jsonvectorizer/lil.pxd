cimport numpy as np

cdef int lil_set_col(
    list[:] rows, list[:] datas, int[:] rs, int c, object d
) except -1

cdef int lil_set(
    list[:] rows, list[:] datas, int[:] rs, int[:] cs, np.ndarray ds
) except -1
