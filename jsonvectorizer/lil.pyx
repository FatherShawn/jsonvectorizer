cimport cython
cimport numpy as np


@cython.cdivision(True)
cdef inline int bisect_left(list a, int x) except -1:
    # Find index at which to insert x to maintain order
    # Simplified version of counterpart in scipy.sparse._csparsetools
    cdef:
        int lo = 0
        int hi = len(a)
        int mid, v

    if hi > 0:
        v = a[hi - 1]
        if x > v:
            return hi

    while lo < hi:
        mid = lo + (hi - lo) // 2
        v = a[mid]
        if v < x:
            lo = mid + 1
        else:
            hi = mid

    return lo


cdef int lil_insert(
    list[:] rows, list[:] datas, int r, int c, object d
) except -1:
    # Insert a single entry into a binary LIL matrix
    # Simplified version of counterpart in scipy.sparse._csparsetools
    cdef:
        list row = rows[r]
        list data = datas[r]
        int pos = bisect_left(row, c)

    if pos == len(row):
        row.append(c)
        data.append(d)
    elif row[pos] != c:
        row.insert(pos, c)
        data.insert(pos, d)
    else:
        data[pos] += d

    return 0


cdef int lil_set_col(
    list[:] rows, list[:] datas, int[:] rs, int c, object d
) except -1:
    # Set rows in a given column of a LIL matrix
    cdef int i
    for i in range(rs.shape[0]):
        lil_insert(rows, datas, rs[i], c, d)

    return 0


cdef int lil_set(
    list[:] rows, list[:] datas, int[:] rs, int[:] cs, np.ndarray ds
) except -1:
    # Set arbitrary entries in a LIL matrix
    cdef int i
    for i in range(rs.shape[0]):
        lil_insert(rows, datas, rs[i], cs[i], ds[i])

    return 0
