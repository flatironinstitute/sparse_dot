# sparse_dot

This is a wrapper for the sparse matrix multiplication in the intel MKL library. 
The main advantage to MKL (which motivated this) is multithreaded sparse matrix multiplication. 
The scipy sparse implementation is single-threaded.