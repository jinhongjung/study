
### array

* 1-D array: A = np.array([1, 2, 3, 4])
    - np.ndim(A): return the dimension of the array A (in this case, it is 1)
    - A.shape: return the shape of the array A (in this case, it is (4, )). This means there are 4 elements at 1st dimension. To get the value, you can use A.shape[0] which returns 4.

* 2-D array: B = np.array([[1,2], [3,4], [5,6]])
    - np.ndim(B) ==> 2
    - B.shape ==> (3, 2) 

* Dot product: np.dot(A, B)
    - Be careful the shapes of A and B, i.e., A.shape[1] == B.shape[0] in matrix
    
