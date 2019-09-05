"""
Given an array p[] which represents the chain of matrices such that the ith matrix Ai is of dimension p[i-1] x p[i]. 
We need to write a function MatrixChainOrder() that should return the minimum number of multiplications needed to multiply the chain.

  Input: p[] = {40, 20, 30, 10, 30}   
  Output: 26000  
  There are 4 matrices of dimensions 40x20, 20x30, 30x10 and 10x30.
  Let the input 4 matrices be A, B, C and D.  The minimum number of 
  multiplications are obtained by putting parenthesis in following way
  (A(BC))D --> 20*30*10 + 40*20*10 + 40*10*30
"""

def matrixes_multiply (A:list):
    """Возвращает наименьшее возможное число операций при перемножении листа матриц.
    A - лист чисел, которые составляют последовательно идушие матрицы
    например A = [1, 2, 3] обозначает, что матрицы A1=1x2, A2=2x3"""
    n = len(A) #число матриц
    F = [[0 for i in range (n)] for i in range (n)]
    for i in range(n):
        F[i][i] = 0
    for L in range (2, n):
        for i in range (1, n-L+1):
            j = i + L -1
            F[i][j] = sys.maxsize
            
            for k in range(i, j):
                operations = F[i][k] + F[k+1][j] + A[i-1]*A[k]*A[j]
                if operations < F[i][j]:
                    F[i][j] = operations
    return F, F[0][-1]
matrixes_multiply([1, 2, 3, 4, 5])