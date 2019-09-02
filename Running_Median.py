"""
Метод quickselect

"""

import random
def quickselect_median(l, pivot_fn=random.choice):
    if len(l) % 2 == 1:     
        return quickselect(l,len(l)//2,pivot_fn)
    else:
        return quickselect(l,(len(l)//2)-1,pivot_fn)


def quickselect(l, k, pivot_fn):
    if len(l) == 1:
        return l[0]

    pivot = pivot_fn(l)

    lows = [el for el in l if el < pivot]
    highs = [el for el in l if el > pivot]
    pivots = [el for el in l if el == pivot]

    if k < len(lows):
        return quickselect(lows, k, pivot_fn)
    elif k < len(lows) + len(pivots):
        return pivots[0]
    else:
        return quickselect(highs, k - len(lows) - len(pivots), pivot_fn)

def median(varibles:list, A:list):
    med = A[0]
    for i in range(1,len(A)):
        med += quickselect_median(A[:i+1])
    return med 
print(median([int(i) for i in input().split()],[int(i) for i in input().split()]))


"""встроенная сортировка"""

def med(varibles:list, A:list):
    median = A[0]
    for i in range(1,len(A)):
        median += sorted(A[:i+1])[i//2]
    return median
print(med([int(i) for i in input().split()], [int(i) for i in input().split()]))
