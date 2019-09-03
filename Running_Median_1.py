""" Find Median from Data Stream
Median is the middle value in an ordered integer list. If the size of the list is even, there is no middle value. 
So the median is the mean of the two middle value.

For example,
[2,3,4], the median is 3

[2,3], the median is 2 (only in that problem)

Design a data structure that supports the following two operations:

void addNum(int num) - Add a integer number from the data stream to the data structure.
double findMedian() - Return the median of all elements so far."""

import heapq
from heapq import heappop, heappush
def median_counter(var:int, nums:list): 
    counter = 0
    heap_max = []
    heap_min = []
    
    for num in nums:
        if len(heap_max) == 0 or heap_max[0]*-1 >= num:
            heapq.heappush(heap_max, num*-1)
        else:
            heappush(heap_min, num) 
            
        N1, N2 = len(heap_max), len(heap_min)
        if N1-N2 == 2:
            x = heappop(heap_max)
            heappush(heap_min, x*-1)
        elif N2-N1 == 2:
            x = heappop(heap_min)
            heappush(heap_max, x*-1)  

        N1, N2 = len(heap_max), len(heap_min)
        if N1 >= N2:
            counter += (heap_max[0]*-1)
        else:
            counter += (heap_min[0])

    return counter
print (median_counter([int(i) for i in input().split()], [int(i) for i in input().split()]))