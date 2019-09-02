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