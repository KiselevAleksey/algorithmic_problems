#!/usr/bin/env python
# coding: utf-8

# In[6]:


def factorial (n):
    K = [0]*n
    if n >= 0 and n <= 2:
        return n
    else:
        K[0] = 1
    for i in range(1, n):
        K[i] = K[i-1]*(i+1)
    print(K, K[n-1])
factorial(10)


# In[26]:


# Выведите элементы списка с чётными индексами.
A =  [int (i) for i in range(1, 6)]
B = [A[i] for i in range (len(A)) if i % 2 == 0]
print (*B) 


# In[55]:


# Найдите наибольший элемент в списке. 
# Выведите значение элемента и его индекс.
A = [1, 2, 3, 2, 1]

B = [max(A), A.index(max(A))]

# B = []
# B.append(max(A))
# for i in range (len(A)):
#     if A[i] == max(A):
#         B.append(i)

print (*B)


# In[64]:


# Выведите список в обратном порядке
A =  [int (i) for i in range(1, 6)]
for i in range(len(A)//2):
    A[i], A[len(A)-1-i] = A[len(A)-1-i], A[i]
print (*A)


# In[33]:


A =  [int (i) for i in range(1, 6)]
print (A[::-1])


# In[68]:


# Переставьте соседние элементы в списке. Задача решается в три строки.
A = [1, 2, 3, 4, 5]
for i in range(0, len(A)-1, 2):
    A[i], A[i+1] = A[i+1], A[i]
print (*A)


# In[48]:


b = 'ab'
a = 'ac'
A,B = [],[]
for i in range(len(a)):
    A.append(a[i])
for i in range(len(b)):
    B.append(b[i])
print(A,B)
    


# In[65]:


for i in 'abc':
    print(i)


# In[52]:


def isAnagram(s: str, t: str) -> bool:
    A,B = [],[]
        
    for i in range(len(s)):
        A.append(s[i])
    for i in range(len(t)):
        B.append(t[i])
    if len(A) == len(B):  
        for i,j in zip(A,B):
            if i !=j:
                return 'false'
        return 'true'
    return 'false'


# In[102]:


class Solution:
    def isAnagram(s: str, t: str) -> bool:
        return sorted(s) == sorted (t)
        
#         A,B = [],[]     
#         for i in range(len(s)):
#             A.append(s[i])
#         for i in range(len(t)):
#             B.append(t[i])
#         if len(A) == len(B):  
#             for i,j in zip(A,B):
#                 if i !=j:
#                     return 'false'
#             return 'true'
#         return 'false'
Solution.isAnagram ("wer", "rew")


# In[86]:


# Выведите элементы, которые встречаются в списке только один раз. 
# Элементы нужно выводить в том порядке, 
# в котором они встречаются в списке.
a = [int(s) for s in input().split()]
for i in range(len(a)):
    for j in range(len(a)):
        if i != j and a[i] == a[j]:
            break
    else:
        print(a[i], end=' ')


# In[123]:


def max_forward(A:list):
    """Требуется поменять местами первый элемент 
    массива с максимальным"""
    A = list(map(int,input().split()))
    max = A[0]
    pos = 0
    for i in range(len(A)):
        if A[i] > max:
            max = A[i]
            pos = i
    A[0], A[pos] = A[pos], A[0]
    return print(A)

max_forward(A)


# In[135]:


def new_digit_input(A:list, B:list):
    """Требуется вставить в данный массив на данное место 
    данный элемент, сдвинув остальные элементы вправо.
    A - массив
    B - число и номер места в массиве, куда надо вставить число"""
    A.append(B[0])
    N = len(A)
    for i in range (B[1]-1, len(A)-1):
        A[i], A[N-1] = A[N-1], A[i]
    return (A)
new_digit_input([0, 3, 2, 0, 4], [5, 1])


# In[21]:


A = [3, 2]
B = [5, 10]
G = [7, 5]
H = [5, 5]

C = []
C = A+B+G+H
x = y = z = 0

for i in range (1, len(C)):
    k = i
    while k>0 and C[k-1] > C[k]:
        C[k], C[k-1] = C [k-1], C[k]
        k -= 1
print ("C:", C)

D = [9, 5, 3, 5, 12, 10]

for pos in range (0, len(D)-1):
    for k in range (pos+1, len(D)):
        if D[k] < D [pos]:
            D[k], D [pos] = D[pos], D[k]

print ("D:", D)

counter = 0
i = j = 0

while i < (len(C)) and j < (len(D)):
    if C[i] >= D[j]:
        counter += 1
        i+=1
        j+=1
    else:
        i+=1

print(counter)


# In[36]:


N = int(input())
A,B = [],[]
for n in range(N):
    a= [int(i) for i in input().split()]
    A.append(a[0])
    B.append(a[1])
for bypass in range(1,N):
    for j in range (0,N-bypass):
        if  B[j] < B[j+1] or (B[j] == B[j+1] and A[j]> A[j+1]):
            B[j],B[j+1] = B[j+1],B[j]
            A[j],A[j+1] = A[j+1],A[j]
for i in range(N):
    print(A[i],B[i],end ='\n')  


# In[11]:


A = [1, 5, 2, 3, 6]
j = A[0]
for i in range(len(A)):
     if j < A[i]:
        j = A[i]
j


# In[1]:


N = int(input())
B = [int(i) for i in input().split()]
x = int(input())
count = 0
for n in B:
    if x == n:
        count+=1
print(count)


# In[3]:


7 // 2


# In[29]:


B = [0,6,8,9,8,6,54,36,5,7,0]
for bypass in range(1,len(B)):
    for j in range (0,len(B)-bypass):
        if  B[j] < B[j+1]:
            B[j],B[j+1] = B[j+1],B[j]


# In[30]:


B


# In[23]:


dict


# In[21]:


dict = {}
for n in range(N):
    a= [int(i) for i in input().split()]
    dict[a[1]] =a[0]


# In[22]:


sorted(dict)


# In[ ]:


num = [0]*N
mark = [0]*N

for n in range(N):
    num[n]=


# In[6]:


k = 1
n = 5
for i in range (1, n+1):
    k *= i
k


# In[8]:


# Вычислить n!
k = 1
n = 5
while n >0:
    k *=n
    n -= 1
k    


# In[21]:


# Вычислить n-ный член последовательности Фибоначчи
n = 10
A =[0,1]+[0]*(n-1)
for i in range (2, n+1):
    A[i] = A [i-1]+ A[i-2]
A[n]


# In[4]:


# Дано натуральное n, вычислить:
# 1 / 0! + 1 / 1! + ... + 1 / n!

n = 5
k = 1
j = 0

for i in range (1,n+1):
    k *= i
    j += 1/k
print (j + 1)


# In[35]:


# Даны два натуральных числа a и b, не равные нулю одновременно. 
# Вычислить НОД(a,b) — наибольший общий делитель а и b.
n = 3000
m = 140
while n != m:
    if n > m:
        n -= m
    else:
        m -=n
print(n)


# In[6]:


# Решить предыдущую задачу, 
# используя в алгоритме Евклида деление с остатком
n = 3001
m = 151
while (n != 0 and m != 0) and (n%m != 0 or m%n != 0):
    if n > m:
        n %= m
    else:
        m %= n
print (max(m, n))


# In[21]:


# Проверить, является ли заданное натуральное число n > 1 простым.
n = 12
i = 2
while i < n:
    if n % i == 0:
        break
    else:
        i += 1
print("Простое" if i == n else "Составное")


# ### Динамическое программирование. Задачи

# ##### Задача 1. Посчитать число последовательностей нулей и единиц длины n, в которых не встречаются две идущие подряд единицы.

# In[25]:


def variations_count (N: int):
    """Очевидно, какими будут значения для N = 0; 1
    Задача решается рекурсивно аналогичным методом
    с поиском чисел Фибаначи"""
    K = [2, 3] + [0]*(N-1)
    for i in range (2, N+1):
        K[i] = K[i-1]+K[i-2]
    return K[N]
variations_count(10)


# In[41]:


def robber (A:list):
    if (len (A)) == 0:
        return 0
    elif (len(A)) <3:
        return max(A)
    else:
        C = [A[0], A[1], A[2]+A[0]]+[0]*(len(A)-3)
        for i in range (3, len(A)):
            C[i] = A[i] + max(C[i-2],C[i-3])
        return max(C)


# In[43]:


r0bber ([1])


# In[47]:


def rob(nums):
    if nums == []: 
        return 0
    if len(nums) == 1:
        return nums[0]
    dp = [0]*len(nums)
    dp[0] = nums[0]
    dp[1] = max(nums[1],nums[0])
    for i in range(2,len(nums)):
        dp[i] = max(dp[i-2]+nums[i],dp[i-1])
    print(dp)
    return dp[-1]

rob([2,1,1,2])


# In[79]:


# Решите задачу о количестве способов достичь точки n из точки 1,
# если кузнечик умеет прыгать +1, +2 и +3.
def count_traj_1 (A:int):
    K = [0, 1, 1]+[0]*(A-2)
    for i in range (3, A+1):
        K[i] = K[i-1]+K[i-2]+K[i-3]
    return K, K[A]

count_traj_1 (4)


# In[80]:


# Решите задачу о количестве способов достичь точки n из точки 1,
# если кузнечик умеет прыгать +1, +2 и *3.
def count_traj_2 (N:int):
    K = [0, 1]+[0]*(N-1)
    for i in range (2, N+1):
        print (i)
        if (i)%3 == 0:
            K[i] = K[i-1]+K[i-2]+1
        else:
            K[i] = K[i-1]+K[i-2]
        
    return K, K[N]

count_traj_2 (6)
    


# #### Ферзь

# In[40]:


def ferz (A, B):
    F = [[0]*(B+1) for i in range (A+1)] #создали массив AxB заполненный нулями
    n = k = 1 # задаем переменные - счетчики
    for i in range(A, -1, -1): #бежим по строке
        for j in range (B, -1, -1): #бежим по столбцу
            if F[i][j] == 0: #если значение клетки где мы стоим нулевое, то
                while n != A+1: #меняем значения по строке на 1 
                    # все, кроме той клеточки, в которой стояли
                    F[i-n][j] = 1
                    n += 1
                while k != B+1: #меняем значения по столбцу на 1 
                    F[i][j-k] = 1
                    k += 1
                n = k = 1
                while n != A+1 or k != B+1:
                    #меняем значения по диагонали на 1 
                    F[i-k][j-n] = 1
                    k += 1
                    n += 1
                n = k = 1
        # когда переходим в след j он уже равен 1
    #тоже самое с i, пока не выйдем на нужную нам клетку, 
    #которая не попадала под изменения  
    return F
# ferz (5,5)


# In[73]:


def ferz_2 (N,M):
    # N - высота массива,M - ширина массива
    F = [[0]*M for i in range (N)] #создали массив AxB заполненный нулями
    
    F[-1][-1]=1
    massiv_x = [0]*(M-1) + [1]
    massiv_diag = [[0]*(M+1) for i in range (N+1)]
    massiv_diag[-1][-1] = 1
    massiv_y = [0]*(N-1) + [1]
    
    
    for i in range(N-1,-1,-1):
        for j in range(M-1,-1,-1):
            if massiv_diag[i+1][j+1]:
                massiv_diag[i][j] = 1
            if massiv_x[j] or massiv_y[i] or massiv_diag[i][j] :
                continue
            else:
                F[i][j] = 1
                massiv_diag[i][j]=1
                massiv_x[j]=1
                massiv_y[i]=1
    return F


# In[128]:


ferz_2 (6,6)


# #### Камни

# In[126]:


def kamni(N,L,R):
    """игра на убирание камней от L до R из кучи N
    выигрывает тот, кто делает последний ход"""
    F = [0]*(N-L+1) + [0]*(L-1)+[0]
    for i in range(-L-1,-N-2,-1):
        for j in range(L,R+1,1):
            if  i+j < 0  and F[i+j] == 0:
                F[i] = 1
                break
            else:
                continue
    return F

kamni(12,3,5)


# #### Игра на деление

# In[129]:


def devisor (N:int):
    """идет игра по правилам:
    1) Выбор числа x: 0 < x < N and N % x == 0.
    2) Замена N на (N - x)
    проигрывает тот, у кого не остается ходов"""
    return True if N % 2 == 0 else False 


# In[136]:


def devisor_2 (N:int):
    F = [0]*(N+1)
    for i in range (2, N+1):
        for j in range (1, i):
            if i%j == 0 and F[i-j] == 0:
                F[i] = 1
    return  F[N]

print (devisor(151))
print (devisor_2(151))


# #### Префикс функция

# In[144]:


def prefix(A:str):

    pi = [0]*(len(A))
    for i in range (len(A)):
        p = pi[i]
        while p >0 and A[i]!= A[p+1]:
            p = A[p]
        if A[i] == A[p]:
            p+=1
            pi[i]=p
    return pi


# In[145]:


prefix('abcabc')


# #### Компьютерная игра

# In[159]:


def game (N:list):
    """Прыжки с платформы на платформу высоты y_1, y_2.
    Найти мин затраты энергии. Затраты:
    С соседней на соседнюю - (abs(y_1-y_2). Через одну (3*abs(y_1-y_3)
    N - высоты платформ"""
    F = [0]*(len(N))
    F[1] = abs(N[0]-N[1])
    for i in range (2, len(N)):
        F[i] = min((abs(N[i]-N[i-2])*3) +F[i-2], abs(N[i]-N[i-1])+F[i-1])
    return F,F[-1]

game ([2,5,7,5,3,8,1,5,8])


# In[166]:


def minCostClimbingStairs(cost):
    cost = cost + [0]
    F = [0]*(len(cost))
    F[0] = cost[0]
    F[1] = cost[1]
    for i in range (2, len(cost)):
        F[i] = cost[i]+min(F[i-1], F[i-2])
    return(F[-1])
minCostClimbingStairs([10, 15, 20])


# #### Проверка списка на максимальный элемент

# In[179]:


def dominantIndex(nums:list):
    """Находит, является ли максимальный элемент списка хотя бы в 4 раза больше,
    чем любой из других элеменов списка"""
    if len(nums) ==0:
        return -1
    if len(nums)==1:
        return 0

    max_element,predmax_element = nums[0],nums[1]
    index_of_max_element = 0

    for i in range(1,len(nums)):
        if nums[i] >= max_element:
            predmax_element = max_element
            max_element = nums[i]
            index_of_max_element = i
        elif  nums[i] >= predmax_element:
            predmax_element = nums[i]

    return index_of_max_element if max_element // 2 >= predmax_element else -1


# In[180]:


dominantIndex([10,4,5,3,0]) == 0


# ## Яндекс контест

# #### Прибыль Остапа

# In[10]:


def profit_search(Variables:list, A:list, B:list):
    """Реализация алгоритма нахождения максимальной прибыли Остапа"""
    A = sorted(A)
    B = sorted(B, reverse=True)
    counter = 0
    for i in range (len(A)):
        for j in range (i, len(B)):
            if B[j] >= A[i]:
                counter += (B[j] - A[i])
                break
            else:
                continue
    return counter

profit_search([int(i) for i in input().split()], [int(i) for i in input().split()], [int(i) for i in input().split()])


# #### Сумма медиан последовательно растущего ряда

# In[1]:


def median_of_nums(varibles:list, A:list):
    """Возвращает сумму медиан при последовательном добавлении элементов ряда"""
    median_counter = A[0]
    for top in range (1, len(A)):
        k = top
        while k > 0 and A[k-1] > A[k]:
            A[k], A[k-1] = A[k-1], A[k]
            k -= 1
        if top%2 == 0:
            median = A[top //2]
        else:
            median = A[(top) // 2] 
        median_counter += median
    return median_counter

print(median_of_nums([int(i) for i in input().split()], [int(i) for i in input().split()]))


# In[97]:


def med(varibles:list, A:list):
    if len(A) == 0:
        return 0
    median = A[0]
    for i in range(1,len(A)):
        median += sorted(A[:i+1])[i//2]
    return median


# In[4]:


import heapq
from heapq import heappush, heappush
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


# #### Площадь пересечения двух прямогольников

# In[23]:


def find_square (A:list, B:list):
    """Поиск площади пересечения 2 прямоуг.
    На вход подается:
    A - координаты 2 противоположных углов прямоуг A
    B - координаты 2 противоположных углов прямоуг B"""
    x11 = max(min(A[0],A[2]),min(B[0],B[2]))
    x12 = min(max(A[0],A[2]),max(B[0],B[2]))
    y11 = max(min(A[1],A[3]),min(B[1],B[3]))
    y12 = min(max(A[1],A[3]),max(B[1],B[3]))
    if (x12 - x11 > 0) and (y12 - y11 > 0):
        sq = (x12 - x11) * (y12 - y11)
    else:
        sq = 0
    return sq
print(find_square([int(i) for i in input().split()],[int(i) for i in input().split()]))


# ## Динамическое программирование по подстрокам

# #### Порядок умножения матриц

# In[6]:


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


# #### Максимальный подпалиндром

# In[48]:


def podpalindrom (A:str):
    """Рекурентный поиск максимального подпалидрома в строке"""
    F = [[(1 if i == j else 0) for i in range (len(A))]for j in range(len(A))]
    for L in range(1, len(A)):
        for i in range(0, len(A)-L):
            j = i + L
            if A[i] == A[j]:
                F[i][j] = F[i+1][j-1] +2
            else:
                F[i][j] = max(F[i][j-1], F[i+1][j])
    return F, F[0][-1]
podpalindrom('AOOPOA')


# #### Задача на подсчет носков разного цвета

# In[11]:


from collections import Counter
def socks_count (n, A):  
    socks, pairs = A, 0
    for s in socks: 
        pairs += socks[s]//2
    return pairs
socks_count((int(input())), Counter(map(int,input().split())))


# #### Поиск равнин

# In[16]:


def valley_found (n:int, A:str):
    hight_counter = 0
    valley_counter = 0
    for i in A:
        if i == 'U':
            hight_counter += 1
        else:
            hight_counter -= 1
        if i == 'D' and hight_counter + 1 == 0:
            valley_counter += 1
    return valley_counter
valley_found(8, 'DDUUUDDDDUU')
        


# #### Прыжки по облакам

# In[21]:


import sys
def jumps(n:int, A:list):
    F = [0] + [sys.maxsize if A[1] else 1] + [0]*(len(A)-2)
    for i in range (2,len(A)):
        if not A[i]:
            F[i] = min (F[i-1],F[i-2]) + 1
        else:
            F[i] = sys.maxsize
    return F[-1]
jumps(6, [0,0,0,0,1,0])


# #### Повторение строк

# In[47]:


def repeat_strings(n:int, s:str):
    if len(s) == 1:
        return n
    else:
        counter = 0
        s = s*((n//len(s))+1)
        for i in range(0, n):
            if s[i] == s[n-1]:
                counter += 1
        return counter
repeat_strings(817723, 'ceebbcb')
            


# In[53]:


from collections import Counter
def repeatedString (s:str, n:int):
    if len(s) == 1:
        return n
    else:
        const = len(s)
        while (len(s)) < n:
            s += s[len(s)-const]
        A = Counter(s)
        for i in A:
            if i == s[-1]:
                return A[i]
repeat_strings (547602, 'gfcaaaecbg')


# #### Two Sum

# In[62]:


def bruteForceTwoSum (A:list, target:int):
    for i in range (len(A)):
        for j in range(i+1, len(A)):
            if A[i] + A[j] == target:
                return (i, j)


# In[63]:


bruteForceTwoSum([2, 7, 11, 15], 9)


# In[72]:


def TwoSum (A:list, target:int):
    complement_map = dict()
    
    for i in range (len(A)):
        complement = target - A[i]
        if A[i] in complement_map:
            return complement_map[A[i]], i
        else: 
            complement_map[complement] = i

TwoSum([7, 11, 15, 2], 9)


# #### 26. Remove Duplicates from Sorted Array

# In[79]:


def removeDuplicates(nums: list):
        counter = 1
        while counter < len(nums):
            if nums[counter] == nums[counter - 1]:
                del(nums[counter])
            else:
                counter += 1
        return(len(nums), nums)
removeDuplicates([0,0,1,1,1,2,2,3,3,4])


# In[ ]:




