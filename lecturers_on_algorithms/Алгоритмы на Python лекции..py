#!/usr/bin/env python
# coding: utf-8

# ## Лекция 2 

# #### Булевы операции

# In[35]:


def check_10_split (N:list, a:int = 10):
    """Проверяет, что хотя бы одно число из списка N кратно a"""
    flag = False
    for i in range (len(N)):
        flag = (N[i]%a == 0) or flag
    return flag
check_10_split ([1, 1, 20, 15])


# In[40]:


def check_3_or_2_split (N:list, a:int = 2, b:int = 3):
    """Проверка делимости хотя бы одного числа в списке на 2 или 3"""
    flag = False
    for i in range (len(N)):
        flag = (N[i]%a == 0 or N[i]%b == 0) or flag
    return flag
check_3_or_2_split ([3, 1, 1, 15])


# ## Лекция 3

# #### Функции

# In[44]:


def max_2 (a:int, b:int):
    """Возвращает большее из 2 чисел"""
    if a > b:
        return (a)
    return (b)
max_2(3,4)


# In[46]:


def max_3 (a, b, c):
    """Возвращает большее из 3 чисел"""
    return max_2(a, max_2(b, c))
max_3 (2,4,3)


# ## Лекция 4

# #### Метод грубой силы (Brute force)

# In[4]:


def is_simple_number(x):
    """Определяет, является ли число простым.
    x - целое положительное число.
    Если да - True, иначе - False"""
    divisor = 2
    while divisor < x:
        if x%divisor == 0:
            return False
        divisor += 1
    return True
is_simple_number(11)


# In[9]:


def facrorize_number(x):
    """Раскладывает число на множители.
    Печатает их на экран.
     x - целое положительное число"""
    divisor = 2
    while x > 1:
        if x%divisor == 0:
            print(divisor)
            x//=divisor
        else: 
            divisor +=1
facrorize_number(999)


# ## Лекция 5

# #### Копирование массива

# In[50]:


N = int(input())
A = [0]*N
B = [0]*N
for i in range (N):
    A[i] = int(input())
for i in range (N):
    B[i] = A[i]
print (B)


# #### Алгоритм линейный поиск в массиве

# In[12]:


def array_search (A:list, N:int, x:int):
    """Осуществляет поиск числа x в массиве А
    от 0 до N-1 индека включительно.
    Возвращает индекс элемента в массиве A.
    Если элемента нет - возращает -1
    Если в массиве несколько элементов равных x, 
    вернет индекс первого по счету"""
    for k in range (N):
        if A[k] == x:
            return k
    return -1

def test_array_search():
    A1 = [1, 2, 3, 4, 5]
    m = array_search(A1, 5, 8)
    if m == -1:
        print ('#test1 - ok')
    else:
        print ('#test1 - fail')
        
        
    A2 = [-1, -2, -3, -4, -5]
    m = array_search(A1, 5, 8)
    if m == 2:
        print ('#test2 - ok')
    else:
        print ('#test2 - fail')
        
    A3 = [10, -2, -3, -4, -5]
    m = array_search(A1, 5, 8)
    if m == 0:
        print ('#test3 - ok')
    else:
        print ('#test3 - fail')
        
test_array_search()


# #### Алгоритм обращения массива

# In[15]:


def invert_array (A:list, N:int):
    """Обращение массива (поворот задом-наперед)
    в рамках индексов от 0 до N-1"""
    for k in range (N//2):
        A[k], A[N-1-k] = A[N-1-k], A[k]

def test_invert_array():
    A1 = [1, 2, 3, 4, 5]
    invert_array (A1, 5)
    if A1 == [5, 4, 3, 2, 1]:
        print ('#test1 - ok')
    else:
        print ('#test1 - fail')
    
    A2 = [0, 0, 0, 0, 0, 0, 0, 10]
    invert_array (A2, 8)
    if A2 == [10, 0, 0, 0, 0, 0, 0, 0]:
        print ('#test2 - ok')
    else:
        print ('#test2 - fail')

test_invert_array()


# #### Алгоритм циклического сдвига 

# In[20]:


def left_shift (A:list, N:int):
    """Алгоритм циклического сдвига влево"""
    tmp = A[0]
    for k in range (N-1):
        A[k] = A[k + 1]
    A[N-1] = tmp

def test_left_shift():
    A1 = [1, 2, 3, 4, 5]
    left_shift (A1, 5)
    if A1 == [2, 3, 4, 5, 1]:
        print ('#test1 - ok')
    else:
        print ('#test1 - fail')
test_left_shift()


# In[28]:


def right_shift (A:list, N:int):
    """Алгоритм циклического сдвига вправо"""
    tmp = A[N-1]
    for k in range (N-2, -1, -1):
        A[k+1] = A[k]
    A[0] = tmp

def test_right_shift():
    A1 = [1, 2, 3, 4, 5]
    right_shift (A1, 5)
    print(A1)
    if A1 == [5, 1, 2, 3, 4]:
        print ('#test1 - ok')
    else:
        print ('#test1 - fail')
test_right_shift()


# ### Алгоритм Решето Эратосфена

# In[43]:


def find_prime_number (A:list, N:int):
    """Алгоритм,смысл которого в вычеркивании чисел,
    кратных уже найденным простым."""
    A = [True] * N
    A[0] = A[1] = False
    for k in range (2, N):
        if A[k]: #True не пишется, тк уже задано ранее
            for m in range (2*k, N, k):
                A[m] = False
    for k in range (N):
        print(k, '-', 'простое' if A[k] else 'составное')

find_prime_number([], 5)


# ## СОРТИРОВКА

# In[74]:


def insert_sort (A):
    """сортировка списка А вставками"""
    N = len (A)
    for top in range (1, N):
        k = top
        while k > 0 and A[k-1] > A[k]:
            A[k], A[k-1] = A[k-1], A[k]
            k -= 1

def choise_sort (A):
    """сортировка списка А выбором"""
    N = len (A)
    for pos in range (0, N-1):
        for k in range (pos+1, N):
            if A[k] < A [pos]:
                A[k], A [pos] = A[pos], A[k]

def bubble_sort (A):
    """сортировка списка А пузырьком"""
    N = len (A)
    for bypass in range (1, N):
        for k in range (0, N-bypass):
            if A[k] > A [k+1]:
                A[k], A [k+1] = A[k+1], A[k]


# In[75]:


def test_sort(sort_algorithm):
#     print ("Тестируем" sort_algorithm.__doc__)
    print("testcase #1:", end="")
    A = [4, 2, 5, 1, 3]
    A_sorted = [1, 2, 3, 4, 5]
    sort_algorithm(A)
    print ("Ok" if A == A_sorted else "Fail")
    
    print("testcase #2:", end="")
    A = list(range(10,20)) +list (range(0,10))
    A_sorted = list(range(0,20))
    sort_algorithm(A)
    print ("Ok" if A == A_sorted else "Fail")
    
    print("testcase #3:", end="")
    A = [4, 2, 4, 2, 1]
    A_sorted = [1, 2, 2, 4, 4]
    sort_algorithm (A)
    print ("Ok" if A == A_sorted else "Fail")


# In[76]:


if __name__ == "__main__":
    test_sort(insert_sort)
    test_sort(choise_sort)
    test_sort(bubble_sort)


# #### Cортировка подсчетом (Count_sort)

# In[101]:


def count_sort(numbers:list, max_value:int):
    """метод сортировки подсчетом"""
    results = []
    counter = [0] * (max_value + 1)
    for i in numbers:
        counter[i] += 1
    for i, count in enumerate(counter):
        results += [i] * count
    print(results)
count_sort([2, 4, 4, 3, 2, 3, 1, 2, 3,], 4)


# ## Рекурсия

# In[110]:


def matryoshka (n):
    if  n == 1:
        print ("Матрешечка")
    else:
        print ("Верх матрешки n=", n)
        matryoshka (n-1)
        print ("Низ матрешки n=", n)

matryoshka (3)


# #### Создание рекурсии - квадрата в квадрате

# In[3]:


import graphics as gr

window = gr.GraphWin ("Russian game", 100, 100)
alpha = 0.2


# In[4]:


def fractal_rectangle(A, B, C, D, deep = 10):
    """Функция рисует квадрат в квадрате
    A, B, C, D - кортежи (1x2) координатами точек квадрата
    deep - глубина рекурсии
    *A - синтаксичесикй сахар, разворачивающий кортеж"""
    if deep < 1:
        return
    for M, N in (A, B), (B, C), (C, D), (D, A):
        gr.Line(gr.Point(*M), gr.Point(*N)).draw(window)
    A1 = (A[0]*(1-alpha) + B[0]*alpha, A[1]*(1-alpha)+ B[1]*alpha)
    B1 = (B[0]*(1-alpha) + C[0]*alpha, B[1]*(1-alpha)+ C[1]*alpha)
    C1 = (C[0]*(1-alpha) + D[0]*alpha, C[1]*(1-alpha)+ D[1]*alpha)
    D1 = (D[0]*(1-alpha) + A[0]*alpha, D[1]*(1-alpha)+ A[1]*alpha)
    fractal_rectangle(A1, B1, C1, D1, deep-1)


# #### Рекурсия Факториал

# In[69]:


def factorial (n:int):
    """Функция считает факториал числа по формуле: 
    n! = (n-1)!*n """
    assert n >= 0, "Факториал не определен"
    assert n <= 100, "Возможно переполнение стэка"
    if n == 0:
        return 1
    return factorial(n-1)*n
factorial (6)


# #### Алгоритм Евклида

# In[72]:


def gcd (a: int, b:int):
    """Нахождение наибольшего общего делителя 2 чисел"""
    if a == b:
        return a
    elif a > b:
        return gcd (a-b, b)
    else:
        return gcd (a, b-a)
gcd(28, 8)


# #### Быстрое возведение в степень (вариант 1)

# In[83]:


def pow_1 (n:int, k:int):
    """Возведение числа n в степень k по формуле:
    n^k = n^(k-1)*n"""
    assert k >= 0, "Только положительные степени"
    if k == 0:
        return 1
    else:
        return pow(n, k-1)*n
pow_1 (3, 4)


# #### Быстрое возведение в степень (вариант 2)

# In[89]:


def pow_2 (n:int, k:int):
    """Возведение в степень с использованием свойства:
    n^k = (n^2)^k/2"""
    
    if k == 0:
        return 1
    elif k % 2 != 0:
        return pow_2(n, k-1)*n
    else:
        return pow_2(n**2, k/2)
pow_2(3, 7)


# #### Ханойская башня - рекурсивное решение задачи.

# In[1]:


def moveTower(height,fromPole, toPole, withPole):
    if height == 0:
        print("Наигрались")
        return
    elif height >= 1:
        moveTower(height-1,fromPole,withPole,toPole)
        moveDisk(fromPole,toPole)
        moveTower(height-1,withPole,toPole,fromPole)

def moveDisk(fp,tp):
    print("moving disk from",fp,"to",tp)

moveTower(2,"A","B","C")


# #### Числа Фибоначчи

# In[79]:


def fib (n:int):
    """N - номер числа в списке"""
    if n <= 1:
        return n
    return fib(n-1)+fib(n-2)
fibonacci_num (4)


# #### Фибоначчи циклическим методом

# In[15]:


n = 100
fib = [0,1] +[0]*(n-1)
for i in range (2, n+1):
    fib[i] = fib[i-1] + fib[i-2] 
fib[n]


# ## Генерация всех перестановок

# In[5]:


def gen_bin (M, prefix = ""):
    """Функция, аналогичная ф-ции ниже, 
    но для двоичной системы счисления"""
    if M == 0:
        print (prefix)
        return
    for digit in "0", "1":
        gen_bin (M-1, prefix+digit)
gen_bin(3)


# In[26]:


def generate_number (N:int, M:int, prefix = None):
    """Генерирует все числа (с лидирующими незначащими нулями)
    в N-Ричной системе счисления (N <= 10)
    длины M"""
    prefix = prefix or []
    if M == 0:
        print (prefix)
        return
    for digit in range (N):
        prefix.append(digit)
        generate_number(N, M-1, prefix)
        prefix.pop( )
generate_number (3, 2, prefix = None)


# In[21]:


def find (number, A):
    """ищет number в A и возвращает True, если такой есть;
    False, если такого нет"""
    for x in A:
        if number == x:
            return True
    return False


def generate_permutations (N:int, M:int = -1, prefix = None):
    """Генерация всех перестановок N чисел в M позициях,
    с префиксом prefix"""
    M = N if M == -1 else M #по умолчанию N чисел в N позициях
    prefix = prefix or []
    if M == 0:
        print(*prefix, end = ',', sep = "")
        return
    for number in range (1, N+1):
        if find(number, prefix):
            continue
        prefix.append(number)
        generate_permutations (N, M-1, prefix)
        prefix.pop()

generate_permutations(3, 3)


# ## Рекурентные функция сортировки. Сортировка слиянием 

# #### Фукнция слияния отсортированных массивов

# In[36]:


def merge(A: list, B: list):
    """Производит слияние 2 массивов A и B в C с одновременной сортировкой"""
    C = [0]* (len(A)+len(B))
    i = j = n = 0
    while i < len(A) and j < len(B):
        if A[i] <= B [j]:
            C[n] = A[i]
            i += 1
            n += 1
        elif A[i] > B[j]:
            C[n] = B[j]
            j += 1
            n +=1
    while i < len(A):
        C[n] = A[i]
        i += 1
        n += 1
    while j < len(B):
        C[n] = B[j]
        j += 1
        n +=1
    return C


# In[42]:


merge ([2, 3, 5, 6, 7], [1, 4, 7, 8])


# #### Реализация сортировки слиянием

# In[56]:


def merge_sort (A):
    """Разбивает список А на 2 части, рекурентно сортирует их
    и далее ранее заданной функцией merge собирает их в единый список"""
    if len(A) <= 1:
        return
    middle = len(A) // 2
    L = [A[i] for i in range (0, middle)]
    R = [A[i] for i in range (middle, len(A))]
    
    print("L:", L)
    merge_sort (L)
    print("R:", R)
    merge_sort (R)
    
    C = merge (L, R)
    print("C:", C)
    for i in range (len(A)):
        A[i] = C[i]
    return A

merge_sort ([1, 5, 3, 2, 2])


# #### Сортировка Тони-Хоара

# In[61]:


def hoar_sort(A):
    """Разделяет массив на 3 подмассива:
    L (left) - меньшие первого числа.
    M (middle) - равные первому числу.
    R (right) - большие первого числа.
    Берется первое число,
    т.к. больше всего похоже по смыслу на случайно выбранное)
    """
    if len (A) <= 1:
        return
    L = []; M = []; R = []
    barrier = A[0]
    for x in A:
        if x < barrier:
            L.append(x)
        elif x == barrier:
            M.append(x)
        else:
            R.append(x)
    
    print ("L:", L)
    print ("M:", M)
    hoar_sort(L)
    print ("R:", R)
    hoar_sort(R)
    
    k = 0
    for i in L+M+R:
        A[k] = i
        k += 1
    print("A:", A)
    return A

hoar_sort([3, 5, 1, 2, 2, 4])


# ## Бинарный поиск в Массиве

# In[31]:


def left_board(A, key):
    """Поиск левой границы в массиве
    A - массив
    key - искомое число в массиве"""
    left = -1
    right = len(A)
    while right - left > 1:
        middle = (left+right)//2
        if A[middle] < key:
            left = middle
        else:
            right = middle
    return left

left_board([1, 2, 2, 4, 5, 7], 8)


# In[32]:


def right_board(A, key):
    """Поиск левой границы в массиве
    A - массив
    key - искомое число в массиве"""
    left = -1
    right = len(A)
    while right - left > 1:
        middle = (left+right)//2
        if A[middle] <= key:
            left = middle
        else:
            right = middle
    return right

right_board([1, 2, 2, 4, 5, 7], 8)


# ## Динамическуое программирование

# #### Испольнитель - кузнечик

# In[12]:


def traj_num (N):
    """кузнечик прыгает из 1 в N с размером прыжка 1 или 2
    Расчитать количество траекторий допрыгать в N"""
    K = [0,1] + [0]*(N-1)
    print(K)
    for i in range (2, N+1):
        K[i] = K[i-2] +K[i-1]
    return K, K[N]
traj_num(10)


# #### Кузнечик с запретными клетками

# In[19]:


def count_traj(N:list, allowed:list):
    """Длина прыжка - 1 or 2 or 3,
    есть запретные клетки
    N - номер клетки
    allowed: list размерность N+1"""
    K = [0,1, int(allowed[2])] + [0]*(N-2)
    print (K)    
    for i in range (3, N+1):
        if allowed[i]: # выпускаем == True, т.к. это предполагается
            K[i] = K[i-1] + K[i-2] + K[i-3]
    return K[N]
count_traj(10, [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1])


# #### Минимальная стоимость достижения клетки N

# In[40]:


def count_min_cost (N, price:list):
    """Каждой клетке присваиваетася определенная стоимость
    ф-ция расчитывает минимальную стоимость достижения клетки N
    возможный шаг = 1 or 2
    C[i] = min cost (минимальная стоимость)"""
#     float("int") - обозначает "-бесконесность"
    C = [float("-inf"), price[1], price[1]+price[2]]+[0]*(N-2)
    for i in range (3, N+1):
        C[i] = price[i] + min(C[i-1], C[i-2])
    return C[N]

count_min_cost (8, [0, 3, 4, 9, 10, 5, 2, 1, 1])  


# #### Задача о грабителе

# In[11]:


def rubber (A:list):
    """Считает наибольшую сумму ряда, в котором запрещено вставать
    на 2 соседние клетки"""
    if (len (A)) == 0:
        return 0
    elif (len(A)) <3:
        return max(A)
    else:
        C = [A[0], A[1], A[2]+A[0]]+[0]*(len(A)-3)
        for i in range (3, len(A)):
            C[i] = A[i] + max(C[i-2],C[i-3])
        return max(C)


# #### Задача о грабителе. Более красивое решение из интернета

# In[134]:


def rob(nums:list):
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
rob ([3, 1, 4, 3, 1, 3, 4, 5, 6,7])


# ## Двумерное динамическое программирование

# #### Пробежка короля из клетки 1:1 в N:M

# In[31]:


def king_race (A:int, B:int):
    """Король бежит по траектории вправо и вниз по 1 клетке.
    Найти число возможных вариантов пробежки"""
    F = [[0]*(B+1) for i in range (A+1)]
    F[1][1] = 1
    for i in range(1, A+1):
        if i == 1:
            for j in range (2, B+1):
                F[i][j] = F[i][i-1]+F[i][j-1]
        else:
            for j in range (1, B+1):
                F[i][j] = F[i-1][j]+F[i][j-1]
    return F, F[-1][-1]
king_race(6, 5)


# #### Наибольшая общая подпоследовательность

# In[35]:


def lcs (A:list, B:list):
    """A, B - массивы,
    ф-ция ищет длину наибольшей подпоследовательность в массивах:
    Допустим нам известна длинна LCS A[i-1], B[j-1], тогда
    1) если a[i] == b[i], то добавляет 1 к счетчику длины общей подпосл.
    2) если a[i] != b[i], то ищет max из F[i][j-1], F[i-1][j]"""
    F = [[0]*(len(B)+1) for i in range (len(A)+1)]
    for i in range (1, len(A)+1):
        for j in range (1, len(B)+1):
            if A[i-1] == B[j-1]:
                F[i][j] = 1 + F[i-1][j-1]
            else:
                F[i][j] = max(F[i-1][j], F[i][j-1])
    return F, F[-1][-1]
lcs([ 5, 3, 8, 1, 1], [4, 5, 6, 3, 8, 1])


# #### Наибольшая возрастающая подпоследовательность

# In[80]:


def gis (A:list):
    """ищет наибольшую возрастающую последовательность в списке"""
    F = [0]*(len(A)+1)
    for i in range (1, len(A)+1):
        m = 0
        print('i=',i,end = '\n')
        for j in range (1, i):
            print('j=',j)
            print (A[j-1], A[i-1])
            print('F[j]=',F[j],'m=',m)
            if A[i-1] > A[j-1] and F[j] > m:
                m = F[j]
                print('m change +1')
        F[i] = m+1
        print('F=',F)
        print()
    return max(F), F
# gis([4, 5, 3, 8, 10])
gis([4,5,3])


# #### Редакционное расстояние между строками (Левенстайна)

# In[149]:


def levenstein (A:str, B:str):
    """Возвращает число необходимых поправок в строке.
    Вероятные поправки:замена, удаление, прибавление символа.
    1)Если символы равны - ред расст равно ред расст предыдущих символов
    2)Если не равны - 1 + ред расст минимальной предыдущей комбинации"""
    F = [[((i+j) if i*j == 0 else 0) for j in range (len(B)+1)] for i in range (len(A)+1)]
#     print(F)
    for i in range (1, len(A)+1):
        print(i)
        for j in range (1, len(B)+1):
            if A[i-1] == B[j-1]:
                print(A[i-1], '=', B[j-1])
                F[i][j] = F[i-1][j-1]
                print('изменяем c',F[i-1][j-1], 'на', F[i][j])
            else:
                F[i][j] = 1+ min(F[i-1][j], F[i][j-1], F[i-1][j-1])
                
    return F, F[len(A)][len(B)]

levenstein ("Pyth", "Peith")


# #### Проверка равенства строк

# In[107]:


def equal (A: str, B:str):
    """Проверяет на равенство строки A и B"""
    if len(A) != len(B):
        return False
    for i in range (len(A)):
        if A[i] != B[i]:
            return False
    return True
equal ("asasaa", "asdasd")


# ### Поиск подстроки в строке

# #### Наивный поиск

# In[122]:


def search_substring (s:str, sub:str):
    """Проверяет наличие подстроки sub в строке s"""
    for i in range (0, len(s) - len(sub)):
        if equal (s[i: i+len(sub)], sub):
            print ("индекс начала совпад элементов:",i )
    return False
search_substring("qwert", "qwq")


# ####  Префикс функция (П) строки

# In[150]:


def prefix(s:str):
    """Ищет наличие повторений максиально возможного префикса в стр.
    (сравнивает его с суффиксами)
    Возвращает число повторяющихся элементов"""
    F = [0]*len(s)
    for i in range(1,len(s)):
        k = F[i-1]
        while k > 0 and s[k] != s[i]:
            k = F[k-1]
        if s[k] == s[i]:
            k = k + 1
        F[i] = k
    return F
prefix("acbadbacbac")


# ## Лекция 13. Структура данных на Python

# #### Стек или очередь LIFO

# In[ ]:


"""clear()
    push(1)
    push(2)
    push(3)
    is_empty()
False
    pop()
3
    pop()
2
    pop()
1
    is_empty()
True"""


# In[169]:


_stack = []
def push(x):
    _stack.append(x)
    
def pop():
    x = _stack.pop()
    return x

def clear():
    _stack.clear()
    
def is_empty():
    return len(_stack) == 0

# if __name__ == "__main__":
#     import doctest
#     doctest.testmod()


# In[170]:


def is_braces_sequence_correct(s):
    """Проверяет корректность скобочной последовательности из 
    круглых и квадратных скобок () []
    
    >>>is_braces_sequence_correct("(([()]))[]")
    True
    >>>is_braces_sequence_correct ("([)[")
    False
    >>>is_braces_sequence_correct ("())")
    False
    """
    for brace in s:
        if brace not in "()[]":
            continue
        if brace in "([":
            A_stack.push(brace)
        else:
            assert brace in ")]", "ожидалась закрывающая скобка:" +str(brace)
            if A_stack.is_empty():
                return False
            left = A_stack.pop()
            assert left in "([", "Ожидалась открывающая скобка:" +str(brace)
            if left == "(":
                right = ")"
            elif left == "[":
                right = "]"
            if right != brace:
                return False
    return A_stack.is_empty()


# ## Лекция 14

# ## Heaps (Кучи)

# In[2]:


# https://aliev.me/runestone/Trees/BinaryHeapImplementation.html
# Более подробно по ссылке
class BinHeap:
    """Выстраивает кучу от min к max"""
    
    def __init__(self):
        """Создается массив с фиктивным нулем для хранения значений.
        0 используется для целочиселнного деления.
        Создается счетчик для хранения размера кучи."""
        self.heapList = [0]
        self.currentSize = 0


    def percUp(self,i):
        """поднимает добавленный элемент на нужное место"""
        while i // 2 > 0:
            if self.heapList[i] < self.heapList[i // 2]:
                tmp = self.heapList[i // 2]
                self.heapList[i // 2] = self.heapList[i]
                self.heapList[i] = tmp
            i = i // 2

    def insert(self,k):
        """Добавляет элемент"""
        self.heapList.append(k)
        self.currentSize = self.currentSize + 1
        self.percUp(self.currentSize)

    def percDown(self,i):
        """Пока мы в пределах длины кучи 
        берем мин. потомка из ф-ции minChild
        если i-й элемент больше потомка - меняем их местами"""
        while (i * 2) <= self.currentSize:
            mc = self.minChild(i)
            if self.heapList[i] > self.heapList[mc]:
                tmp = self.heapList[i]
                self.heapList[i] = self.heapList[mc]
                self.heapList[mc] = tmp
            i = mc

    def minChild(self,i):
        """Сравнивает между собой по 2 потомка 2 потомков, 
        возвращает индекс наименьшего"""
        if i * 2 + 1 > self.currentSize:
            return i * 2
        else:
            if self.heapList[i*2] < self.heapList[i*2+1]:
                return i * 2
            else:
                return i * 2 + 1

    def delMin(self):
        """Заменяет мин. знач. списка последним значением списка.
        Удаляет последнее значение.
        Возвращает замененное начальлное значение """
        retval = self.heapList[1]
        self.heapList[1] = self.heapList[self.currentSize]
        self.currentSize = self.currentSize - 1
        self.heapList.pop()
        self.percDown(1)
        return retval

    def buildHeap(self,alist):
        """Добавляет элементы из листа в кучу и сортирует их там"""
        i = len(alist) // 2
        self.currentSize = len(alist)
        self.heapList = [0] + alist[:]
        while (i > 0):
            self.percDown(i)
            i = i - 1
        return self.heapList

bh = BinHeap()
bh.buildHeap([9,5,6,2,3])
print(bh.buildHeap([9,5,6,2,3]))

print(bh.delMin())
print(bh.delMin())
print(bh.delMin())
print(bh.delMin())
print(bh.delMin())

