"""
747. Largest Number At Least Twice of Others

In a given integer array nums, there is always exactly one largest element.

Find whether the largest element in the array is at least twice as much as every other number in the array.

If it is, return the index of the largest element, otherwise return -1.

Input: nums = [3, 6, 1, 0]
Output: 1
Explanation: 6 is the largest integer, and for every other number in the array x,
6 is more than twice as big as x.  The index of value 6 is 1, so we return 1.

"""

def dominantIndex(nums):
    
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

print(dominantIndex([10,4,5,3,0]) == 0,dominantIndex([]) == -1,dominantIndex([5,6]) == -1,dominantIndex([5]) == 0)


