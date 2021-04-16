# Biggie Size - Given a list, write a function that changes all positive numbers in the list to "big".
# Example: biggie_size([-1, 3, 5, -5]) returns that same list, but whose values are now [-1, "big", "big", -5]

def biggie_size(arr):
    answer = []
    for x in range(0,len(arr), 1):
        if arr[x] > 0:
            answer.append("big")
        else:
            answer.append(arr[x])
    return answer
print(biggie_size([-1, 0, 5, -5]))


# Count Positives - Given a list of numbers, create a function to replace the last value with the number of positive values. (Note that zero is not considered to be a positive number).
# Example: count_positives([-1,1,1,1]) changes the original list to [-1,1,1,3] and returns it
# Example: count_positives([1,6,-4,-2,-7,-2]) changes the list to [1,6,-4,-2,-7,2] and returns it

def count_positives(arr):
    count = 0
    for x in range(0,len(arr), 1):
        if arr[x] > 0:
            count += 1
    arr[len(arr)-1] = count
    return arr
print(count_positives([1,6,-4,-2,-7,-2]))

# Sum Total - Create a function that takes a list and returns the sum of all the values in the array.
# Example: sum_total([1,2,3,4]) should return 10
# Example: sum_total([6,3,-2]) should return 7

def sum_total(arr):
    sum = 0
    for x in range(len(arr)):
        sum += arr[x]
    return sum
print(sum_total([6,3,-2]))

# Average - Create a function that takes a list and returns the average of all the values.
# Example: average([1,2,3,4]) should return 2.5

def average(arr):
    sum = 0
    for x in range(len(arr)):
        sum += arr[x]
    avg = sum / len(arr)
    return avg
print(average([1,2,3,10]))

# Length - Create a function that takes a list and returns the length of the list.
# Example: length([37,2,1,-9]) should return 4
# Example: length([]) should return 0

def length(arr):
    return len(arr)
print(length([37,2,1,-9]))

# Minimum - Create a function that takes a list of numbers and returns the minimum value in the list. (Optional) If the list is empty, have the function return False.
# Example: minimum([37,2,1,-9]) should return -9
# (Optional) Example: minimum([]) should return False

def minimum(arr):
    if len(arr) == 0:
        return False
    min = arr[0]
    for x in range(1, len(arr), 1):
        if arr[x] < min:
            min = arr[x]
    return min
print(minimum([37,2,1,-9]))

# Maximum - Create a function that takes a list and returns the maximum value in the array. (Optional) If the list is empty, have the function return False.
# Example: maximum([37,2,1,-9]) should return 37
# (Optional) Example: maximum([]) should return False

def maximum(arr):
    if len(arr) == 0:
        return False
    max = arr[0]
    for x in range(1, len(arr), 1):
        if arr[x] > max:
            max = arr[x]
    return max
print(maximum([37,2,1,-9]))

# Ultimate Analysis (Optional) - Create a function that takes a list and returns a dictionary that has the sumTotal, average, minimum, maximum and length of the list.
# Example: ultimate_analysis([37,2,1,-9]) should return {'sumTotal': 31, 'average': 7.75, 'minimum': -9, 'maximum': 37, 'length': 4 }

def ultimate_analysis(arr):
    answer = {}
    min = arr[0]
    max = arr[0]
    sum = 0
    answer['length'] = len(arr)
    for x in range(len(arr)):
        if arr[x] < min:
            min = arr[x]
        elif arr[x] > max:
            max = arr[x]
        sum += arr[x]
    answer['min'] = min
    answer['max'] = max
    answer['sum'] = sum
    answer['avg'] = sum / len(arr)
    return answer

print(ultimate_analysis([37,2,1,-9]))

# Reverse List (Optional) - Create a function that takes a list and return that list with values reversed. Do this without creating a second list. (This challenge is known to appear during basic technical interviews.)
# Example: reverse_list([37,2,1,-9]) should return [-9,1,2,37]

def reverse_list(arr):
    for x in range(0, len(arr) // 2,1):
        temp = arr[x]
        arr[x] = arr[len(arr) - 1 - x]
        arr[len(arr) - 1 - x] = temp
    return arr
print(reverse_list([37,2,5,9,1,-9]))

