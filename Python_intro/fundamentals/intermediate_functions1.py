
import random

def rand_int(min=0, max=100):
    if min > max:
        return "min can't be larger than max"
    
    answer = (round(random.random() * (max-min) + min))
    return answer
print(rand_int(max=50, min=50))

