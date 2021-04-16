print("hello world")

name = "noelle"
print("hello", name, "!")
print("hello " + name + "!")
number = 42
print("hello", 42)
# print("hello" + 42)
fav_food1 = "sushi"
fav_food2 = "pizza"
print("I love to eat {} and {}" .format(fav_food1, fav_food2))
print(f"I love to eat {fav_food1} and {fav_food2}")


new_person = {'name': 'John', 'age': 38, 'weight': 160.2, 'has_glasses': False}
print(new_person)
print(len(new_person['name']))

# y = 0
# x = 5
# for x in range (0,10,1):
#     print(x)
#     y += 1
# print(y)

my_list = ["abc", 123, "xyz"]

for i in range(0, len(my_list), 1):
    print(i, my_list[i])

for k, v in new_person.items():
    print(k, " = ", v)

count = 0
while count < 5:
    print("looping - ", count)
    count += 1
else:
    print("we are don")
    
def add(a,b):
    x = a+b
    return x
print(add(3,4))

def say_hi(name):
    greeting = "Hi " + name
    return greeting
test = say_hi("Jerry")
print(test)

sum1 = add(3,2)
sum2 = add(12,3)
sum3 = sum1 + sum2
print(sum3)

print(5//3)

def be_cheerful(name="", repeat=2):
    print(f"food morning {name}\n" * repeat)
be_cheerful()
be_cheerful("tim")
be_cheerful("tim", 4)
be_cheerful(repeat=3,name="Jerry")
be_cheerful(name="Robbert")