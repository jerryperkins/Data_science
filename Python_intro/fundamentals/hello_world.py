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
    
