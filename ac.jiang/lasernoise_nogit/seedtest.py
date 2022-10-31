import random

print ("Random number with seed 30")
random.seed(30)
print ("first Number ", random.randint(25,50))

random.seed(31)
print ("Second Number ", random.randint(25,50))

random.seed(32)
print ("Third Number ", random.randint(25,50))
