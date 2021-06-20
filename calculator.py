import numpy as np


# This function adds two numbers
def add(x, y):
    return x + y


# This function subtracts two numbers
def subtract(x, y):
    return x - y


# This function multiplies two numbers
def multiply(x, y):
    return x * y


# This function divides two numbers
def divide(x, y):
    return x / y


def simple_calculations(num1, num2, choice):
    # Calls the appropriate function and stores the result in a list
    if choice == '1':
        result = add(num1, num2)
        previous_results.append(result)
        print(result)

    elif choice == '2':
        result = subtract(num1, num2)
        previous_results.append(result)
        print(result)

    elif choice == '3':
        result = multiply(num1, num2)
        previous_results.append(result)
        print(result)

    elif choice == '4':
        result = divide(num1, num2)
        previous_results.append(result)
        print(result)


def scientific_calculations(num1, choice):
    if choice == '6':
        type = input('Choose sin/cos/tan : s/c/t')
        if type == 's':
            result = np.sin(num1)
            previous_results.append(result)
            print(result)

        elif type == 'c':
            result = np.cos(num1)
            previous_results.append(result)
            print(result)

        elif type == 't':
            result = np.tan(num1)
            previous_results.append(result)
            print(result)

    elif choice == '7':
        result = num1 ** 2
        previous_results.append(result)
        print(result)

    elif choice == '8':
        result = np.sqrt(num1)
        previous_results.append(result)
        print(result)


print("Select operation.")
print("1.Add")
print("2.Subtract")
print("3.Multiply")
print("4.Divide")
print("5.Display saved results")
print("6.Calculate Angle")
print("7.Square")
print("8.Square Root")

# Stores the last 5 results
previous_results = []

continue_calculating = True

while continue_calculating:

    # Take input from the user
    choice = input("Enter choice(1/2/3/4/5/6/7/8): ")

    # Check if choice is one of the four options
    if choice in ('1', '2', '3', '4'):
        num1 = float(input("Enter first number: "))
        num2 = float(input("Enter second number: "))

        simple_calculations(num1, num2, choice)

    # Prints the list
    elif choice == '5':
        print(previous_results)

    elif choice in ('6', '7', '8'):
        num1 = float(input("Enter a number: "))

        scientific_calculations(num1, choice)

    else:
        print("Invalid Input")

    user_continue = input('Do you want to continue? : y/n')

    # Checks if the user wants to continue or exit
    if user_continue == 'y':
        continue_calculating = True
    else:
        continue_calculating = False

    # Ensures there are only 5 values in memory
    if len(previous_results) > 5:
        previous_results.pop()
