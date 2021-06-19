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


print("Select operation.")
print("1.Add")
print("2.Subtract")
print("3.Multiply")
print("4.Divide")
print("5.Display saved results")

# Stores the last 5 results
previous_results = []

continue_calculating = True

while continue_calculating:

    # Take input from the user
    choice = input("Enter choice(1/2/3/4/5): ")

    # Check if choice is one of the four options
    if choice in ('1', '2', '3', '4'):
        num1 = float(input("Enter first number: "))
        num2 = float(input("Enter second number: "))

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

    # Prints the list
    elif choice == '5':
        print(previous_results)

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
