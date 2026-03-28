import random

def greet(name):
    """Print a greeting message to the specified name.
    
    Args:
        name: The name to greet.
    """
    print(f"Hello, {name}!")
    print(f"Your random number is: {random.randint(100, 999)}")
