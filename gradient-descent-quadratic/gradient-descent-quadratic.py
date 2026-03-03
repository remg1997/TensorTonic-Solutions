def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    initial = x0
    for _ in range(steps):
        f_prime = 2*a*initial + b
        initial = initial - (1*lr*f_prime)

    return initial
        
        