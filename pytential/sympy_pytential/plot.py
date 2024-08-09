import matplotlib.pyplot as plt

def plot_fcn(pytentials, args):
    # Plot the function
    plt.plot(x_values, y_values, label='fa.fcn(x)')
    plt.xlabel('x')
    plt.ylabel('fa.fcn(x)')
    plt.title('Plot of fa.fcn over a range of values')
    plt.legend()
    plt.grid(True)
    plt.show()
    # Define the range of values
    start = 0
    end = 2 * np.pi
    num_points = 100
    x_values = np.linspace(start, end, num_points)

    # Evaluate the function over the range of values
    y_values = fcn(x_values)

    # Plot the function
    plt.plot(x_values, y_values, label='fa.fcn(x)')
    plt.xlabel('x')