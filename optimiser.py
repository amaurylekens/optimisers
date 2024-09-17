import numpy as np
import matplotlib.pyplot as plt

# Define a simple function of two variables to optimize
def f(x_1, x_2):
    return x_1**2 + x_2**2 + 5*x_1 + 3*x_2 + 10

# Gradient of the function
def grad_f(x_1, x_2):
    return np.array([2*x_1 + 5, 2*x_2 + 3])

# Define a function with several local minima
def f1(x_1, x_2):
    return np.sin(x_1) * np.cos(x_2) + np.exp((x_1**2 + x_2**2) / 20)

# Gradient of the function
def grad_f1(x_1, x_2):
    dx1 = np.cos(x_1) * np.cos(x_2) + x_1 * np.exp((x_1**2 + x_2**2) / 20) / 10
    dx2 = -np.sin(x_1) * np.sin(x_2) + x_2 * np.exp((x_1**2 + x_2**2) / 20) / 10
    return np.array([dx1, dx2])

# Adam optimizer implementation
class AdamOptimizer:
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = 0
        self.v = 0
        self.t = 0

    def update(self, theta, gradient):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        theta_new = theta - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return theta_new

def get_plot_range(f, margin=1.0):
    # Sample points to estimate a good plot range
    x = np.linspace(-10, 10, 1000)
    y = np.linspace(-10, 10, 1000)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    
    # Find min and max coordinates where the function is below a threshold
    threshold = np.min(Z) + 0.1 * (np.max(Z) - np.min(Z))
    mask = Z < threshold
    x_min, x_max = X[mask].min(), X[mask].max()
    y_min, y_max = Y[mask].min(), Y[mask].max()
    
    # Add margin
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= margin * x_range
    x_max += margin * x_range
    y_min -= margin * y_range
    y_max += margin * y_range
    
    return x_min, x_max, y_min, y_max

def test_adam_optimizer(f, grad_f):
    adam = AdamOptimizer(learning_rate=0.05)
    theta = np.array([10.0, 10.0])
    num_iterations = 500
    theta_history = [theta]
    z_history = [f(*theta)]

    for _ in range(num_iterations):
        gradient = grad_f(*theta)
        theta = adam.update(theta, gradient)
        theta_history.append(theta)
        z_history.append(f(*theta))

    theta_history = np.array(theta_history)

    # Get plot range
    x_min, x_max, y_min, y_max = get_plot_range(f)

    # Plot the results
    plt.figure(figsize=(18, 6))
    
    plt.subplot(131)
    plt.plot(range(num_iterations + 1), theta_history[:, 0], label='x_1')
    plt.plot(range(num_iterations + 1), theta_history[:, 1], label='x_2')
    plt.title('x_1 and x_2 vs. Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.legend()

    plt.subplot(132)
    plt.plot(range(num_iterations + 1), z_history)
    plt.title('f(x_1, x_2) vs. Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('f(x_1, x_2)')

    plt.subplot(133)
    x_1 = np.linspace(x_min, x_max, 100)
    x_2 = np.linspace(y_min, y_max, 100)
    X_1, X_2 = np.meshgrid(x_1, x_2)
    Z = f(X_1, X_2)
    plt.contour(X_1, X_2, Z, levels=50)
    plt.colorbar(label='f(x_1, x_2)')
    plt.plot(theta_history[:, 0], theta_history[:, 1], 'r-', label='Optimization path')
    plt.plot(theta_history[0, 0], theta_history[0, 1], 'go', label='Start')
    plt.plot(theta_history[-1, 0], theta_history[-1, 1], 'ro', label='End')
    plt.title('Optimization Path')
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.legend()

    plt.tight_layout()
    plt.show()

    print(f"Final x_1, x_2: {theta}")
    print(f"Final f(x_1, x_2): {f(*theta)}")

def compare_adam_betas(f, grad_f):
    beta_pairs = [
        (0.9, 0.999),  # Default values
        (0.1, 0.1),    # Very low beta values
        (0.5, 0.5),    # Moderate beta values
        (0.99, 0.9999),# High beta values
        (0.01, 0.99),  # Extreme combination: very low beta1, high beta2
        (0.99, 0.01)   # Extreme combination: high beta1, very low beta2
    ]
    
    # Get plot range
    x_min, x_max, y_min, y_max = get_plot_range(f)
    
    plt.figure(figsize=(20, 24))
    
    for i, (beta1, beta2) in enumerate(beta_pairs):
        adam = AdamOptimizer(learning_rate=0.05, beta1=beta1, beta2=beta2)
        theta = np.array([4.0, 4.0])  # Starting point
        num_iterations = 1000
        theta_history = [theta]
        m_hat_history = []
        v_hat_history = []

        for _ in range(num_iterations):
            gradient = grad_f(*theta)
            theta = adam.update(theta, gradient)
            theta_history.append(theta)
            m_hat = adam.m / (1 - adam.beta1 ** adam.t)
            v_hat = adam.v / (1 - adam.beta2 ** adam.t)
            m_hat_history.append(m_hat)
            v_hat_history.append(v_hat)

        theta_history = np.array(theta_history)
        m_hat_history = np.array(m_hat_history)
        v_hat_history = np.array(v_hat_history)

        plt.subplot(6, 3, 3*i + 1)
        x_1 = np.linspace(x_min, x_max, 200)
        x_2 = np.linspace(y_min, y_max, 200)
        X_1, X_2 = np.meshgrid(x_1, x_2)
        Z = f(X_1, X_2)
        plt.contour(X_1, X_2, Z, levels=50)
        plt.colorbar(label='f(x_1, x_2)')
        plt.plot(theta_history[:, 0], theta_history[:, 1], 'r-', label='Optimization path')
        plt.plot(theta_history[0, 0], theta_history[0, 1], 'go', label='Start')
        plt.plot(theta_history[-1, 0], theta_history[-1, 1], 'ro', label='End')
        plt.title(f'Adam: β1={beta1}, β2={beta2}')
        plt.xlabel('x_1')
        plt.ylabel('x_2')
        plt.legend()

        plt.subplot(6, 3, 3*i + 2)
        plt.plot(range(num_iterations), m_hat_history[:, 0], label='m_hat[0]')
        plt.plot(range(num_iterations), m_hat_history[:, 1], label='m_hat[1]')
        plt.title(f'm_hat evolution (β1={beta1}, β2={beta2})')
        plt.xlabel('Iterations')
        plt.ylabel('m_hat')
        plt.legend()

        plt.subplot(6, 3, 3*i + 3)
        plt.plot(range(num_iterations), v_hat_history[:, 0], label='v_hat[0]')
        plt.plot(range(num_iterations), v_hat_history[:, 1], label='v_hat[1]')
        plt.title(f'v_hat evolution (β1={beta1}, β2={beta2})')
        plt.xlabel('Iterations')
        plt.ylabel('v_hat')
        plt.legend()

        print(f"Final x_1, x_2 for β1={beta1}, β2={beta2}: {theta}")
        print(f"Final f(x_1, x_2): {f(*theta)}")

    plt.tight_layout()
    plt.show()

def analyze_adam_performance(beta_pairs, num_runs=10, num_iterations=1000):
    results = []
    
    for beta1, beta2 in beta_pairs:
        run_results = []
        for _ in range(num_runs):
            adam = AdamOptimizer(learning_rate=0.05, beta1=beta1, beta2=beta2)
            theta = np.array([4.0, 4.0])  # Starting point
            
            for _ in range(num_iterations):
                gradient = grad_f1(*theta)
                theta = adam.update(theta, gradient)
            
            final_value = f1(*theta)
            run_results.append(final_value)
        
        results.append(run_results)
    
    return results

def plot_performance_comparison(beta_pairs, results):
    plt.figure(figsize=(12, 6))
    plt.boxplot(results, labels=[f'β1={b1}, β2={b2}' for b1, b2 in beta_pairs])
    plt.title('Adam Performance Comparison')
    plt.ylabel('Final f(x_1, x_2) value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def compare_adam_betas():
    beta_pairs = [
        (0.9, 0.999),  # Default values
        (0.1, 0.1),    # Very low beta values
        (0.5, 0.5),    # Moderate beta values
        (0.99, 0.9999),# High beta values
        (0.01, 0.99),  # Extreme combination: very low beta1, high beta2
        (0.99, 0.01)   # Extreme combination: high beta1, very low beta2
    ]
    
    # ... (keep the existing visualization code)

    # Add performance analysis
    print("\nPerforming statistical analysis...")
    results = analyze_adam_performance(beta_pairs)
    plot_performance_comparison(beta_pairs, results)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    for i, (beta1, beta2) in enumerate(beta_pairs):
        mean_value = np.mean(results[i])
        std_value = np.std(results[i])
        min_value = np.min(results[i])
        max_value = np.max(results[i])
        print(f"β1={beta1}, β2={beta2}:")
        print(f"  Mean: {mean_value:.4f}")
        print(f"  Std Dev: {std_value:.4f}")
        print(f"  Min: {min_value:.4f}")
        print(f"  Max: {max_value:.4f}")
        print()

# Run the tests
test_adam_optimizer(f, grad_f)
test_adam_optimizer(f1, grad_f1)

# Run the comparisons
compare_adam_betas(f, grad_f)
compare_adam_betas(f1, grad_f1)
