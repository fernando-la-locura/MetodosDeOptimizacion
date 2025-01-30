import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative
import streamlit as st

class ConvexityAnalyzer:
    def __init__(self, f, domain, name="f(x)"):
        """
        Initialize the convexity analyzer with a function and its domain.
        
        Parameters:
        f (callable): The function to analyze
        domain (tuple): (min_x, max_x) for the analysis
        name (str): Name of the function for display
        """
        self.f = f
        self.domain = domain
        self.name = name
        
    def second_derivative(self, x):
        """Calculate the second derivative at point x."""
        return derivative(self.f, x, n=2, dx=1e-6)
    
    def check_convexity_by_second_derivative(self, num_points=1000):
        """
        Check convexity using the second derivative criterion.
        Returns True if f''(x) ≥ 0 for all tested points.
        """
        x_vals = np.linspace(self.domain[0], self.domain[1], num_points)
        second_derivs = [self.second_derivative(x) for x in x_vals]
        is_convex = all(d >= -1e-10 for d in second_derivs)  # Small tolerance for numerical errors
        return is_convex, x_vals, second_derivs
    
    def check_convexity_by_definition(self, num_points=20):
        """
        Check convexity using the direct definition with random points.
        Returns percentage of tests that satisfy convexity condition.
        """
        successful_tests = 0
        total_tests = num_points * (num_points - 1) // 2
        x_vals = np.linspace(self.domain[0], self.domain[1], num_points)
        
        tests_results = []
        for i in range(num_points):
            for j in range(i + 1, num_points):
                x1, x2 = x_vals[i], x_vals[j]
                lambda_val = np.random.random()
                
                # Calculate convex combination point
                x_combined = lambda_val * x1 + (1 - lambda_val) * x2
                
                # Check if convexity condition holds
                f_combined = self.f(x_combined)
                f_convex_comb = lambda_val * self.f(x1) + (1 - lambda_val) * self.f(x2)
                
                is_convex = f_combined <= f_convex_comb + 1e-10
                if is_convex:
                    successful_tests += 1
                    
                tests_results.append({
                    'x1': x1,
                    'x2': x2,
                    'lambda': lambda_val,
                    'convex': is_convex
                })
                
        return successful_tests / total_tests, tests_results
    
    def plot_function_and_convexity(self):
        """Generate plots demonstrating the function's convexity."""
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot the function
        x = np.linspace(self.domain[0], self.domain[1], 1000)
        y = [self.f(xi) for xi in x]
        ax1.plot(x, y, 'b-', label=self.name)
        ax1.set_title(f"Function: {self.name}")
        ax1.grid(True)
        ax1.legend()
        
        # Plot second derivative
        is_convex, x_vals, second_derivs = self.check_convexity_by_second_derivative()
        ax2.plot(x_vals, second_derivs, 'r-', label="f''(x)")
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax2.set_title("Second Derivative")
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        return fig

    def demonstrate_convexity_with_points(self, x1, x2, lambda_val):
        """
        Demonstrate convexity by plotting a specific example with two points.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot the function
        x = np.linspace(self.domain[0], self.domain[1], 1000)
        y = [self.f(xi) for xi in x]
        ax.plot(x, y, 'b-', label=self.name)
        
        # Calculate convex combination point
        x_combined = lambda_val * x1 + (1 - lambda_val) * x2
        y_combined = self.f(x_combined)
        y_convex_comb = lambda_val * self.f(x1) + (1 - lambda_val) * self.f(x2)
        
        # Plot points and line
        ax.plot([x1, x2], [self.f(x1), self.f(x2)], 'r--', label='Chord')
        ax.plot([x1, x2], [self.f(x1), self.f(x2)], 'ro', label='Points')
        ax.plot([x_combined], [y_combined], 'go', label='f(λx₁ + (1-λ)x₂)')
        ax.plot([x_combined], [y_convex_comb], 'mo', label='λf(x₁) + (1-λ)f(x₂)')
        
        ax.grid(True)
        ax.legend()
        ax.set_title(f"Convexity Demonstration (λ={lambda_val:.2f})")
        
        return fig

def main():
    st.title("Function Convexity Analyzer")
    
    # Define example functions
    functions = {
        "x²": (lambda x: x**2, (-5, 5)),
        "ex": (lambda x: np.exp(x), (-2, 2)),
        "x⁴": (lambda x: x**4, (-3, 3)),
        "log(x)": (lambda x: np.log(x), (0.1, 5))
    }
    
    # Function selection
    selected_function = st.selectbox(
        "Select a function to analyze:",
        list(functions.keys())
    )
    
    f, domain = functions[selected_function]
    analyzer = ConvexityAnalyzer(f, domain, selected_function)
    
    # Analyze convexity
    is_convex, x_vals, second_derivs = analyzer.check_convexity_by_second_derivative()
    convexity_score, tests = analyzer.check_convexity_by_definition()
    
    # Display results
    st.write(f"### Analysis Results for {selected_function}")
    st.write(f"Second Derivative Test: {'Convex' if is_convex else 'Not Convex'}")
    st.write(f"Definition Test Score: {convexity_score:.2%} of tests passed")
    
    # Plot function and second derivative
    fig = analyzer.plot_function_and_convexity()
    st.pyplot(fig)
    
    # Interactive convexity demonstration
    st.write("### Interactive Convexity Demonstration")
    col1, col2, col3 = st.columns(3)
    with col1:
        x1 = st.slider("x₁", float(domain[0]), float(domain[1]), float(domain[0]))
    with col2:
        x2 = st.slider("x₂", float(domain[0]), float(domain[1]), float(domain[1]))
    with col3:
        lambda_val = st.slider("λ", 0.0, 1.0, 0.5)
    
    demo_fig = analyzer.demonstrate_convexity_with_points(x1, x2, lambda_val)
    st.pyplot(demo_fig)

if __name__ == "__main__":
    main()