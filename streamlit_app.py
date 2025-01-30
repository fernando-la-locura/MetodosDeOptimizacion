import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sympy import symbols, diff, lambdify

def is_convex(f_expr, x_symbol):
    """
    Verifica la convexidad de una función usando la segunda derivada.
    """
    second_derivative = diff(f_expr, x_symbol, 2)
    second_derivative_func = lambdify(x_symbol, second_derivative, 'numpy')
    return second_derivative_func

def plot_function(f_expr, x_range=(-5, 5)):
    """
    Grafica la función y su segunda derivada.
    """
    x = np.linspace(x_range[0], x_range[1], 400)
    x_symbol = symbols('x')
    f_func = lambdify(x_symbol, f_expr, 'numpy')
    second_derivative_func = is_convex(f_expr, x_symbol)
    
    y = f_func(x)
    y2 = second_derivative_func(x)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Función f(x)'))
    fig.add_trace(go.Scatter(x=x, y=y2, mode='lines', name='Segunda Derivada f"(x)', line=dict(dash='dot')))
    
    fig.update_layout(title='Gráfico de la Función y su Segunda Derivada', xaxis_title='x', yaxis_title='f(x)')
    
    return fig, y2

# Configuración de la página
st.set_page_config(page_title='Verificador de Convexidad', page_icon='📈', layout='wide')

st.title('📈 Verificador de Convexidad de Funciones')
st.markdown("""Ingrese una función matemática para verificar si es convexa basándose en su segunda derivada.""")

# Entrada de función
f_input = st.text_input("Ingrese la función f(x):", "x**2")
x_symbol = symbols('x')
try:
    f_expr = eval(f_input, {"x": x_symbol, "np": np})
    fig, second_derivative_vals = plot_function(f_expr)
    
    st.plotly_chart(fig)
    
    if np.all(second_derivative_vals >= 0):
        st.success("La función es convexa en el rango dado.")
    else:
        st.warning("La función no es convexa en todo el rango dado.")
except Exception as e:
    st.error(f"Error al interpretar la función: {e}")
