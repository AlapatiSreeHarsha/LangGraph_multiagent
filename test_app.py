import streamlit as st
import math
import numpy as np

# Set Streamlit page configuration
st.set_page_config(page_title="Scientific Calculator", page_icon="🧮", layout="centered")

# Title and Introduction
st.title("🧮 Scientific Calculator")
st.write("Perform basic, advanced mathematical operations, and solve equations!")

# Sidebar for user input
st.sidebar.title("Select Operation")
operation = st.sidebar.selectbox(
    "Choose a mathematical operation:",
    [
        "Addition",
        "Subtraction",
        "Multiplication",
        "Division",
        "Power",
        "Square Root",
        "Logarithm",
        "Trigonometry (sin, cos, tan)",
        "Solve n-th Degree Equation"
    ]
)

# Input Fields
st.write("### Enter Input Values")
if operation in ["Addition", "Subtraction", "Multiplication", "Division", "Power"]:
    num1 = st.number_input("Enter the first number:", value=0.0)
    num2 = st.number_input("Enter the second number:", value=0.0)

elif operation == "Square Root":
    num1 = st.number_input("Enter the number to find the square root:", value=0.0)

elif operation == "Logarithm":
    num1 = st.number_input("Enter the number:", value=0.0)
    base = st.number_input("Enter the base (default: 10):", value=10.0)

elif operation == "Trigonometry (sin, cos, tan)":
    angle = st.number_input("Enter the angle in degrees:", value=0.0)

elif operation == "Solve n-th Degree Equation":
    st.write("Enter the coefficients of the polynomial equation:")
    st.write("Example: For \(2x^3 - 4x^2 + 3x - 5 = 0\), enter coefficients as `2, -4, 3, -5`")
    coefficients = st.text_input("Enter coefficients (comma-separated):", value="")

# Perform Calculation
st.write("### Result")
try:
    if operation == "Addition":
        result = num1 + num2
        st.success(f"The sum of {num1} and {num2} is {result}")
    elif operation == "Subtraction":
        result = num1 - num2
        st.success(f"The difference between {num1} and {num2} is {result}")
    elif operation == "Multiplication":
        result = num1 * num2
        st.success(f"The product of {num1} and {num2} is {result}")
    elif operation == "Division":
        result = num1 / num2
        st.success(f"The division of {num1} by {num2} is {result}")
    elif operation == "Power":
        result = math.pow(num1, num2)
        st.success(f"{num1} raised to the power of {num2} is {result}")
    elif operation == "Square Root":
        result = math.sqrt(num1)
        st.success(f"The square root of {num1} is {result}")
    elif operation == "Logarithm":
        result = math.log(num1, base)
        st.success(f"The logarithm of {num1} with base {base} is {result}")
    elif operation == "Trigonometry (sin, cos, tan)":
        rad = math.radians(angle)
        sin_val = math.sin(rad)
        cos_val = math.cos(rad)
        tan_val = math.tan(rad)
        st.success(f"sin({angle}) = {sin_val:.4f}")
        st.success(f"cos({angle}) = {cos_val:.4f}")
        st.success(f"tan({angle}) = {tan_val:.4f}")
    elif operation == "Solve n-th Degree Equation":
        if coefficients:
            coeff_list = [float(c.strip()) for c in coefficients.split(",")]
            roots = np.roots(coeff_list)
            st.success(f"The roots of the equation are: {', '.join(map(str, roots))}")
        else:
            st.warning("Please enter the coefficients to solve the equation.")
except Exception as e:
    st.error(f"Error: {e}")

# Footer
st.write("---")
st.write("Designed with ❤️ by Harsha")
