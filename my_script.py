import streamlit as st
import matplotlib.pyplot as plt

# Login
password = st.text_input("Enter password", type="password")
if password != "1234":
    st.stop()

# Create a simple plot
plt.plot([1,2,3,4], [10,20,15,30])
st.pyplot(plt)
