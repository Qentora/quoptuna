import streamlit as st


def main_page():
    """This is the main page of the app."""
    st.markdown(
        '<div class="main-title">'
        "QuOptuna: Optimizing Quantum Models with Optuna</div>"
    )
    st.markdown(
        '<div class="description">'
        "Welcome to QuOptuna! This app helps you optimize quantum models using Optuna."
        "The plots will be updated shortly.</div>",
        unsafe_allow_html=True,
    )
