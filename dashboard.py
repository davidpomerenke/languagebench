import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Language Bench")

st.write("## Results")


results = pd.read_json("results.json")

st.dataframe(results)

for language in results["target_language"].unique():
    st.write(f"## {language}")
    fig = px.bar(
        results[results["target_language"] == language],
        x="model",
        y="bleu",
        range_y=[0, 1],
    )
    st.plotly_chart(fig)
