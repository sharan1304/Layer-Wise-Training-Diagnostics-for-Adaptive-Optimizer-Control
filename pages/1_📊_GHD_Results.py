"""Sidebar page: 📊 GHD Results."""

import streamlit as st

st.set_page_config(page_title="GHD Results", page_icon="📊", layout="wide")

from ghd_results_page import render  # noqa: E402

render()
