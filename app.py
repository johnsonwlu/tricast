"""Tricast entry point: page configuration and navigation.

The pages live in `views/` rather than Streamlit's magic `pages/` folder so
they can be given proper plain-English names in the sidebar — the automatic
folder-based navigation would label this file "app" and the others by their
filenames.
"""

import streamlit as st

st.set_page_config(
    page_title="Tricast",
    layout="wide",
    initial_sidebar_state="expanded",
)

navigation = st.navigation([
    st.Page("views/watchlist.py", title="Your watchlist",
            url_path="watchlist", default=True),
    st.Page("views/stock_detail.py", title="One stock in detail",
            url_path="stock"),
    st.Page("views/economy.py", title="The economy",
            url_path="economy"),
    st.Page("views/track_record.py", title="Track record",
            url_path="track-record"),
])
navigation.run()
