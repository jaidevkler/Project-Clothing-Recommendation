import streamlit as st

st.set_page_config(
    page_title="Code & Couture",
    page_icon="👋",
)

col1, col2 = st.columns([3,7])
with col1:  
    st.image("Resources/streamlit/logo.jpeg")
with col2:
    st.write("# Welcome to Code & Couture! 👋")

st.image('Resources/streamlit/team.png')

st.markdown(
    """
    Introducing a multi-modal fashion search system that seamlessly integrates image and text-based queries to deliver personalized and enhanced fashion recommendations. Leveraging the power of Open AI, Google Search API, and (ADD TECHNOLOGY HERE) this innovative solution empowers users to discover their perfect fashion matches with unprecedented accuracy and efficiency.

With its advanced deep learning algorithms, the multi-modal fashion search system analyzes both the visual features of clothing items and the semantic information conveyed by text descriptions. By combining these inputs, it generates a comprehensive understanding of user preferences, enabling it to curate personalized and relevant fashion suggestions. Whether users are looking for specific styles, colors, patterns, or even celebrity-inspired outfits, this state-of-the-art system ensures an unparalleled shopping experience.
"""
)