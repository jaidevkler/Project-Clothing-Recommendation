import streamlit as st

# Page config
st.set_page_config(
    page_title="Code & Couture",
    page_icon="👋")

# Create two columns
col1, col2 = st.columns([3,7])
# Logo
with col1:  
    st.image("Resources/streamlit/logo.jpeg")
# Welcome message
with col2:
    st.write("# Welcome to Code & Couture! 👋")

# Image of the team
st.image('Resources/streamlit/team.png')

# Message
st.markdown(
    """
    Introducing a multi-modal fashion search system that seamlessly integrates image and text-based queries to deliver personalized and enhanced fashion recommendations. Leveraging the power of U-Net model with a MobileNetV2 backbone, Open AI GPT-4o, Gemini 1.5 Pro and Google Shopping Search API this innovative solution empowers users to discover their perfect fashion matches with  unprecedented accuracy and efficiency.

With its advanced deep learning algorithms, the multi-modal fashion search system analyzes both the visual features of clothing items and the semantic information conveyed by text descriptions. By combining these inputs, it generates a comprehensive understanding of user preferences, enabling it to curate personalized and relevant fashion suggestions. Whether users are looking for specific styles, colors, patterns, or even celebrity-inspired outfits, this state-of-the-art system ensures an unparalleled shopping experience.
"""
)