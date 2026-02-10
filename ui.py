import streamlit as st
import requests

st.set_page_config(page_title="Sentiment Analysis", layout="centered")

st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>Sentiment Analysis Web App</h1>", unsafe_allow_html=True)
st.write("---")

st.markdown("""
<div style='text-align: center; font-size: 16px; margin-bottom:12px'>
Enter any review or tweet below and get its sentiment analysis result instantly!
</div>
""", unsafe_allow_html=True)

user_input = st.text_area("Your Review or Tweet:", 
                          placeholder="Type your text here...", 
                          height=150)

if st.button('Analyze Sentiment', type="primary"):
    if user_input.strip() != "":
        url = 'http://127.0.0.1:8000/predict'  
        try:
            response = requests.post(url, json={'review': user_input})
            
            if response.status_code == 200:
                data = response.json()
                sentiment = data.get('sentiment')

                if sentiment == "Positive":
                    st.success(f"Positive")
                elif sentiment == "Negative":
                    st.error(f"Negative")
                else:
                    st.info(f"Neutral")
            else:
                st.error("Error: Unable to analyze sentiment. Try again.")
        except Exception as e:
            st.error(f"API Error: {str(e)}")
    else:
        st.warning("Please enter some text to analyze.")
