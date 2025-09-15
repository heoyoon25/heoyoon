import streamlit as st
st.title('연산')

st.header('A와 B 더하기')
A = st.text_input('A를 입력하세요:', key= 'A_input')
B = st.text_input('B를 입력하세요:', key= 'B_input')
result = A + B
st.success(f'A와 B의 합은?{result}')
