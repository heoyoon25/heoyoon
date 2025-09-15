import streamlit as st
st.title('연산')

st.header('A와 B 더하기')
A = st.text_input('A를 입력하세요:', key= 'A_input')
B = st.text_input('B를 입력하세요:', key= 'B_input')
num_A = float(A)
num_B = float(B)
result = num_A + num_B
st.success(f'A와 B의 합은?{result}')
