import streamlit as st
st.title('연산')

st.header('두 숫자 더하기')
col1, col2 = st.columns(2)

with col1:
    A = st.text_input('첫 번째 숫자 (A)를 입력하세요:', key='A_input')
with col2:
    B = st.text_input('두 번째 숫자 (B)를 입력하세요:', key='B_input')
try:
    num_A = int(A)
    num_B = int(B)
    sum_result = num_A + num_B
    st.success(f'두 수의 합은?{sum_result}')

except ValueError:
    st.error('유효한 숫자를 입력해 주세요')
