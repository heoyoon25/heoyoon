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


st.markdown('---')
st.header('선택된 숫자까지의 합계 계산')

col1, col2 = st.columns(2)

with col1:
    num_a = st.number_input('첫 번째 숫자 (A)를 입력하세요:', min_value=1, step=1, key='num_a')
with col2:
    num_b = st.number_input('두 번째 숫자 (B)를 입력하세요:', min_value=1, step=1, key='num_b')

options = [num_a, num_b]
selected_num = st.selectbox('1부터 합을 계산할 숫자를 선택하세요:',options)

def calculate_sum(n):
    return sum(range(1, n + 1))

if selected_num:
    result = calculate_sum(selected_num)
    st.success(f'1부터 **{selected_num}**까지의 모든 정수 합은 **{result}**입니다.')
