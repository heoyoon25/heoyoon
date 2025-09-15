import streamlit as st
st.title('A,B 숫자 합하기')
st.write('A와 B의 값을 더해줍니다.')
A = int(input('첫번째 숫자: '))
B = int(input('두번째 숫자: '))
result = A+B
print('두 숫자의 합은?',result)
