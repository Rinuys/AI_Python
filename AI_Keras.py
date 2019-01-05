import numpy as np
# 학습할 수식의 최대 길이 설정
# 두자릿수 + 두자릿수의 경우 온점을 포함 최대 10개 문자가 올 수 있음
max_len = 10
# 미리 테스트에 사용할 수식들을 정의해
# 학습 데이터에서 제외
test_str_list = ['10+10=20.', '99+99=198.', '13+24=37.']
# 두자릿수 덧셈에서 발생 가능한 모든 수식을 생성
# (테스트 데이터 제외)
train_str_list = []
for i in range(1, 100):
for j in range(1,100):
a = str(i)
b = str(j)
c = str(i+j)
sentence = a +'+'+b+'='+c+'.'