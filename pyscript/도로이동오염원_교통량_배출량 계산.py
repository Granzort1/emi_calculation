import numpy as np
import pandas as pd
import matplotlib





# 1.1) 배기가스 - 엔진가열

Eij = VKT * (EFi/1000) * DF * (1- R/100)

#(1) VKT는 주행거리로써 상의후 결정



'''
(2) EFi는 배출계수(g/km) 
"경기도 차량등록대수의 차종 비율 사용(동일비율 가정) ->  측정한 차량대수를 등록된 차종별 차량수로 분류
 -> 차종별 연료비율 사용해 연료 분류 -> 차종-연료별 연식비율 사용해 연식 분류 -> 각 차종,연료,연식에 따라 배출계수 적용(SCC코드로 적용)


'''

#저감장치 제거율=>DPF 사용가정, 전국 부착률 50%가정
R_co, R_voc, R_pm = 99.5, 90, 83.6



def calculate_Engine_HotStart_EFi():












