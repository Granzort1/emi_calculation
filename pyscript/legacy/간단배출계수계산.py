import pandas as pd
import numpy as np
import re
from math import exp


def evaluate_formula(formula, v_value):
    # 입력값이 숫자인 경우 그대로 반환
    if isinstance(formula, (int, float)):
        return formula

    # 문자열이 아닌 경우 문자열로 변환
    formula = str(formula)

    # E 표기법을 파이썬 표현으로 변환
    formula = formula.replace('^', '**')

    # EXP를 exp로 변환 (대소문자 모두 처리)
    formula = formula.replace('EXP', 'exp')
    formula = formula.replace('exp', 'exp')

    # 특수 마이너스 기호를 일반 빼기 기호로 변환
    formula = formula.replace('–', '-')  # en dash
    formula = formula.replace('—', '-')  # em dash
    formula = formula.replace('−', '-')  # minus sign

    # V를 실제 값으로 대체
    formula = formula.replace('V', str(v_value))

    # 공백 제거
    formula = formula.replace(' ', '')

    try:
        # 계산 실행
        result = eval(formula)
        return result
    except Exception as e:
        print(f"Error processing formula: {formula}")
        print(f"Error message: {str(e)}")
        return None


# 엑셀 파일 읽기 - 모든 열을 문자열로 읽기
file_path = r"C:\dense_traffic_emi\factor\factor.xlsx"
df = pd.read_excel(file_path, dtype={'배출계수': str})

# VOC 행만 필터링
voc_df = df[df['물질'] == 'VOC'].copy()

# 계산할 V 값들 정의 (예: 10부터 60까지 10단위로)
v_values = [40, 48, 64]

# 각 V 값에 대해 계산하여 새로운 열 추가
for v in v_values:
    column_name = f'V={v} 적용 배출계수'
    voc_df[column_name] = voc_df['배출계수'].apply(
        lambda x: evaluate_formula(x, v)
    )

# 계산된 값들을 기준으로 정렬 (V=48 값 기준)
voc_df_sorted = voc_df.sort_values(by='V=48 적용 배출계수', ascending=False)

# 결과를 새로운 엑셀 파일로 저장
output_path = r"C:\dense_traffic_emi\factor\factor_calculated_VOC_only.xlsx"
voc_df_sorted.to_excel(output_path, index=False)

print("\n계산이 완료되었습니다.")
print(f"\n다음 V 값들에 대한 계산이 수행되었습니다: {v_values}")
print(f"\nVOC 데이터 수: {len(voc_df)} 행")
print("\n계산된 결과 샘플 (처음 5행):")
print(voc_df_sorted.head())