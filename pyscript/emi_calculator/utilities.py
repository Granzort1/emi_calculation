import re
import math
import pandas as pd
import unicodedata

def calculate_cold_hot_ratio(fuel, pollutant, T):
    # 엔진미가열 대비 엔진가열 배출 비율 계산 함수
    fuel = fuel.upper()
    T = float(T)
    if T > 30:
        T = 30
    ratio = None

    if fuel == '휘발유':
        # Closed loop Gasoline Powered Vehicles (자동제어)
        if pollutant == 'CO':
            ratio = 9.04 - 0.09 * T
        elif pollutant == 'NOx':
            ratio = 3.66 - 0.006 * T
        elif pollutant == 'VOC':
            ratio = 12.59 - 0.06 * T
        else:
            ratio = 1
    elif fuel == '경유':
        if pollutant == 'CO':
            ratio = 1.9 - 0.03 * T
        elif pollutant == 'NOx':
            ratio = 1.3 - 0.013 * T
        elif pollutant == 'VOC':
            ratio = 3.1 - 0.09 * T
            if T > 29:
                ratio = max(ratio, 0.5)
        elif pollutant in ['PM10', 'PM25']:
            ratio = 3.1 - 0.1 * T
            if T > 26:
                ratio = max(ratio, 0.5)
        else:
            ratio = 1
    elif fuel == 'LPG':
        if pollutant == 'CO':
            ratio = 3.66 - 0.09 * T
        elif pollutant == 'NOx':
            ratio = 0.98 - 0.006 * T
        elif pollutant == 'VOC':
            ratio = 2.24 - 0.06 * T
            if T > 29:
                ratio = max(ratio, 0.5)
        else:
            ratio = 1
    else:
        ratio = 1  # 기타 연료는 비율 1로 처리

    # 비율이 0 이하일 경우 최소값 0.5로 설정
    if ratio <= 0:
        ratio = 0.5

    return ratio

def check_model_year_condition(condition_str, model_year):
    try:
        model_year = int(model_year)
        condition_str_original = condition_str  # 디버깅용
        
        # 문자열 변환 및 정규화
        condition_str = str(condition_str).strip().upper()
        
        # 유니코드 정규화 (모든 유형의 따옴표를 표준화)
        condition_str = unicodedata.normalize('NFKD', condition_str)
        
        # 모든 유형의 따옴표 제거
        # 따옴표 문자들의 더 포괄적인 리스트
        # 특수 문자 및 따옴표 제거
        condition_str = condition_str.replace("’", "").replace("‘", "")
        condition_str = condition_str.replace("“", "").replace("”", "")
        condition_str = condition_str.replace("″", "").replace("'", "").replace('"', '')
        
        # 'ALL' 처리
        if condition_str == 'ALL':
            return True

        # 'Y'를 실제 연식으로 대체
        condition_str = condition_str.replace('Y', str(model_year))

        # 연산자/피연산자 사이에 공백 추가
        condition_str = re.sub(r'([<>=!]=?)', r' \1 ', condition_str)
        condition_str = re.sub(r'\s+', ' ', condition_str).strip()

        # 'AND', 'OR' 처리
        condition_str = condition_str.replace('AND', ' and ').replace('OR', ' or ')

        # '='을 '=='로 교체 (>=, <= 등 제외)
        condition_str = re.sub(r'(?<![<>!])=(?!=)', '==', condition_str)
        
        # 중복 공백 제거
        
        # print(f"Original: '{condition_str_original}', Processed: '{condition_str}'")

        # 안전한 eval 사용
        allowed_names = {"__builtins__": None}
        result = eval(condition_str, allowed_names, {})
        return result

    except Exception as e:
        print(f"Error evaluating condition '{condition_str_original}': {e}")
        return False

def check_additional_conditions(condition_str, V, T):
    if pd.isna(condition_str) or condition_str.strip() == '':
        return True
    condition_str_original = condition_str  # 디버깅용
    
    # 문자열 변환 및 정규화
    condition_str = str(condition_str).strip().upper()
    
    # 유니코드 정규화
    condition_str = condition_str.replace('V', str(V)).replace('T', str(T))

    # 특수 문자 및 따옴표 제거
    condition_str = condition_str.replace("’", "").replace("‘", "")
    condition_str = condition_str.replace("“", "").replace("”", "")
    condition_str = condition_str.replace("″", "").replace("'", "").replace('"', '')

    condition_str = condition_str.replace('AND', ' and ').replace('OR', ' or ')
    condition_str = re.sub(r'(?<![<>!])=(?!=)', '==', condition_str)

    try:
        allowed_names = {"__builtins__": None}
        result = eval(condition_str, allowed_names, {})
        return result
    except Exception as e:
        print(f"Error evaluating additional condition '{condition_str_original}': {e}")
        return False

def calculate_emission_factor(EFi_formula, V):
    if pd.isna(EFi_formula) or str(EFi_formula).strip() == '':
        print("EFi_formula is empty or NaN.")
        return 0

    EFi_formula = str(EFi_formula)
    EFi_formula = EFi_formula.replace('×', '*').replace('–', '-').replace('−', '-')
    EFi_formula = EFi_formula.replace('V', str(V))
    EFi_formula = EFi_formula.replace('^', '**')
    EFi_formula = EFi_formula.replace('EXP', 'math.exp').replace('Exp', 'math.exp')

    try:
        EFi_value = eval(EFi_formula, {"__builtins__": None, 'math': math})
        return EFi_value
    except Exception as e:
        print(f"Error calculating emission factor with formula '{EFi_formula}': {e}")
        return 0

def calculate_deterioration_factor(DF_str, model_year):
    match = re.match(r'Tf(\d+)p_(\d+)W', str(DF_str))
    if match:
        a = int(match.group(1))
        b = int(match.group(2))
        dy = 2024 - int(model_year)
        # 기존 코드와 동일
        DF = min(max(1 + (dy - b) * (a / 100), 1), 1 + (a / 10))
        return DF
    else:
        return 1


# ===== [ 새로운 SOx 계산용 유틸 함수 추가 ] =====

def calculate_sox_Aj(sox_indicator, v):
    """
    황산화물배출계수 컬럼의 값(예: 'E16')과 속도 v를 받아서
    Aj 값을 계산해주는 함수.
    식은 질문에서 제시된 그대로 적용.
    """
    sox_indicator = str(sox_indicator).strip().upper()
    # v는 main.py에서 넘어온 V(속도)

    if sox_indicator == '0':
        return 0

    elif sox_indicator == 'E01':
        # E01 = 0.01090*(v**2) - 1.5100*v + 93.672
        return 0.01090*(v**2) - 1.5100*v + 93.672

    elif sox_indicator == 'E04':
        # E04 = 0.01870*(v**2) - 2.6974*v + 156.77
        return 0.01870*(v**2) - 2.6974*v + 156.77

    elif sox_indicator == 'E09':
        # E09 = 45
        return 45

    elif sox_indicator == 'E12':
        # E12 = 0.00790*(v**2) - 1.3123*v + 83.660
        return 0.00790*(v**2) - 1.3123*v + 83.660

    elif sox_indicator == 'E16':
        # E16 = if 5 <= v <= 60: 1919.0*(v**(-0.5396))
        #       elif 60 < v <= 120: 0.0447*(v**2) - 7.072*v + 478
        #       else: None
        if 5 <= v <= 60:
            return 1919.0*(v**(-0.5396))
        elif 60 < v <= 120:
            return 0.0447*(v**2) - 7.072*v + 478
        else:
            return None

    elif sox_indicator == 'E17':
        # E17 = ...
        if 5 <= v <= 60:
            return 1425.2*(v**(-0.7593))
        elif 60 < v <= 100:
            return 0.0082*(v**2) - 0.0430*v + 60.12
        else:
            return None

    elif sox_indicator == 'E18':
        # E18 = ...
        if 5 <= v <= 60:
            return 1068.4*(v**(-0.4905))
        elif 60 < v <= 100:
            return 0.0126*(v**2) - 0.6589*v + 141.2
        else:
            return None

    elif sox_indicator == 'E19':
        # E19 = ...
        if 5 <= v <= 60:
            return 1595.1*(v**(-0.4744))
        elif 60 < v <= 100:
            return 0.0382*(v**2) - 5.1630*v + 399.3
        else:
            return None

    elif sox_indicator == 'E20':
        # E20 = 25
        return 25

    elif sox_indicator == 'E21':
        # E21 = 0.02000*(v**2) - 2.0750*v + 77.10
        return 0.02000*(v**2) - 2.0750*v + 77.10

    else:
        return None


def get_sulfur_fuel(fuel):
    """
    Sfuel 값을 반환.
    - 경유: 0.00053
    - 휘발유: 0.00063
    - LPG: 0.0004
    - 나머지: 0
    """
    fuel = str(fuel).strip().upper()
    if fuel == '경유':
        return 0.00053
    elif fuel == '휘발유':
        return 0.00063
    elif fuel == 'LPG':
        return 0.0004
    else:
        return 0