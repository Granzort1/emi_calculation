import pandas as pd
import numpy as np
import os
import re
import math

def calculate_cold_start_emissions(inputdata, VKT, V, T):
    # 1. 필요한 파일 경로 설정
    factor_dir = r"C:\dense_traffic_emi\factor"
    vehicle_type_ratio_file = os.path.join(factor_dir, "vehicle_type_ratio_coefficient_v1.xlsx")
    vehicle_fuel_ratio_file = os.path.join(factor_dir, "vehicle_fuel_ratio_v1.xlsx")
    vehicle_age_ratio_file = os.path.join(factor_dir, "vehicle_age_ratio_v1.xlsx")
    emission_factor_file = os.path.join(factor_dir, "EFi_DF_Factor_ver7.xlsx")

    # 2. 입력 데이터 및 보조 데이터 로드
    vehicle_type_ratio = pd.read_excel(vehicle_type_ratio_file)
    vehicle_fuel_ratio = pd.read_excel(vehicle_fuel_ratio_file)
    vehicle_age_ratio = pd.read_excel(vehicle_age_ratio_file)
    emission_factors = pd.read_excel(emission_factor_file)

    # 3. 엔진미가열 배출 적용 대상 차종 정의
    target_classes = ['01_car', '03_van']  # 택시 제외

    # 4. 결과를 저장할 데이터프레임 초기화
    pollutants = ['CO', 'NOx', 'VOC']
    cold_emissions = pd.DataFrame()

    # 5. 필요한 상수 설정
    l_trip = 12.4  # 평균 1회 주행거리 (km)

    # 엔진미가열 상태 주행거리 분율 계산
    beta = 0.647 - 0.025 * l_trip - (0.00974 - 0.000385 * l_trip) * T

    # 배출계수 매핑 정의 (EFi_DF_Factor_ver7.xlsx 파일의 차종 이름과 매핑)
    ef_class_mapping = {
        '01_car': '승용차',
        '03_van': '승합차'
    }

    # 6. 입력 데이터 행별로 배출량 계산
    for idx, row in inputdata.iterrows():
        DateTime = row['DateTime']
        emission_row = {'DateTime': DateTime}

        for vehicle_class in target_classes:
            vehicle_count = row[vehicle_class]
            if vehicle_count == 0:
                continue  # 차량 수가 0이면 계산 생략

            # 배출계수 파일에서 사용할 차종 이름 가져오기
            ef_class_name = ef_class_mapping[vehicle_class]

            # 해당 차종의 배출계수 필터링
            ef_subset = emission_factors[emission_factors['차종'] == ef_class_name]

            # 오염물질별로 배출량 합산을 위한 딕셔너리 초기화
            total_emissions_by_pollutant = {pollutant: 0 for pollutant in pollutants}

            # 연료별로 계산하여 차종별로 합산
            if vehicle_class not in vehicle_fuel_ratio.columns:
                print(f"{vehicle_class} 열이 vehicle_fuel_ratio 데이터에 없습니다.")
                continue

            fuel_ratios = vehicle_fuel_ratio[['fuel', vehicle_class]].dropna()
            fuel_ratios.columns = ['연료', '비율']
            fuel_ratios['비율'] = fuel_ratios['비율'] / fuel_ratios['비율'].sum()
            fuel_ratios['연료'] = fuel_ratios['연료'].astype(str).str.strip().str.upper()

            if vehicle_class not in vehicle_age_ratio.columns:
                print(f"{vehicle_class} 열이 vehicle_age_ratio 데이터에 없습니다.")
                continue

            age_ratios = vehicle_age_ratio[['model_year', vehicle_class]].dropna()
            age_ratios.columns = ['연식', '비율']
            age_ratios['비율'] = age_ratios['비율'] / age_ratios['비율'].sum()

            # 연료 및 연식별로 배출량 계산하여 합산
            for idx_f, fuel_row in fuel_ratios.iterrows():
                fuel = fuel_row['연료']
                fuel_ratio = fuel_row['비율']

                # 연료에 해당하는 배출계수 필터링
                ef_fuel_subset = ef_subset[ef_subset['연료'] == fuel]

                if ef_fuel_subset.empty:
                    continue

                for idx_a, age_row in age_ratios.iterrows():
                    model_year = age_row['연식']
                    age_ratio = age_row['비율']

                    # 배출계수 조건에 맞는 데이터 필터링
                    ef_age_subset = ef_fuel_subset[ef_fuel_subset['연식'].apply(lambda x: check_model_year_condition(x, model_year))]

                    if ef_age_subset.empty:
                        continue

                    # 추가조건 처리
                    ef_age_subset = ef_age_subset[ef_age_subset['추가조건'].apply(lambda x: check_additional_conditions(x, V, T))]

                    if ef_age_subset.empty:
                        continue

                    # 오염물질별로 배출계수 추출 및 배출량 계산
                    for pollutant in pollutants:
                        ef_pollutant = ef_age_subset[ef_age_subset['물질'] == pollutant]
                        if ef_pollutant.empty:
                            continue

                        # 배출계수 수식 계산
                        EFi_formula = ef_pollutant['배출계수'].iloc[0]
                        EFi_value = calculate_emission_factor(EFi_formula, V)

                        # 열화계수 계산
                        DF_str = ef_pollutant['열화계수'].iloc[0]
                        DF_value = calculate_deterioration_factor(DF_str, model_year)

                        # 엔진가열 배출계수 계산
                        eHOT = EFi_value / 1000 * DF_value  # 단위: g/km

                        # 엔진미가열 대비 엔진가열 배출 비율 계산
                        e_cold_ratio = calculate_cold_hot_ratio(fuel, pollutant, T)

                        if e_cold_ratio is None:
                            continue

                        # 음수 방지 로직 추가
                        delta_ratio = e_cold_ratio - 1
                        if delta_ratio < 0:
                            delta_ratio = 0  # 음수일 경우 0으로 설정

                        # 엔진미가열 배출량 계산
                        E_cold = beta * vehicle_count * VKT * eHOT * delta_ratio * fuel_ratio * age_ratio

                        # 오염물질별로 배출량 합산
                        total_emissions_by_pollutant[pollutant] += E_cold

            # 차종별 오염물질 배출량 저장 (연료 종류 합산)
            for pollutant in pollutants:
                key = f"{vehicle_class}_{pollutant}"
                emission_row[key] = total_emissions_by_pollutant[pollutant]

        # 각 행의 배출량을 데이터프레임에 추가
        cold_emissions = cold_emissions._append(emission_row, ignore_index=True)

    # 전체 물질별 총합 계산하여 새로운 열에 저장
    if not cold_emissions.empty:
        for pollutant in pollutants:
            pollutant_columns = [f"{vc}_{pollutant}" for vc in target_classes]
            existing_columns = [col for col in pollutant_columns if col in cold_emissions.columns]
            if existing_columns:
                cold_emissions[f"Total_{pollutant}"] = cold_emissions[existing_columns].sum(axis=1)

    # 결과 저장
    output_file = 'C:/dense_traffic_emi/output/cold_start_emissions.xlsx'
    cold_emissions.to_excel(output_file, index=False)
    print(f"엔진미가열 배출량이 '{output_file}'에 저장되었습니다.")

def calculate_cold_hot_ratio(fuel, pollutant, T):
    # 엔진미가열 대비 엔진가열 배출 비율 계산 함수
    fuel = fuel.upper()
    T = float(T)
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
            # VOC는 Ta > 29℃일 때 최소값 0.5 적용
            if T > 29:
                ratio = max(ratio, 0.5)
        elif pollutant == 'PM':
            ratio = 3.1 - 0.1 * T
            # PM은 Ta > 26℃일 때 최소값 0.5 적용
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
            # VOC는 Ta > 29℃일 때 최소값 0.5 적용
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
    # 연식 조건 확인 함수
    try:
        model_year = int(model_year)
        condition_str = str(condition_str).strip().upper()

        # 'ALL' 처리
        if condition_str == 'ALL':
            return True

        # 'Y'를 실제 연식으로 대체
        condition_str = condition_str.replace('Y', str(model_year))

        # 연산자와 피연산자 사이에 공백 추가
        condition_str = re.sub(r'([<>=!]=?)', r' \1 ', condition_str)
        condition_str = re.sub(r'\s+', ' ', condition_str).strip()

        # 'AND', 'OR' 처리
        condition_str = condition_str.replace('AND', ' and ').replace('OR', ' or ')

        # '='을 '=='로 대체 (단, '>=' 또는 '<=' 등의 연산자는 제외)
        condition_str = re.sub(r'(?<![<>!])=(?!=)', '==', condition_str)

        # 안전한 eval 사용
        allowed_names = {"__builtins__": None}
        result = eval(condition_str, allowed_names, {})
        return result

    except Exception as e:
        print(f"Error evaluating condition '{condition_str}': {e}")
        return False

def check_additional_conditions(condition_str, V, T):
    # 추가 조건 확인 함수
    if pd.isna(condition_str) or condition_str.strip() == '':
        return True
    condition_str = condition_str.strip().upper()
    condition_str = condition_str.replace('V', str(V)).replace('T', str(T))
    condition_str = condition_str.replace('AND', ' and ').replace('OR', ' or ')
    condition_str = re.sub(r'(?<![<>!])=(?!=)', '==', condition_str)
    try:
        allowed_names = {"__builtins__": None}
        result = eval(condition_str, allowed_names, {})
        return result
    except Exception as e:
        print(f"Error evaluating additional condition '{condition_str}': {e}")
        return False

def calculate_emission_factor(EFi_formula, V):
    # 배출계수 수식 계산 함수
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
    # 열화계수 계산 함수
    match = re.match(r'Tf(\d+)p_(\d+)W', str(DF_str))
    if match:
        a = int(match.group(1))
        b = int(match.group(2))
        dy = 2024 - int(model_year)  # 현재 연도를 2024로 수정
        DF = min(max(1 + (dy - b) * (a / 100), 1), 1 + (a / 10))
        return DF
    else:
        return 1

#########################아래에서 계산##################

if __name__ == '__main__':
    # 입력 데이터 로드
    inputdata = pd.read_excel('C:/dense_traffic_emi/input/OJ_TS_summer.xlsx')

    # VKT, V, T 값 설정
    VKT = 0.79  # 주행거리 (km)
    V = 48  # 속도 (km/h)
    T = 31.1  # 온도 (°C)

    # 엔진미가열 배출량 계산 함수 호출
    calculate_cold_start_emissions(inputdata, VKT, V, T)
