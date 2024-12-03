import pandas as pd
import numpy as np
import os
import re
import math

def calculate_emissions(inputdata, VKT, V, T):
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

    # 3. 클래스와 배출계수 파일의 차종 매핑
    class_mapping = {
        '01_car': '승용차',
        '02_taxi': '승용차',  # 택시를 승용차로 매핑
        '03_van': '승합차',
        '04_bus': '버스',
        '05_LightTruck': '화물차',
        '06_HeavyTruck': '화물차',
        '07_SpecialVehicle': '특수차',
        '08_Motorcycle': '이륜차'
    }

    # 이륜차는 제외하므로 대상 클래스 리스트 업데이트
    target_classes = ['01_car', '02_taxi', '03_van', '04_bus', '05_LightTruck', '06_HeavyTruck', '07_SpecialVehicle']

    # 4. 결과를 저장할 딕셔너리 초기화
    pollutants = ['CO', 'NOx', 'PM25', 'PM10', 'VOC']
    results = {pollutant: [] for pollutant in pollutants}

    # 보조 데이터 사전 생성
    auxiliary_data = {
        'vehicle_type_ratio': vehicle_type_ratio,
        'vehicle_fuel_ratio': vehicle_fuel_ratio,
        'vehicle_age_ratio': vehicle_age_ratio,
        'emission_factors': emission_factors,
        'class_mapping': class_mapping,
        'target_classes': target_classes,
        'pollutants': pollutants,
        'VKT': VKT,
        'V': V,
        'T': T
    }

    # 멀티프로세싱 비활성화
    emissions_list = []
    for idx, row in inputdata.iterrows():
        emission_row = calculate_emission_for_time(row, auxiliary_data)
        emissions_list.append(emission_row)

    # 6. 결과를 정리하여 저장
    for emission_row in emissions_list:
        datetime = emission_row['DateTime']
        for pollutant in pollutants:
            pollutant_cols = [col for col in emission_row.keys() if col.endswith(f"_{pollutant}")]
            if not pollutant_cols:
                continue
            pollutant_data = {'DateTime': datetime}
            total_emission = 0  # 총 배출량 계산을 위한 변수 초기화
            for col in pollutant_cols:
                pollutant_data[col] = emission_row[col]
                total_emission += emission_row[col]  # 각 차종의 배출량을 합산
            # 총 배출량을 새로운 열에 추가
            pollutant_data[f'Total_{pollutant}'] = total_emission
            results[pollutant].append(pollutant_data)

    # 7. 결과를 데이터프레임으로 변환하고 엑셀 파일로 저장
    for pollutant in pollutants:
        if results[pollutant]:
            df = pd.DataFrame(results[pollutant])
            output_file = f"C:/dense_traffic_emi/output/emission_results_{pollutant}.xlsx"
            df.to_excel(output_file, index=False)

    print("배출량 계산이 완료되었습니다.")

def calculate_emission_for_time(row, auxiliary_data):
    emission_row = {'DateTime': row['DateTime']}
    vehicle_type_ratio = auxiliary_data['vehicle_type_ratio']
    vehicle_fuel_ratio = auxiliary_data['vehicle_fuel_ratio']
    vehicle_age_ratio = auxiliary_data['vehicle_age_ratio']
    emission_factors = auxiliary_data['emission_factors']
    class_mapping = auxiliary_data['class_mapping']
    target_classes = auxiliary_data['target_classes']
    pollutants = auxiliary_data['pollutants']
    VKT = auxiliary_data['VKT']
    V = auxiliary_data['V']
    T = auxiliary_data['T']

    # `소분류`와 `연료` 값 정규화
    emission_factors['소분류'] = emission_factors['소분류'].astype(str).str.strip().str.upper()
    emission_factors['연료'] = emission_factors['연료'].astype(str).str.strip().str.upper()

    for vehicle_class in target_classes:
        vehicle_count = row[vehicle_class]
        class_name = class_mapping[vehicle_class]

        # 5.1 소분류 비율 적용
        class_type_column = vehicle_class + '_type'
        if class_type_column not in vehicle_type_ratio.columns:
            print(f"{class_type_column} 열이 vehicle_type_ratio 데이터에 없습니다.")
            continue

        type_ratios = vehicle_type_ratio[[class_type_column, vehicle_class]].dropna()
        type_ratios.columns = ['소분류', '비율']
        type_ratios['비율'] = type_ratios['비율'] / type_ratios['비율'].sum()
        type_ratios['소분류'] = type_ratios['소분류'].astype(str).str.strip().str.upper()

        # 5.2 연료 비율 적용
        if vehicle_class not in vehicle_fuel_ratio.columns:
            print(f"{vehicle_class} 열이 vehicle_fuel_ratio 데이터에 없습니다.")
            continue

        fuel_ratios = vehicle_fuel_ratio[['fuel', vehicle_class]].dropna()
        fuel_ratios.columns = ['연료', '비율']
        fuel_ratios['비율'] = fuel_ratios['비율'] / fuel_ratios['비율'].sum()
        fuel_ratios['연료'] = fuel_ratios['연료'].astype(str).str.strip().str.upper()

        # 버스의 경우 연료 매핑
        if vehicle_class == '04_bus':
            fuel_ratios_list = []
            for idx2, subtype_row in type_ratios.iterrows():
                subtype = subtype_row['소분류']
                subtype_ratio = subtype_row['비율']
                if subtype == '시내버스':
                    fuel = 'CNG'
                elif subtype in ['시외버스', '전세버스', '고속버스']:
                    fuel = '경유'
                else:
                    fuel = '기타'
                fuel_ratios_list.append({'소분류': subtype, '연료': fuel, '비율': subtype_ratio})
            type_ratios = pd.DataFrame(fuel_ratios_list)
        else:
            type_ratios['key'] = 0
            fuel_ratios['key'] = 0
            type_ratios = pd.merge(type_ratios, fuel_ratios, on='key')
            type_ratios['비율'] = type_ratios['비율_x'] * type_ratios['비율_y']
            type_ratios = type_ratios[['소분류', '연료', '비율']]

        # 5.3 연식 비율 적용
        if vehicle_class not in vehicle_age_ratio.columns:
            print(f"{vehicle_class} 열이 vehicle_age_ratio 데이터에 없습니다.")
            continue

        age_ratios = vehicle_age_ratio[['model_year', vehicle_class]].dropna()
        age_ratios.columns = ['연식', '비율']
        age_ratios['비율'] = age_ratios['비율'] / age_ratios['비율'].sum()
        age_ratios_sorted = age_ratios.sort_values('연식')
        age_ratios_sorted['누적비율'] = age_ratios_sorted['비율'].cumsum()

        # 5.4 소분류-연료-연식 조합 생성
        type_ratios['key'] = 0
        age_ratios_sorted['key'] = 0
        combinations = pd.merge(type_ratios, age_ratios_sorted, on='key')
        combinations['조합비율'] = combinations['비율_x'] * combinations['비율_y']
        combinations = combinations[['소분류', '연료', '연식', '누적비율', '조합비율']]
        combinations['차종'] = class_name

        # 배출계수 및 열화계수 적용 전에 디버깅 정보 출력
        print(f"Processing class: {vehicle_class}, vehicle_count: {vehicle_count}")
        print(f"Number of combinations: {len(combinations)}")

        emission_factors_filtered = emission_factors[emission_factors['차종'] == class_name]

        # 디버깅 정보 출력
        print(f"Number of emission factors for {class_name}: {len(emission_factors_filtered)}")

        for idx_c, combo in combinations.iterrows():
            sub_type = combo['소분류']
            fuel = combo['연료']
            model_year = combo['연식']
            combo_ratio = combo['조합비율']
            vehicle_percentile = combo['누적비율']

            # 해당 조건의 배출계수 필터링
            ef_condition = (emission_factors_filtered['소분류'] == sub_type) & \
                           (emission_factors_filtered['연료'] == fuel) & \
                           (emission_factors_filtered['연식'].apply(lambda x: check_model_year_condition(x, model_year)))

            # 추가조건 처리
            ef_subset = emission_factors_filtered[ef_condition]
            ef_subset = ef_subset[ef_subset['추가조건'].apply(lambda x: check_additional_conditions(x, V, T))]

            if ef_subset.empty:
                print(f"No emission factors found for combination: {sub_type}, {fuel}, {model_year}")
                continue
            else:
                print(f"Emission factors found for combination: {sub_type}, {fuel}, {model_year}")

            for pollutant in pollutants:
                ef_pollutant = ef_subset[ef_subset['물질'] == pollutant]
                if ef_pollutant.empty:
                    continue

                # 배출계수 수식 계산
                EFi_formula = ef_pollutant['배출계수'].iloc[0]
                EFi_value = calculate_emission_factor(EFi_formula, V)

                # 열화계수 계산
                DF_str = ef_pollutant['열화계수'].iloc[0]
                DF_value = calculate_deterioration_factor(DF_str, model_year)

                # R 값 적용
                R_value = 0
                if fuel == '경유' and pollutant in ['CO', 'VOC', 'PM10', 'PM25']:
                    if vehicle_percentile <= 0.095:
                        R_installation_rate = 0.358  # 저감장치 부착률
                        if pollutant == 'CO':
                            R_value = 99.5 * R_installation_rate
                        elif pollutant == 'VOC':
                            R_value = 90 * R_installation_rate
                        elif pollutant in ['PM10', 'PM25']:
                            R_value = 83.6 * R_installation_rate

                # 배출량 계산
                Eij = VKT * (EFi_value / 1000) * DF_value * (1 - R_value / 100)
                emission = Eij * vehicle_count * combo_ratio

                # 결과 저장
                key = f"{class_name}_{pollutant}"
                if key not in emission_row:
                    emission_row[key] = emission
                else:
                    emission_row[key] += emission

    return emission_row

def check_model_year_condition(condition_str, model_year):
    try:
        model_year = int(model_year)
        condition_str_original = condition_str  # 디버깅을 위한 원본 조건 저장
        condition_str = str(condition_str).strip().upper()

        # 따옴표 등 특수 문자 제거
        condition_str = condition_str.replace("’", "").replace("'", "").replace('"', '')

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

        # 디버깅용 출력
        # print(f"Evaluating condition: '{condition_str_original}' -> '{condition_str}'")

        # 안전한 eval 사용
        allowed_names = {"__builtins__": None}
        result = eval(condition_str, allowed_names, {})
        return result

    except Exception as e:
        print(f"Error evaluating condition '{condition_str}': {e}")
        return False

def check_additional_conditions(condition_str, V, T):
    if pd.isna(condition_str) or condition_str.strip() == '':
        return True
    condition_str_original = condition_str  # 디버깅을 위한 원본 조건 저장
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
    if pd.isna(EFi_formula) or str(EFi_formula).strip() == '':
        print("EFi_formula is empty or NaN.")
        return 0
    EFi_formula = str(EFi_formula)  # 문자열로 변환
    # 특수 문자 대체
    EFi_formula = EFi_formula.replace('×', '*').replace('–', '-')
    EFi_formula = EFi_formula.replace('V', str(V))
    EFi_formula = EFi_formula.replace('^', '**')
    EFi_formula = EFi_formula.replace('EXP', 'math.exp')  # 'EXP'를 'math.exp'로 대체
    EFi_formula = EFi_formula.replace('Exp', 'math.exp')  # 'Exp'를 'math.exp'로 대체
    try:
        # 안전한 eval 사용
        EFi_value = eval(EFi_formula, {"__builtins__": None, 'math': math})
        return EFi_value
    except Exception as e:
        print(f"Error calculating emission factor with formula '{EFi_formula}': {e}")
        return 0

def calculate_deterioration_factor(DF_str, model_year):
    match = re.match(r'Tf(\d+)p_(\d+)W', DF_str)
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
    inputdata = pd.read_excel('C:/dense_traffic_emi/input/OJ_TS_summer.xlsx')

    # VKT, V, T 값 설정
    VKT = 0.79  # 주행거리 (km)
    V = 48  # 속도 (km/h)
    T = 31.1  # 온도 (°C)

    calculate_emissions(inputdata, VKT, V, T)
