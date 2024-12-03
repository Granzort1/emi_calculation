import pandas as pd
import numpy as np
import os
import re

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
        '02_taxi': '택시',
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
    results = {pollutant: pd.DataFrame() for pollutant in pollutants}

    # 5. 각 시간별로 계산 시작
    for idx, row in inputdata.iterrows():
        datetime = row['DateTime']
        emission_row = {'DateTime': datetime}

        # 각 클래스별로 배출량 계산
        for vehicle_class in target_classes:
            vehicle_count = row[vehicle_class]
            class_name = class_mapping[vehicle_class]

            # 5.1 소분류 비율 적용
            # 해당 클래스의 소분류 비율 가져오기
            class_type_column = vehicle_class + '_type'  # 예: '01_car_type'
            if class_type_column not in vehicle_type_ratio.columns:
                print(f"{class_type_column} 열이 vehicle_type_ratio 데이터에 없습니다.")
                continue  # 다음 클래스 처리

            type_ratios = vehicle_type_ratio[[class_type_column, vehicle_class]].dropna()
            type_ratios.columns = ['소분류', '비율']
            type_ratios['비율'] = type_ratios['비율'] / type_ratios['비율'].sum()

            # 5.2 연료 비율 적용
            if vehicle_class not in vehicle_fuel_ratio.columns:
                print(f"{vehicle_class} 열이 vehicle_fuel_ratio 데이터에 없습니다.")
                continue  # 다음 클래스 처리

            fuel_ratios = vehicle_fuel_ratio[['fuel', vehicle_class]].dropna()
            fuel_ratios.columns = ['연료', '비율']
            fuel_ratios['비율'] = fuel_ratios['비율'] / fuel_ratios['비율'].sum()

            # 버스의 경우 연료 매핑
            if vehicle_class == '04_bus':
                # 버스의 소분류별로 연료를 지정
                fuel_ratios_list = []
                for idx2, subtype_row in type_ratios.iterrows():
                    subtype = subtype_row['소분류']
                    subtype_ratio = subtype_row['비율']
                    if subtype == '시내버스':
                        fuel = 'CNG'
                    elif subtype in ['시외버스', '전세버스', '고속버스']:
                        fuel = '경유'
                    else:
                        fuel = '기타'  # 기타 연료가 있다면 처리 필요
                    fuel_ratios_list.append({'소분류': subtype, '연료': fuel, '비율': subtype_ratio})
                type_ratios = pd.DataFrame(fuel_ratios_list)
            else:
                # 연료 비율 적용
                type_ratios['key'] = 0
                fuel_ratios['key'] = 0
                type_ratios = pd.merge(type_ratios, fuel_ratios, on='key')
                type_ratios['비율'] = type_ratios['비율_x'] * type_ratios['비율_y']
                type_ratios = type_ratios[['소분류', '연료', '비율']]

            # 5.3 연식 비율 적용
            if vehicle_class not in vehicle_age_ratio.columns:
                print(f"{vehicle_class} 열이 vehicle_age_ratio 데이터에 없습니다.")
                continue  # 다음 클래스 처리

            age_ratios = vehicle_age_ratio[['model_year', vehicle_class]].dropna()
            age_ratios.columns = ['연식', '비율']
            age_ratios['비율'] = age_ratios['비율'] / age_ratios['비율'].sum()
            age_ratios_sorted = age_ratios.sort_values('연식')
            age_ratios_sorted['누적비율'] = age_ratios_sorted['비율'].cumsum()

            # 5.4 소분류-연료-연식 조합 생성
            type_ratios['key'] = 0
            age_ratios['key'] = 0
            combinations = pd.merge(type_ratios, age_ratios, on='key')
            combinations['조합비율'] = combinations['비율_x'] * combinations['비율_y']
            combinations = combinations[['소분류', '연료', '연식', '조합비율']]

            # 5.5 배출계수 및 열화계수 적용
            for index, combo in combinations.iterrows():
                sub_type = combo['소분류']
                fuel = combo['연료']
                model_year = combo['연식']
                combo_ratio = combo['조합비율']

                # 해당 조건의 배출계수 필터링
                ef_condition = (emission_factors['차종'] == class_name) & \
                               (emission_factors['소분류'] == sub_type) & \
                               (emission_factors['연료'] == fuel) & \
                               (emission_factors['연식'].apply(lambda x: check_model_year_condition(x, model_year)))

                # 추가조건 처리 (속도 V, 온도 T)
                if '추가조건' in emission_factors.columns:
                    ef_condition &= emission_factors['추가조건'].apply(lambda x: check_additional_conditions(x, V, T))

                ef_subset = emission_factors[ef_condition]

                if ef_subset.empty:
                    continue  # 해당 조건에 맞는 배출계수가 없으면 다음으로

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
                        # 생산연도가 하위 9.5%에 해당하는지 확인
                        vehicle_percentile = age_ratios_sorted[age_ratios_sorted['연식'] == model_year]['누적비율'].values[0]
                        if vehicle_percentile <= 0.095:
                            R_installation_rate = 0.358  # 저감장치 부착률
                            if pollutant == 'CO':
                                R_value = 99.5 * R_installation_rate
                            elif pollutant == 'VOC':
                                R_value = 90 * R_installation_rate
                            elif pollutant in ['PM10', 'PM25']:
                                R_value = 83.6 * R_installation_rate
                    else:
                        R_value = 0

                    # 배출량 계산
                    Eij = VKT * (EFi_value / 1000) * DF_value * (1 - R_value / 100)
                    # 차량 대수 및 조합비율 적용
                    emission = Eij * vehicle_count * combo_ratio

                    # 결과 저장
                    key = f"{class_name}_{pollutant}"
                    if key not in emission_row:
                        emission_row[key] = emission
                    else:
                        emission_row[key] += emission

        # 시간별 결과를 결과 데이터프레임에 추가
        for pollutant in pollutants:
            pollutant_cols = [col for col in emission_row.keys() if col.endswith(f"_{pollutant}")]
            if not pollutant_cols:
                continue
            pollutant_data = {'DateTime': datetime}
            for col in pollutant_cols:
                pollutant_data[col] = emission_row[col]
            # 수정된 부분: pd.concat을 사용하여 데이터 추가
            results[pollutant] = pd.concat([results[pollutant], pd.DataFrame([pollutant_data])], ignore_index=True)

    # 6. 결과를 엑셀 파일로 저장
    for pollutant in pollutants:
        output_file = f"emission_results_{pollutant}.xlsx"
        results[pollutant].to_excel(output_file, index=False)

    print("배출량 계산이 완료되었습니다.")

# 추가 함수들 정의
def check_model_year_condition(condition_str, model_year):
    # 연식 조건 문자열을 파싱하여 해당 연식이 조건을 만족하는지 확인
    # 예: "01=Y", "11<=Y<=13", "04>Y"
    try:
        model_year = int(model_year)
        condition_str = condition_str.replace('Y', str(model_year))
        condition_str = condition_str.replace('=', '==')  # '='를 '=='로 변경
        condition_str = condition_str.replace('<=', '<=').replace('>=', '>=').replace('<', '<').replace('>', '>')
        return eval(condition_str)
    except:
        return False

def check_additional_conditions(condition_str, V, T):
    # 추가조건 문자열을 파싱하여 V, T를 사용하여 조건을 만족하는지 확인
    if pd.isna(condition_str) or condition_str.strip() == '':
        return True
    condition_str = condition_str.replace('V', str(V)).replace('T', str(T))
    condition_str = condition_str.replace('and', ' and ').replace('or', ' or ')
    condition_str = condition_str.replace('<=', '<=').replace('>=', '>=').replace('<', '<').replace('>', '>')
    try:
        return eval(condition_str)
    except:
        return False

def calculate_emission_factor(EFi_formula, V):
    # 배출계수 수식에서 V를 대입하여 계산
    EFi_formula = EFi_formula.replace('V', str(V))
    # 지수 연산 처리 (예: '^'를 '**'로 변경)
    EFi_formula = EFi_formula.replace('^', '**')
    # 수식 계산
    try:
        EFi_value = eval(EFi_formula)
        return EFi_value
    except:
        return 0

def calculate_deterioration_factor(DF_str, model_year):
    # 열화계수 문자열을 파싱하여 DF를 계산
    # DF_str 예시: "Tf5p_5W"
    # 여기서 a와 b 추출
    match = re.match(r'Tf(\d+)p_(\d+)W', DF_str)
    if match:
        a = int(match.group(1))
        b = int(match.group(2))
        dy = 2024 - int(model_year)  # 현재 연도를 2024로 수정
        DF = min(max(1 + (dy - b) * (a / 100), 1), 1 + (a / 10))
        return DF
    else:
        return 1  # 매칭되지 않으면 1로 설정

#########################아래에서 계산##################

inputdata = pd.read_excel('C:/dense_traffic_emi/input/OJ_TS_summer.xlsx')

# VKT, V, T 값 설정
VKT = 0.79  # 예시 주행거리
V = 48  # 예시 속도 (km/h)
T = 20  # 예시 온도

calculate_emissions(inputdata, VKT, V, T)
