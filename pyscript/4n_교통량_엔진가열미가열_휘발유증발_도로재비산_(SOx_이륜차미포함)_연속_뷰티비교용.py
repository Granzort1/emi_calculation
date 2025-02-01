import pandas as pd
import numpy as np
import os
import re
import math
import multiprocessing

def calculate_emissions(inputdata, VKT, V, T, ta_min, ta_rise, P_4N, output_dir, sL=0.06):

    factor_dir = r"C:\emi_calculation\factor"  # 1. 필요한 파일 경로 설정
    vehicle_type_ratio_file = os.path.join(factor_dir, "vehicle_type_ratio_coefficient_v1.xlsx")
    vehicle_fuel_ratio_file = os.path.join(factor_dir, "vehicle_fuel_ratio_v1.xlsx")
    vehicle_age_ratio_file = os.path.join(factor_dir, "vehicle_age_ratio_v1.xlsx")
    emission_factor_file = os.path.join(factor_dir, "EFi_DF_Factor_ver7.xlsx")

    # 2. 입력 데이터 및 보조 데이터 로드
    vehicle_type_ratio = pd.read_excel(vehicle_type_ratio_file)
    vehicle_fuel_ratio = pd.read_excel(vehicle_fuel_ratio_file)
    vehicle_age_ratio = pd.read_excel(vehicle_age_ratio_file)
    emission_factors = pd.read_excel(emission_factor_file)

    # 3. 입력 클래스 리스트 정의 (이륜차 포함)
    target_classes = ['01_car', '02_taxi', '03_van', '04_bus', '05_LightTruck', '06_HeavyTruck', '07_SpecialVehicle', '08_Motorcycle']

    # 4. 결과를 저장할 딕셔너리 초기화
    pollutants = ['CO', 'NOx', 'PM25', 'PM25_재비산', 'PM10', 'PM10_재비산', 'VOC', 'SOx', 'TSP']
    results = {pollutant: [] for pollutant in pollutants}

    # 배출계수를 찾지 못한 조합들을 저장할 리스트 초기화
    manager = multiprocessing.Manager()
    missing_emission_factors = manager.list()

    # 배출계수 매핑 정의
    ef_class_mapping = {
        '01_car': '승용차',
        '02_taxi': '승용차',
        '03_van': '승합차',
        '04_bus': '버스',
        '05_LightTruck': '화물차',
        '06_HeavyTruck': '화물차',
        '07_SpecialVehicle': '특수차',
        '08_Motorcycle': '이륜차'
    }

    # 소분류 매핑 정의
    subtype_mapping = {
        '01_car': ['경형', '소형', '중형', '대형'],
        '02_taxi': ['소형', '중형', '대형'],
        '03_van': ['경형', '소형', '중형', '대형', '특수형'],
        '04_bus': ['시내', '시외', '전세', '고속'],
        '05_LightTruck': ['경형', '소형'],
        '06_HeavyTruck': ['중형', '대형', '특수형', '덤프트럭', '콘크리트 믹서'],
        '07_SpecialVehicle': ['구난차', '견인차', '기타'],
        '08_Motorcycle': ['경형', '소형', '중형', '대형']
    }

    # 보조 데이터 사전 생성
    auxiliary_data = {
        'vehicle_type_ratio': vehicle_type_ratio,
        'vehicle_fuel_ratio': vehicle_fuel_ratio,
        'vehicle_age_ratio': vehicle_age_ratio,
        'emission_factors': emission_factors,
        'ef_class_mapping': ef_class_mapping,
        'subtype_mapping': subtype_mapping,
        'target_classes': target_classes,
        'pollutants': pollutants,
        'VKT': VKT,
        'V': V,
        'T': T,
        'ta_min': ta_min,
        'ta_rise': ta_rise,
        'P_4N': P_4N,  # 추가
        'sL': sL,  # 추가
        'missing_emission_factors': missing_emission_factors
    }

    # 멀티프로세싱 구현
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        args = [(row, auxiliary_data) for idx, row in inputdata.iterrows()]
        emissions_list = pool.starmap(calculate_emission_for_time, args)

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
            # 총 배출량을 새로운 열에 추가 - 여기를 수정
            pollutant_data[f'Total_{pollutant} (g/h)'] = total_emission  # (g/h) 추가
            results[pollutant].append(pollutant_data)

    # 7. 결과를 데이터프레임으로 변환하고 엑셀 파일로 저장
    for pollutant in pollutants:
        if results[pollutant]:
            df = pd.DataFrame(results[pollutant])

            if pollutant == 'TSP':
                # TSP는 별도 파일로 저장
                df = pd.DataFrame(results[pollutant])
                if f'Total_{pollutant}' in df.columns:
                    df = df.rename(columns={f'Total_{pollutant}': f'Total_{pollutant} (g/h)'})
                output_file = os.path.join(output_dir, f"resuspension_dust_{pollutant}_4n.xlsx")
                df.to_excel(output_file, index=False)
            elif pollutant in ['PM25', 'PM10']:
                # PM25와 PM10은 재비산 데이터와 함께 저장
                # PM25와 PM10은 재비산 데이터와 함께 저장
                base_pollutant = pollutant  # 기본 오염물질 (PM25 또는 PM10)
                resuspension_pollutant = f"{pollutant}_재비산"  # 재비산 오염물질

                # 기본 오염물질과 재비산 데이터 모두 있는 경우
                if results[base_pollutant] and results[resuspension_pollutant]:
                    base_df = pd.DataFrame(results[base_pollutant])
                    resuspension_df = pd.DataFrame(results[resuspension_pollutant])

                    # 재비산 데이터의 열 이름 변경
                    resuspension_df.columns = [
                        col.replace(resuspension_pollutant, f"{base_pollutant}_재비산")
                        if col != 'DateTime' and col != f'Total_{resuspension_pollutant} (g/h)'
                        else col
                        for col in resuspension_df.columns
                    ]

                    # DateTime 열을 기준으로 데이터프레임 병합
                    merged_df = pd.merge(base_df, resuspension_df.drop(f'Total_{resuspension_pollutant} (g/h)', axis=1),
                                         on='DateTime')

                    # 총 배출량 다시 계산 (기본 + 재비산)
                    total_cols = [col for col in merged_df.columns
                                  if col.endswith(f"_{base_pollutant}") or
                                  col.endswith(f"_{base_pollutant}_재비산")]
                    total_cols = [col for col in total_cols
                                  if not col.startswith('Total_')]

                    # 총합 계산 및 Total 열 추가 (기존 Total 열 제거 후)
                    if f'Total_{base_pollutant} (g/h)' in merged_df.columns:
                        merged_df = merged_df.drop(f'Total_{base_pollutant} (g/h)', axis=1)
                    merged_df[f'Total_{base_pollutant} (g/h)'] = merged_df[total_cols].sum(axis=1)

                    # 열 순서 재배열
                    cols = ['DateTime']  # DateTime 열을 첫 번째로

                    # 각 차량 클래스별로 일반 및 재비산 열을 쌍으로 정렬
                    vehicle_classes = ['01_car', '02_taxi', '03_van', '04_bus', '05_LightTruck',
                                       '06_HeavyTruck', '07_SpecialVehicle', '08_Motorcycle']

                    for vehicle_class in vehicle_classes:
                        # 일반 배출 열
                        regular_col = f"{vehicle_class}_{base_pollutant}"
                        if regular_col in merged_df.columns:
                            cols.append(regular_col)

                        # 재비산 배출 열
                        resuspension_col = f"{vehicle_class}_{base_pollutant}_재비산"
                        if resuspension_col in merged_df.columns:
                            cols.append(resuspension_col)

                    # Total 열을 마지막으로
                    cols.append(f'Total_{base_pollutant} (g/h)')


                    # 열 순서 적용
                    merged_df = merged_df[cols]

                    # 결과 저장
                    output_file = os.path.join(output_dir, f"emission_results_{base_pollutant}_4n.xlsx")
                    merged_df.to_excel(output_file, index=False)
            else:
                # 나머지 물질들 (CO, NOx, VOC, SOx) 저장
                # 열 순서 재배열
                cols = ['DateTime']
                vehicle_classes = ['01_car', '02_taxi', '03_van', '04_bus', '05_LightTruck',
                                   '06_HeavyTruck', '07_SpecialVehicle', '08_Motorcycle']

                for vehicle_class in vehicle_classes:
                    col = f"{vehicle_class}_{pollutant}"
                    if col in df.columns:
                        cols.append(col)

                # Total 열 계산 및 추가
                emission_cols = [col for col in df.columns if col.endswith(f"_{pollutant}")]
                emission_cols = [col for col in emission_cols if not col.startswith('Total_')]

                if f'Total_{pollutant}' in df.columns:
                    df = df.drop(f'Total_{pollutant}', axis=1)
                df[f'Total_{pollutant} (g/h)'] = df[emission_cols].sum(axis=1)

                cols.append(f'Total_{pollutant} (g/h)')
                df = df[cols]

                # 결과 저장
                output_file = os.path.join(output_dir, f"emission_results_{pollutant}_4n.xlsx")
                df.to_excel(output_file, index=False)



    # 배출계수를 찾지 못한 조합들을 엑셀 파일로 저장
    if missing_emission_factors:
        missing_df = pd.DataFrame(list(missing_emission_factors))
        missing_output_file = os.path.join(output_dir, "missing_emission_factors.xlsx")
        missing_df.to_excel(missing_output_file, index=False)
        print(f"배출계수를 찾지 못한 조합들이 '{missing_output_file}' 파일에 저장되었습니다.")

    print("배출량 계산이 완료되었습니다.")

def calculate_emission_for_time(row, auxiliary_data):
    #region[계수설정]
    emission_row = {'DateTime': row['DateTime']}
    vehicle_type_ratio = auxiliary_data['vehicle_type_ratio']
    vehicle_fuel_ratio = auxiliary_data['vehicle_fuel_ratio']
    vehicle_age_ratio = auxiliary_data['vehicle_age_ratio']
    emission_factors = auxiliary_data['emission_factors']
    ef_class_mapping = auxiliary_data['ef_class_mapping']
    subtype_mapping = auxiliary_data['subtype_mapping']
    target_classes = auxiliary_data['target_classes']
    pollutants = auxiliary_data['pollutants']
    VKT = auxiliary_data['VKT']
    V = auxiliary_data['V']
    T = auxiliary_data['T']
    P_4N = auxiliary_data.get('P_4N', 0)    # 추가
    sL = auxiliary_data.get('sL', 0.06)     # 추가
    ta_min = auxiliary_data['ta_min']
    ta_rise = auxiliary_data['ta_rise']
    missing_emission_factors = auxiliary_data['missing_emission_factors']

    # `소분류`와 `연료` 값 정규화
    emission_factors['소분류'] = emission_factors['소분류'].astype(str).str.strip().str.upper()
    emission_factors['연료'] = emission_factors['연료'].astype(str).str.strip().str.upper()

    # 엔진미가열 상태 주행거리 분율 계산을 위한 상수 설정
    l_trip = 12.4  # 평균 1회 주행거리 (km)
    beta = 0.647 - 0.025 * l_trip - (0.00974 - 0.000385 * l_trip) * T

    # 연료별 황 함량 (g/L) 설정 (예시 값, 실제 값으로 대체 필요)
    sulfur_content = {
        '경유': 0.00001,     # 10 ppm
        '휘발유': 0.00001,   # 10 ppm
        'LPG': 0.00005,      # 50 ppm
        'CNG': 0.0,          # 황 함량 없음
        '기타': 0.0
    }

    # 연료별 연비 (km/L) 설정 (예시 값, 실제 값으로 대체 필요)
    fuel_economy = {
        '경유': 12.0,
        '휘발유': 10.0,
        'LPG': 8.0,
        'CNG': 5.0,
        '기타': 7.0
    }
    daily_vkt = {
        '01_car': 28,     # 승용차
        '02_taxi': 28,   # 택시
        '03_van': 30,     # 승합차
        '04_bus': 30,    # 버스
        '05_LightTruck': 54,  # 화물차
        '06_HeavyTruck': 54,
        '07_SpecialVehicle': 37,
        '08_Motorcycle': 21.5  # 이륜차
    }
    RVP = 54 #연평균기장 단일값

    e_d = 9.1 * math.exp(0.0158 * (RVP - 61.2) + 0.0574 * (ta_min - 22.5) + 0.0614 * (ta_rise - 11.7))
    e_R_HOT = 0.136 * math.exp(-5.967 + 0.04259 * RVP + 0.1773 * T)
    e_R_WARM = 0.136 * math.exp(-5.967 + 0.04259 * RVP + 0.1773 * T)



    # endregion

    for vehicle_class in target_classes:
    #region[분류관련코드]

        vehicle_count = row[vehicle_class]
        if vehicle_count == 0:
            continue  # 차량 수가 0이면 계산 생략

        # 배출계수 파일에서 사용할 차종 이름 및 소분류 목록 가져오기
        ef_class_name = ef_class_mapping[vehicle_class]
        subtypes = subtype_mapping[vehicle_class]

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
        if vehicle_class == '02_taxi':
            # 택시의 연료 비율은 LPG 100%
            fuel_ratios = pd.DataFrame({'연료': ['LPG'], '비율': [1.0]})
        elif vehicle_class not in vehicle_fuel_ratio.columns:
            print(f"{vehicle_class} 열이 vehicle_fuel_ratio 데이터에 없습니다.")
            continue
        else:
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
                if subtype == '시내':
                    fuel = 'CNG'
                elif subtype in ['시외', '전세', '고속']:
                    fuel = '경유'
                else:
                    fuel = '기타'
                fuel_ratios_list.append({'소분류': subtype, '연료': fuel, '비율': subtype_ratio})
            type_fuel_ratios = pd.DataFrame(fuel_ratios_list)
        else:
            # 소분류와 연료 비율을 병합
            type_ratios['key'] = 0
            fuel_ratios['key'] = 0
            type_fuel_ratios = pd.merge(type_ratios, fuel_ratios, on='key')
            type_fuel_ratios['비율'] = type_fuel_ratios['비율_x'] * type_fuel_ratios['비율_y']
            type_fuel_ratios = type_fuel_ratios[['소분류', '연료', '비율']]

        # 5.3 연식 비율 적용
        if vehicle_class not in vehicle_age_ratio.columns:
            print(f"{vehicle_class} 열이 vehicle_age_ratio 데이터에 없습니다.")
            continue

        age_ratios = vehicle_age_ratio[['model_year', vehicle_class]].dropna()
        age_ratios.columns = ['연식', '비율']
        age_ratios['key'] = 0

        # 5.4 소분류-연료-연식 조합 생성
        type_fuel_ratios['key'] = 0
        combinations = pd.merge(type_fuel_ratios, age_ratios, on='key')

        # 경유 차량과 비경유 차량을 분리하여 누적비율 계산
        combinations_list = []

        for (subtype, fuel), group in combinations.groupby(['소분류', '연료']):
            group = group.copy()
            if fuel == '경유':
                # 경유 차량의 경우 연식별 누적비율 계산
                group['조합비율'] = group['비율_x'] * group['비율_y']
                group = group.sort_values('연식')
                total_ratio = group['조합비율'].sum()
                if total_ratio > 0:  # 0으로 나누기 방지
                    group['누적비율'] = (group['조합비율'].cumsum()) / total_ratio
            else:
                # 비경유 차량은 누적비율을 1로 설정
                group['조합비율'] = group['비율_x'] * group['비율_y']
                group['누적비율'] = 1.0

            combinations_list.append(group)

        # 모든 조합을 다시 하나의 데이터프레임으로 통합
        combinations = pd.concat(combinations_list)
        combinations = combinations[['소분류', '연료', '연식', '누적비율', '조합비율']]
        combinations['차종'] = ef_class_name


        emission_factors_filtered = emission_factors[emission_factors['차종'] == ef_class_name]
    # endregion

        #여기 아래가 각 세부분류 내부에서 계산할 코드임
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
                # 배출계수를 찾지 못한 조합을 리스트에 추가
                missing_emission_factors.append({
                    'DateTime': row['DateTime'],
                    '입력클래스': vehicle_class,
                    '차종': ef_class_name,
                    '소분류': sub_type,
                    '연료': fuel,
                    '연식': model_year
                })
                # 이륜차 등의 경우 배출계수가 없을 때 연료 소비량을 이용한 계산
                if vehicle_class == '08_Motorcycle':
                    # 연료 소비량 계산 (L/km)
                    if fuel in fuel_economy:
                        fuel_consumption = 1 / fuel_economy[fuel]  # L/km
                    else:
                        fuel_consumption = 0

                    # 연료별 배출인자 (예시 값, 실제 값으로 대체 필요)
                    fuel_emission_factors = {
                        'CO': {'경유': 0, '휘발유': 0, 'LPG': 0, 'CNG': 0, '기타': 0},
                        'NOx': {'경유': 0, '휘발유': 0, 'LPG': 0, 'CNG': 0, '기타': 0},
                        'VOC': {'경유': 0, '휘발유': 0, 'LPG': 0, 'CNG': 0, '기타': 0},
                        'PM10': {'경유': 0, '휘발유': 0, 'LPG': 0, 'CNG': 0, '기타': 0},
                        'PM25': {'경유': 0, '휘발유': 0, 'LPG': 0, 'CNG': 0, '기타': 0}
                    }

                    for pollutant in pollutants:
                        if pollutant == 'SOx':
                            # SOx 배출량 계산
                            if fuel in sulfur_content:
                                sulfur = sulfur_content[fuel]  # g/L
                            else:
                                sulfur = 0
                            emission = vehicle_count * VKT * fuel_consumption * sulfur * 2  # SO2로 전환되므로 2 곱함
                        else:
                            # 배출인자가 없으면 0으로 처리
                            emission_factor = fuel_emission_factors.get(pollutant, {}).get(fuel, 0)
                            emission = vehicle_count * VKT * emission_factor * combo_ratio

                        # 결과 저장
                        key = f"{vehicle_class}_{pollutant}"
                        if key not in emission_row:
                            emission_row[key] = emission
                        else:
                            emission_row[key] += emission

                continue

            for pollutant in pollutants:
                if pollutant == 'SOx':
                    # SOx 배출량 계산
                    if fuel in sulfur_content:
                        sulfur = sulfur_content[fuel]  # g/L
                    else:
                        sulfur = 0
                    # 연료 소비량 계산 (L/km)
                    if fuel in fuel_economy:
                        fuel_consumption = 1 / fuel_economy[fuel]  # L/km
                    else:
                        fuel_consumption = 0
                    # 배출량 계산
                    Eij = VKT * fuel_consumption * sulfur * 2  # SO2로 전환되므로 2 곱함
                    emission = Eij * vehicle_count * combo_ratio
                    # 결과 저장
                    key = f"{vehicle_class}_{pollutant}"
                    if key not in emission_row:
                        emission_row[key] = emission
                    else:
                        emission_row[key] += emission
                    continue

                if pollutant in ['PM25_재비산', 'PM10_재비산', 'TSP']:
                    if pollutant == 'PM25_재비산':
                        ef_pollutant = ef_subset[ef_subset['물질'] == 'PM25']
                        k = 0.15
                    elif pollutant == 'PM10_재비산':
                        ef_pollutant = ef_subset[ef_subset['물질'] == 'PM10']
                        k = 0.62
                    elif pollutant == 'TSP':
                        ef_pollutant = ef_subset[ef_subset['물질'] == 'PM10']
                        k = 3.23
                    else:
                        pass

                    if ef_pollutant.empty:
                        continue

                    W_avg = ef_pollutant['평균차중'].iloc[0]

                    Efpm = k * (sL ** 0.91) * (W_avg ** 1.02)
                    E_PM = VKT * vehicle_count * combo_ratio * Efpm * (1 - P_4N)  #g/h단위 배출량
                    emission = E_PM
                else:
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

                    # 배출량 계산 (엔진미가열 배출량 포함)
                    Eij = (EFi_value * DF_value * (1 - R_value / 100)) * VKT * vehicle_count * combo_ratio
                    eHOT = EFi_value
                    # 엔진미가열 배출량 계산 (승용차와 승합차에만 적용)
                    if vehicle_class in ['01_car', '03_van']:
                        e_cold_ratio = calculate_cold_hot_ratio(fuel, pollutant, T)
                        delta_ratio = e_cold_ratio - 1

                        E_cold = (beta * vehicle_count * VKT * eHOT * delta_ratio * combo_ratio)
                        emission = Eij + E_cold #g/h
                    else:
                        # 일반 배출량 계산
                        emission = Eij  #g/h


                    if fuel == '휘발유':
                        R = vehicle_count * combo_ratio * VKT * (0.6 * e_R_HOT + 0.4*e_R_WARM)
                        E_EVA = (((vehicle_count * combo_ratio * e_d * VKT) / (daily_vkt.get(vehicle_class))) + R)
                        emission += E_EVA #g/h
                    else:
                        pass






                # 결과 저장
                key = f"{vehicle_class}_{pollutant}"
                if key not in emission_row:
                    emission_row[key] = emission
                else:
                    emission_row[key] += emission

    return emission_row

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
        condition_str_original = condition_str  # 디버깅을 위한 원본 조건 저장
        condition_str = str(condition_str).strip().upper()

        # 특수 문자 및 따옴표 제거
        condition_str = condition_str.replace("’", "").replace("‘", "")
        condition_str = condition_str.replace("“", "").replace("”", "")
        condition_str = condition_str.replace("″", "").replace("'", "").replace('"', '')

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
        print(f"Error evaluating condition '{condition_str_original}': {e}")
        return False

def check_additional_conditions(condition_str, V, T):
    if pd.isna(condition_str) or condition_str.strip() == '':
        return True
    condition_str_original = condition_str  # 디버깅을 위한 원본 조건 저장
    condition_str = condition_str.strip().upper()
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
        DF = min(max(1 + (dy - b) * (a / 100), 1), 1 + (a / 10))
        return DF
    else:
        return 1

#########################아래에서 계산##################

if __name__ == '__main__':
 #   입력 파일과 매개변수 매핑

    경기동로_동측_ViewT_VKT = 0.637
    경기동로_서측_ViewT_VKT = 0.413
    경기동로_남측_ViewT_VKT = 0.802
    경기동로_북측_ViewT_VKT = 1.267
    경기대로_ViewT_VKT     = 0.802
    동부대로_남측_ViewT_VKT = 0.64
    동부대로_북측_ViewT_VKT = 0.108



    input_files = {
        '경기동로_여름_동측도로_차량수.xlsx': {
            'VKT': 경기동로_동측_ViewT_VKT,
            'V': 64,
            'T': 31.1,
            'ta_min': 28.1,
            'ta_rise': 8.3
        },
        '경기동로_여름_서측도로_차량수.xlsx': {
            'VKT': 경기동로_서측_ViewT_VKT,
            'V': 64,
            'T': 31.1,
            'ta_min': 28.1,
            'ta_rise': 8.3
        },
        '경기동로_여름_북측도로_차량수.xlsx': {
            'VKT': 경기동로_북측_ViewT_VKT,
            'V': 64,
            'T': 31.1,
            'ta_min': 28.1,
            'ta_rise': 8.3
        },

        '동부대로_여름_남측도로_차량수.xlsx': {
            'VKT': 동부대로_남측_ViewT_VKT,
            'V': 64,
            'T': 31.1,
            'ta_min': 28.1,
            'ta_rise': 8.3
        },
        '경기대로_여름_도로_차량수.xlsx': {
            'VKT': 경기대로_ViewT_VKT,
            'V': 64,
            'T': 31.1,
            'ta_min': 28.1,
            'ta_rise': 8.3
        },
    }

    input_files_fall = {
        '경기동로_가을_동측도로_차량수.xlsx': {
            'VKT': 경기동로_동측_ViewT_VKT,
            'V': 64,
            'T': 18.4,
            'ta_min': 15.2,
            'ta_rise': 8
        },
        '경기동로_가을_서측도로_차량수.xlsx': {
            'VKT': 경기동로_서측_ViewT_VKT,
            'V': 64,
            'T': 18.4,
            'ta_min': 15.2,
            'ta_rise': 8
        },
        '경기동로_가을_북측도로_차량수.xlsx': {
            'VKT': 경기동로_북측_ViewT_VKT,
            'V': 64,
            'T': 18.4,
            'ta_min': 15.2,
            'ta_rise': 8
        },

        '동부대로_가을_남측도로_차량수.xlsx': {
            'VKT': 동부대로_남측_ViewT_VKT,
            'V': 64,
            'T': 18.4,
            'ta_min': 15.2,
            'ta_rise': 8
        },
        '경기대로_가을_도로_차량수.xlsx': {
            'VKT': 경기대로_ViewT_VKT,
            'V': 64,
            'T': 18.4,
            'ta_min': 15.2,
            'ta_rise': 8
        },
    }


    input_files_winter = {
        '경기동로_겨울_동측도로_차량수.xlsx': {
            'VKT': 경기동로_동측_ViewT_VKT,
            'V': 64,
            'T': 0.4,
            'ta_min': -4.3,
            'ta_rise': 8.5
        },
        '경기동로_겨울_서측도로_차량수.xlsx': {
            'VKT': 경기동로_서측_ViewT_VKT,
            'V': 64,
            'T': 0.4,
            'ta_min': -4.3,
            'ta_rise': 8.5
        },
        '경기동로_겨울_북측도로_차량수.xlsx': {
            'VKT': 경기동로_북측_ViewT_VKT,
            'V': 64,
            'T': 0.4,
            'ta_min': -4.3,
            'ta_rise': 8.5
        },

        '동부대로_겨울_남측도로_차량수.xlsx': {
            'VKT': 동부대로_남측_ViewT_VKT,
            'V': 64,
            'T': 0.4,
            'ta_min': -4.3,
            'ta_rise': 8.5
        },
        '경기대로_겨울_도로_차량수.xlsx': {
            'VKT': 경기대로_ViewT_VKT,
            'V': 64,
            'T': 0.4,
            'ta_min': -4.3,
            'ta_rise': 8.5
        },
    }



    # 입력 기본 경로
    input_base_dir = 'C:/emi_calculation/input'
    # 출력 기본 경로
    output_base_dir = 'C:/emi_calculation/output/ViewT'


    # 계산에 필요한 파라미터 설정
    # ta_min = 28.1       # 최저온도평균 (2일이상시)
    # ta_rise = 8.3       # 일교차평균 (2일이상시) 1일은 단일값
    # VKT = 0.77         # 주행거리 (km)
    # V = 48             # 속도 (km/h)
    # T = 31.1           # 온도 (°C)
    sL = 0.06
    P_4N = 0

    # 봄, 가을, 겨울 input 파일 집합
    input_files_seasons = [
        input_files,  # 여름
        input_files_fall,  # 가을
        input_files_winter  # 겨울
    ]

    # 시즌별로 처리
    for season_idx, input_files in enumerate(input_files_seasons):
        if season_idx == 0:
            print("=== 여름 데이터 처리 시작 ===")
        elif season_idx == 1:
            print("=== 가을 데이터 처리 시작 ===")
        elif season_idx == 2:
            print("=== 겨울 데이터 처리 시작 ===")

        # 각 파일별로 처리
        for input_file, params in input_files.items():
            # 전체 입력 파일 경로
            full_input_path = os.path.join(input_base_dir, input_file)


            # 출력 디렉토리 생성 (파일명에서 확장자 제거)
            output_dir_name = os.path.splitext(input_file)[0]
            output_dir = os.path.join(output_base_dir, f'{output_dir_name}_emi')

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # 입력 데이터 로드
            try:
                inputdata = pd.read_excel(full_input_path)
                print(f"처리 중: {input_file}")
                print(f"매개변수: VKT={params['VKT']}, V={params['V']}, T={params['T']}, ta_min={params['ta_min']}, ta_rise={params['ta_rise']}")

                # 배출량 계산 함수 호출
                calculate_emissions(inputdata,
                                 VKT=params['VKT'],
                                 V=params['V'],
                                 T=params['T'],
                                 ta_min=params['ta_min'],
                                 ta_rise=params['ta_rise'],
                                 P_4N=P_4N,
                                 sL=sL,
                                 output_dir=output_dir)

                print(f"완료: {input_file}\n")

            except Exception as e:
                print(f"오류 발생 ({input_file}): {str(e)}\n")

        # 시즌별 데이터 처리 완료 메시지
        if season_idx == 0:
            print("=== 여름 데이터 처리 완료 ===\n")
        elif season_idx == 1:
            print("=== 가을 데이터 처리 완료 ===\n")
        elif season_idx == 2:
            print("=== 겨울 데이터 처리 완료 ===\n")
