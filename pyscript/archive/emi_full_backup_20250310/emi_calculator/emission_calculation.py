import math
import multiprocessing
import pandas as pd
import os

# utilities.py의 함수들 임포트
# 상대경로 임포트를 절대경로로 수정
from .utilities import (
    calculate_cold_hot_ratio, 
    check_model_year_condition,
    check_additional_conditions, 
    calculate_emission_factor,
    calculate_deterioration_factor, 
    calculate_sox_Aj,
    get_sulfur_fuel
)


def calculate_emissions(inputdata, VKT, V, T, ta_min, ta_rise, P_4N, output_dir, sL=0.06):
    """
    기존 스크립트의 calculate_emissions() 함수.
    여기서 SOx 계산 로직을 새 규칙으로 수정함.
    """
    factor_dir = r"C:\emi_calculation\factor"  # 1. 필요한 파일 경로 설정
    vehicle_type_ratio_file = os.path.join(factor_dir, "vehicle_type_ratio_coefficient_v1.xlsx")
    vehicle_fuel_ratio_file = os.path.join(factor_dir, "vehicle_fuel_ratio_v1.xlsx")
    vehicle_age_ratio_file = os.path.join(factor_dir, "vehicle_age_ratio_v1.xlsx")
    emission_factor_file = os.path.join(factor_dir, "EFi_DF_Factor_ver8.xlsx")

    # 2. 보조 데이터 로드
    vehicle_type_ratio = pd.read_excel(vehicle_type_ratio_file)
    vehicle_fuel_ratio = pd.read_excel(vehicle_fuel_ratio_file)
    vehicle_age_ratio = pd.read_excel(vehicle_age_ratio_file)
    emission_factors = pd.read_excel(emission_factor_file)

    # 3. 입력 클래스 리스트 (이륜차 포함)
    target_classes = [
        '01_car', '02_taxi', '03_van', '04_bus',
        '05_LightTruck', '06_HeavyTruck', '07_SpecialVehicle', '08_Motorcycle'
    ]

    # 4. 결과를 저장할 딕셔너리
    pollutants = ['CO', 'NOx', 'PM25', 'PM25_재비산', 'PM10',
                  'PM10_재비산', 'VOC', 'SOx', 'TSP']
    results = {pollutant: [] for pollutant in pollutants}

    # 배출계수를 찾지 못한 조합들 저장할 리스트
    manager = multiprocessing.Manager()
    missing_emission_factors = manager.list()

    # 배출계수 매핑 정의
    ef_class_mapping = {
        '01_car': '승용차',
        '02_taxi': '택시',
        '03_van': '승합차',
        '04_bus': '버스',
        '05_LightTruck': '화물차',
        '06_HeavyTruck': '화물차',
        '07_SpecialVehicle': '특수차',
        '08_Motorcycle': '이륜차'
    }

    # 소분류 매핑
    subtype_mapping = {
        '01_car': ['경형', '소형', '중형', '대형'],
        '02_taxi': ['소형', '중형', '대형'],
        '03_van': ['경형', '소형', '중형', '대형', '특수형'],
        '04_bus': ['시내버스', '시외버스', '전세버스', '고속버스'],
        '05_LightTruck': ['경형', '소형'],
        '06_HeavyTruck': ['중형', '대형', '특수형', '덤프트럭', '콘크리트믹서'],
        '07_SpecialVehicle': ['구난차', '견인차', '기타'],
        '08_Motorcycle': ['']
    }

    # 보조 데이터 사전
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
        'P_4N': P_4N,
        'sL': sL,
        'missing_emission_factors': missing_emission_factors
    }

    # 멀티프로세싱
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        args = [(row, auxiliary_data) for idx, row in inputdata.iterrows()]
        emissions_list = pool.starmap(calculate_emission_for_time, args)

    # 6. 결과 정리 & 저장
    for emission_row in emissions_list:
        datetime = emission_row['DateTime']
        for pollutant in pollutants:
            pollutant_cols = [col for col in emission_row.keys() if col.endswith(f"_{pollutant}")]
            if not pollutant_cols:
                continue
            pollutant_data = {'DateTime': datetime}
            total_emission = 0
            for col in pollutant_cols:
                pollutant_data[col] = emission_row[col]
                total_emission += emission_row[col]
            pollutant_data[f'Total_{pollutant} (g/h)'] = total_emission
            results[pollutant].append(pollutant_data)

    # 7. DataFrame 변환 후 엑셀 저장
    for pollutant in pollutants:
        if results[pollutant]:
            df = pd.DataFrame(results[pollutant])

            if pollutant == 'TSP':
                # TSP 별도 파일
                df = pd.DataFrame(results[pollutant])
                if f'Total_{pollutant}' in df.columns:
                    df = df.rename(columns={f'Total_{pollutant}': f'Total_{pollutant} (g/h)'})
                output_file = os.path.join(output_dir, f"resuspension_dust_{pollutant}_4n.xlsx")
                df.to_excel(output_file, index=False)

            elif pollutant in ['PM25', 'PM10']:
                # PM25, PM10은 재비산 함께
                base_pollutant = pollutant
                resuspension_pollutant = f"{pollutant}_재비산"

                if results[base_pollutant] and results[resuspension_pollutant]:
                    base_df = pd.DataFrame(results[base_pollutant])
                    resuspension_df = pd.DataFrame(results[resuspension_pollutant])

                    # 재비산 열 이름 변경
                    resuspension_df.columns = [
                        col.replace(resuspension_pollutant, f"{base_pollutant}_재비산")
                        if col not in ['DateTime', f'Total_{resuspension_pollutant} (g/h)']
                        else col
                        for col in resuspension_df.columns
                    ]

                    merged_df = pd.merge(
                        base_df,
                        resuspension_df.drop(f'Total_{resuspension_pollutant} (g/h)', axis=1),
                        on='DateTime'
                    )

                    # 총 배출량 다시 계산
                    total_cols = [
                        col for col in merged_df.columns
                        if (col.endswith(f"_{base_pollutant}") or
                            col.endswith(f"_{base_pollutant}_재비산"))
                    ]
                    total_cols = [col for col in total_cols if not col.startswith('Total_')]

                    if f'Total_{base_pollutant} (g/h)' in merged_df.columns:
                        merged_df = merged_df.drop(f'Total_{base_pollutant} (g/h)', axis=1)
                    merged_df[f'Total_{base_pollutant} (g/h)'] = merged_df[total_cols].sum(axis=1)

                    # 열 순서 재배열
                    cols = ['DateTime']
                    vehicle_classes = [
                        '01_car', '02_taxi', '03_van', '04_bus',
                        '05_LightTruck', '06_HeavyTruck', '07_SpecialVehicle', '08_Motorcycle'
                    ]

                    for vehicle_class in vehicle_classes:
                        regular_col = f"{vehicle_class}_{base_pollutant}"
                        if regular_col in merged_df.columns:
                            cols.append(regular_col)
                        resuspension_col = f"{vehicle_class}_{base_pollutant}_재비산"
                        if resuspension_col in merged_df.columns:
                            cols.append(resuspension_col)

                    cols.append(f'Total_{base_pollutant} (g/h)')
                    merged_df = merged_df[cols]

                    output_file = os.path.join(output_dir, f"emission_results_{base_pollutant}_4n.xlsx")
                    merged_df.to_excel(output_file, index=False)

            else:
                # CO, NOx, VOC, SOx 등
                cols = ['DateTime']
                vehicle_classes = [
                    '01_car', '02_taxi', '03_van', '04_bus',
                    '05_LightTruck', '06_HeavyTruck', '07_SpecialVehicle', '08_Motorcycle'
                ]

                for vehicle_class in vehicle_classes:
                    col = f"{vehicle_class}_{pollutant}"
                    if col in df.columns:
                        cols.append(col)

                # Total 열 계산
                emission_cols = [col for col in df.columns if col.endswith(f"_{pollutant}")]
                emission_cols = [col for col in emission_cols if not col.startswith('Total_')]

                if f'Total_{pollutant}' in df.columns:
                    df = df.drop(f'Total_{pollutant}', axis=1)
                df[f'Total_{pollutant} (g/h)'] = df[emission_cols].sum(axis=1)

                cols.append(f'Total_{pollutant} (g/h)')
                df = df[cols]

                output_file = os.path.join(output_dir, f"emission_results_{pollutant}_4n.xlsx")
                df.to_excel(output_file, index=False)

    # 누락된 배출계수 조합 저장
    if missing_emission_factors:
        missing_df = pd.DataFrame(list(missing_emission_factors))
        missing_output_file = os.path.join(output_dir, "missing_emission_factors.xlsx")
        missing_df.to_excel(missing_output_file, index=False)
        print(f"배출계수를 찾지 못한 조합들이 '{missing_output_file}' 파일에 저장되었습니다.")

    print("배출량 계산이 완료되었습니다.")


def calculate_emission_for_time(row, auxiliary_data):
    """
    실제 각 row(시간)에 대해 배출량을 계산하는 함수.
    여기서 SOx 계산 부분만 새 규칙 적용.
    """

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
    P_4N = auxiliary_data.get('P_4N', 0)
    sL = auxiliary_data.get('sL', 0.06)
    ta_min = auxiliary_data['ta_min']
    ta_rise = auxiliary_data['ta_rise']
    missing_emission_factors = auxiliary_data['missing_emission_factors']

    # `소분류`, `연료` 정규화
    emission_factors['소분류'] = emission_factors['소분류'].astype(str).str.strip().str.upper()
    emission_factors['연료'] = emission_factors['연료'].astype(str).str.strip().str.upper()

    # 엔진미가열 상태 주행거리 분율 (원본)
    l_trip = 12.4
    beta = 0.647 - 0.025 * l_trip - (0.00974 - 0.000385 * l_trip) * T

    # (이전) 연료별 황함량 -> 이제 SOx 부분에서 사용 안 함.

    # 연료별 연비 (예시)
    fuel_economy = {
        '경유': 12.0,
        '휘발유': 10.0,
        'LPG': 8.0,
        'CNG': 5.0,
        '기타': 7.0
    }

    # 하루 주행거리 (예시)
    daily_vkt = {
        '01_car': 28,
        '02_taxi': 28,
        '03_van': 30,
        '04_bus': 30,
        '05_LightTruck': 54,
        '06_HeavyTruck': 54,
        '07_SpecialVehicle': 37,
        '08_Motorcycle': 21.5
    }

    RVP = 54
    e_d = 9.1 * math.exp(0.0158 * (RVP - 61.2) + 0.0574 * (ta_min - 22.5) + 0.0614 * (ta_rise - 11.7))
    e_R_HOT = 0.136 * math.exp(-5.967 + 0.04259 * RVP + 0.1773 * T)
    e_R_WARM = 0.136 * math.exp(-5.967 + 0.04259 * RVP + 0.1773 * T)

    for vehicle_class in target_classes:
        vehicle_count = row[vehicle_class]
        if vehicle_count == 0:
            continue  # 차량 수 0이면 패스

        ef_class_name = ef_class_mapping[vehicle_class]
        subtypes = subtype_mapping[vehicle_class]

        # 소분류 비율
        class_type_column = vehicle_class + '_type'
        if class_type_column not in vehicle_type_ratio.columns:
            print(f"{class_type_column} 열이 vehicle_type_ratio 데이터에 없습니다.")
            continue

        type_ratios = vehicle_type_ratio[[class_type_column, vehicle_class]].dropna()
        type_ratios.columns = ['소분류', '비율']
        type_ratios['비율'] = type_ratios['비율'] / type_ratios['비율'].sum()
        type_ratios['소분류'] = type_ratios['소분류'].astype(str).str.strip().str.upper()

        # 연료 비율
        if vehicle_class == '02_taxi':
            # 택시 = LPG 100%
            fuel_ratios = pd.DataFrame({'연료': ['LPG'], '비율': [1.0]})
        elif vehicle_class not in vehicle_fuel_ratio.columns:
            print(f"{vehicle_class} 열이 vehicle_fuel_ratio 데이터에 없습니다.")
            continue
        else:
            fuel_ratios = vehicle_fuel_ratio[['fuel', vehicle_class]].dropna()
            fuel_ratios.columns = ['연료', '비율']
            fuel_ratios['비율'] = fuel_ratios['비율'] / fuel_ratios['비율'].sum()
            fuel_ratios['연료'] = fuel_ratios['연료'].astype(str).str.strip().str.upper()

        # 버스(subtype)에 따른 연료 매핑
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
                fuel_ratios_list.append({
                    '소분류': subtype,
                    '연료': fuel,
                    '비율': subtype_ratio
                })
            type_fuel_ratios = pd.DataFrame(fuel_ratios_list)
        else:
            type_ratios['key'] = 0
            fuel_ratios['key'] = 0
            type_fuel_ratios = pd.merge(type_ratios, fuel_ratios, on='key')
            type_fuel_ratios['비율'] = type_fuel_ratios['비율_x'] * type_fuel_ratios['비율_y']
            type_fuel_ratios = type_fuel_ratios[['소분류', '연료', '비율']]

        # 연식 비율
        if vehicle_class not in vehicle_age_ratio.columns:
            print(f"{vehicle_class} 열이 vehicle_age_ratio 데이터에 없습니다.")
            continue

        age_ratios = vehicle_age_ratio[['model_year', vehicle_class]].dropna()
        age_ratios.columns = ['연식', '비율']
        age_ratios['key'] = 0

        # 소분류-연료-연식 조합
        type_fuel_ratios['key'] = 0
        combinations = pd.merge(type_fuel_ratios, age_ratios, on='key')

        # 경유/비경유 분리 후 누적비율 계산
        combinations_list = []
        for (subtype, fuel), group in combinations.groupby(['소분류', '연료']):
            group = group.copy()
            if fuel == '경유':
                group['조합비율'] = group['비율_x'] * group['비율_y']
                group = group.sort_values('연식')
                total_ratio = group['조합비율'].sum()
                if total_ratio > 0:
                    group['누적비율'] = group['조합비율'].cumsum() / total_ratio
            else:
                group['조합비율'] = group['비율_x'] * group['비율_y']
                group['누적비율'] = 1.0

            combinations_list.append(group)

        combinations = pd.concat(combinations_list)
        combinations = combinations[['소분류', '연료', '연식', '누적비율', '조합비율']]
        combinations['차종'] = ef_class_name

        emission_factors_filtered = emission_factors[emission_factors['차종'] == ef_class_name]

        for idx_c, combo in combinations.iterrows():
            sub_type = combo['소분류']
            fuel = combo['연료']
            model_year = combo['연식']
            combo_ratio = combo['조합비율']
            vehicle_percentile = combo['누적비율']

            # ef_condition = '소분류', '연료', '연식' 일치 + 추가조건
            ef_condition = (
                (emission_factors_filtered['소분류'] == sub_type) &
                (emission_factors_filtered['연료'] == fuel) &
                (emission_factors_filtered['연식'].apply(
                    lambda x: check_model_year_condition(x, model_year)
                ))
            )
            ef_subset = emission_factors_filtered[ef_condition]
            ef_subset = ef_subset[ef_subset['추가조건'].apply(lambda x: check_additional_conditions(x, V, T))]

            if ef_subset.empty:
                missing_emission_factors.append({
                    'DateTime': row['DateTime'],
                    '입력클래스': vehicle_class,
                    '차종': ef_class_name,
                    '소분류': sub_type,
                    '연료': fuel,
                    '연식': model_year
                })
                # 이륜차 특수처리 (원본 코드 유지)
                if vehicle_class == '08_Motorcycle':
                    if fuel in fuel_economy:
                        fuel_consumption = 1 / fuel_economy[fuel]
                    else:
                        fuel_consumption = 0

                    # 여기서도 SOx가 있으면 어떻게? -> 이미 ef_subset 없으니 0으로 처리
                    fuel_emission_factors = {
                        'CO': {'경유': 0, '휘발유': 0, 'LPG': 0, 'CNG': 0, '기타': 0},
                        'NOx': {'경유': 0, '휘발유': 0, 'LPG': 0, 'CNG': 0, '기타': 0},
                        'VOC': {'경유': 0, '휘발유': 0, 'LPG': 0, 'CNG': 0, '기타': 0},
                        'PM10': {'경유': 0, '휘발유': 0, 'LPG': 0, 'CNG': 0, '기타': 0},
                        'PM25': {'경유': 0, '휘발유': 0, 'LPG': 0, 'CNG': 0, '기타': 0}
                    }

                    # SOx나 기타 물질은 0 처리
                    for pollutant in pollutants:
                        if pollutant == 'SOx':
                            pass
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
                    # [새 규칙] EFi_DF_Factor_ver8.xlsx 안에 '황산화물배출계수' 열이 있다고 가정
                    # 예: ef_pollutant['황산화물배출계수'].iloc[0] => E01, E16 등
                    sox_indicator = ef_subset['황산화물배출계수'].iloc[0]
                    Aj = calculate_sox_Aj(sox_indicator, V)  # Aj
                    Sfuel = get_sulfur_fuel(fuel)           # Sfuel

                    if Aj is None:
                        # 속도 범위 벗어나면 0 처리
                        emission = 0
                    else:
                        # Esox = (2 * Aj * Sfuel/100) * VKT × 차량수 × 조합비율
                        emission = (2.0 * Aj * Sfuel / 100.0) * VKT * vehicle_count * combo_ratio

                    key = f"{vehicle_class}_{pollutant}"
                    emission_row[key] = emission_row.get(key, 0) + emission

                elif pollutant in ['PM25_재비산', 'PM10_재비산', 'TSP']:
                    # 재비산 계산 (원본)
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

                    # 나머지 배출물질(CO, NOx, VOC, PM10, PM25 등) (원본)
                    EFi_formula = ef_pollutant['배출계수'].iloc[0]
                    EFi_value = calculate_emission_factor(EFi_formula, V)

                    DF_str = ef_pollutant['열화계수'].iloc[0]
                    DF_value = calculate_deterioration_factor(DF_str, model_year)

                    R_value = 0
                    if fuel == '경유' and pollutant in ['CO', 'VOC', 'PM10', 'PM25']:
                        if vehicle_percentile <= 0.095:
                            R_installation_rate = 0.358
                            if pollutant == 'CO':
                                R_value = 99.5 * R_installation_rate
                            elif pollutant == 'VOC':
                                R_value = 90 * R_installation_rate
                            elif pollutant in ['PM10', 'PM25']:
                                R_value = 83.6 * R_installation_rate

                    Eij = (EFi_value * DF_value * (1 - R_value / 100)) * VKT * vehicle_count * combo_ratio
                    eHOT = EFi_value

                    # 엔진미가열(승용, 승합)
                    if vehicle_class in ['01_car', '03_van']:
                        e_cold_ratio = calculate_cold_hot_ratio(fuel, pollutant, T)
                        delta_ratio = e_cold_ratio - 1
                        E_cold = (beta * vehicle_count * VKT * eHOT * delta_ratio * combo_ratio)
                        emission = Eij + E_cold
                    else:
                        emission = Eij

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
