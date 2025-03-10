import pandas as pd
import numpy as np
import os
import re
from datetime import datetime
import unicodedata

# 파일 경로
factor_file_path = r"C:\emi_calculation\factor\EFi_DF_Factor_ver7.xlsx"
output_dir = r"C:\emi_calculation\factor"
output_file_name = f"EFi_DF_Factor_ver7_updated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
output_path = os.path.join(output_dir, output_file_name)
missing_data_file = os.path.join(output_dir, f"missing_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")

# 추가데이터 (여기에 직접 입력)
additional_data_text = """VOCs (단위: NMVOC분율),휘발유  (y<2010),휘발유 (y>=2010),"디젤_승용,벤,소형화물 (y<2010)","디젤_승용,벤,소형화물  (y>=2010)","중대형화물, 특수, 버스",LPG,,중금속(단위:g/km),휘발유,"경유, y<2007","경유, 2007<y<2010","경유, 2010<=y",CNG,,카보닐화합물 (단위: NMVOC분율),휘발유 (y<2010),휘발유 (y>=2010),"디젤_승용,벤,소형화물 (y<2010)","디젤_승용,벤,소형화물  (y>=2010)","중대형화물, 특수, 버스",LPG,,PAHs(단위:g/km),휘발유,"디젤_승용,벤,소형화물","중대형화물, 특수, 버스",LPG,,중금속(단위:ppm/wt fuel),휘발유,경유,,PAHs(단위:g/km),휘발유,"디젤_승용,벤,소형화물","중대형화물, 특수, 버스",LPG
Benzene,5.60%,8.20%,2.00%,0.40%,0.10%,0.60%,,Cr(Ⅵ),1.93E-08,3.22E-08,9.55E-09,3.48E-09,3.38E-10,,Formaldehyde,1.70%,1.20%,12.00%,13.30%,8.40%,1.60%,,Naphthalene,6.1E-04,2.1E-03,5.7E-05,4.00E-05,,Cd,0.0002,0.00005,,Naphthalene,6.1E-04,2.1E-03,5.7E-05,4.00E-05
Toluene,11.00%,16.00%,0.70%,0.10%,0.00%,1.20%,,Ni,0.000002415,0.00002254,1.11E-06,4.23E-07,1.61E-08,,Acetaldehyde,0.80%,0.50%,6.50%,7.20%,4.60%,1.80%,,Benzo[a]pyrene,3.2E-07,6.3E-07,9.0E-07,1.00E-08,,,,,,Benzo[a]pyrene,3.2E-07,6.3E-07,9.0E-07,1.00E-08
Ethylbenzene,1.90%,2.80%,0.30%,0.10%,0.00%,0.20%,,Mn,2.14E-06,0.00001288,1.10E-06,0.000000322,2.14E-06,,Acrolein,0.20%,0.10%,3.60%,4.00%,1.80%,0.60%,,,,,,,,,,,,,,,,
Xylenes,7.70%,11.20%,0.90%,0.20%,1.40%,1.00%,,,,,,,,,Propionaldehyde,0.10%,0.00%,1.80%,2.00%,1.30%,0.70%,,,,,,,,,,,,,,,,
Styrene,1.00%,1.50%,0.40%,0.10%,0.60%,0.00%,,,,,,,,,2-Butanone,0.10%,0.00%,1.20%,1.30%,0.00%,0.00%,,,,,,,,,,,,,,,,
"1,3,5-Trimethylbenzene",1.40%,2.10%,0.30%,0.10%,0.50%,0.10%,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
"1,2,4-Trimethylbenzene",4.20%,6.10%,0.60%,0.10%,0.90%,0.30%,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
Hexane,1.60%,1.60%,0.00%,0.00%,0.00%,0.00%,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,"""

# 추가데이터를 데이터프레임으로 변환
import io

additional_data_df = pd.read_csv(io.StringIO(additional_data_text), header=0)

print("추가데이터 구조 확인:")
print(additional_data_df.shape)
print("추가데이터 열 목록:")
print(additional_data_df.columns.tolist())

# 계수파일 읽기
print("계수파일 로딩 중...")
try:
    # 데이터 형식을 명시적으로 지정하여 읽기
    factor_df = pd.read_excel(factor_file_path)
    print(f"계수파일 로딩 완료: {factor_df.shape[0]}행, {factor_df.shape[1]}열")
except Exception as e:
    print(f"계수파일 로딩 오류: {e}")
    exit(1)


def normalize_str(s):
    """문자열을 정규화하고 모든 특수 문자 제거"""
    if not isinstance(s, str):
        return s

    # 유니코드 정규화 (NFC)
    s = unicodedata.normalize('NFKD', s)

    # 모든 따옴표 유형 제거
    s = s.replace("'", "").replace("'", "").replace("'", "")
    s = s.replace(""", "").replace(""", "").replace("″", "")
    s = s.replace('"', '').replace("「", "").replace("」", "")
    s = s.replace('`', '')

    # 알파벳, 숫자, 일반 연산자만 유지
    s = re.sub(r'[^\w\s<>=!+\-*/.,()]', '', s)

    return s.strip()


def check_year_range(condition, year):
    """
    연식 조건을 직접 확인하는 함수 - eval 사용하지 않음
    """
    # 문자열 정규화
    condition = normalize_str(condition)

    # 'all'인 경우
    if condition.lower() == 'all':
        return True

    # 단순 조건들에 대한 처리

    # 연도 == Y 형식 (예: 2000=Y)
    match = re.search(r'(\d+)=Y', condition)
    if match:
        year_val = int(match.group(1))
        return year == year_val

    # Y 범위 (예: 2000<=Y<=2002)
    match = re.search(r'(\d+)<=Y<=(\d+)', condition)
    if match:
        start_year = int(match.group(1))
        end_year = int(match.group(2))
        return start_year <= year <= end_year

    # 상한선 (예: Y<=2002)
    match = re.search(r'Y<=(\d+)', condition)
    if match:
        end_year = int(match.group(1))
        return year <= end_year

    # 하한선 (예: 2002<=Y)
    match = re.search(r'(\d+)<=Y', condition)
    if match and not '=' in condition.split('<=Y')[1]:
        start_year = int(match.group(1))
        return year >= start_year

    # 작은 값 (예: Y<2002 또는 2002>Y)
    match_1 = re.search(r'Y<(\d+)', condition)
    match_2 = re.search(r'(\d+)>Y', condition)
    if match_1:
        end_year = int(match_1.group(1))
        return year < end_year
    elif match_2:
        end_year = int(match_2.group(1))
        return year < end_year

    # 큰 값 (예: Y>2002 또는 2002<Y)
    match_1 = re.search(r'Y>(\d+)', condition)
    match_2 = re.search(r'(\d+)<Y', condition)
    if match_1:
        start_year = int(match_1.group(1))
        return year > start_year
    elif match_2:
        start_year = int(match_2.group(1))
        return year > start_year

    # 일치하는 패턴이 없으면 기본적으로 True 반환 (안전 옵션)
    return True


def extract_year_range_regex(year_str):
    """
    연식 정보에서 연도 범위 추출 - 정규식 활용
    """
    # 문자열이 아니면 처리하지 않음
    if not isinstance(year_str, str):
        return (None, None)

    # 문자열 정규화
    clean_year_str = normalize_str(year_str)

    # y<2010 형식 처리
    match = re.search(r'y<(\d+)', clean_year_str)
    if match:
        year = int(match.group(1))
        return (None, year - 1)

    # y>=2010 형식 처리
    match = re.search(r'y>=(\d+)', clean_year_str)
    if match:
        year = int(match.group(1))
        return (year, None)

    # 2007<y<2010 형식 처리
    match = re.search(r'(\d+)<y<(\d+)', clean_year_str)
    if match:
        start_year = int(match.group(1)) + 1
        end_year = int(match.group(2)) - 1
        return (start_year, end_year)

    # 2010<=y 형식 처리
    match = re.search(r'(\d+)<=y', clean_year_str)
    if match and ',' not in clean_year_str:
        year = int(match.group(1))
        return (year, None)

    # y<2007, 형식 처리 (쉼표가 있는 경우)
    if ', y<' in clean_year_str:
        match = re.search(r', y<(\d+)', clean_year_str)
        if match:
            year = int(match.group(1))
            return (None, year - 1)

    # 2007<y<2010, 형식 처리 (쉼표가 있는 경우)
    if ', ' in clean_year_str:
        parts = clean_year_str.split(', ')
        match = re.search(r'(\d+)<y<(\d+)', parts[1])
        if match:
            start_year = int(match.group(1)) + 1
            end_year = int(match.group(2)) - 1
            return (start_year, end_year)

    # 2010<=y, 형식 처리 (쉼표가 있는 경우)
    if ', ' in clean_year_str:
        parts = clean_year_str.split(', ')
        match = re.search(r'(\d+)<=y', parts[1])
        if match:
            year = int(match.group(1))
            return (year, None)

    # 연도 정보가 없는 경우
    return (None, None)


def is_year_in_range_direct(factor_year_str, condition_start, condition_end):
    """연식 조건이 범위에 포함되는지 확인 - eval 없이 직접 확인"""
    try:
        # 정보가 없으면 모든 경우 매칭으로 취급
        if not isinstance(factor_year_str, str) or factor_year_str.strip() == '':
            return True

        # 문자열 정규화
        clean_year_str = normalize_str(factor_year_str)

        # 'all'인 경우
        if clean_year_str.lower() == 'all':
            return True

        # factor_year_str에서 연도 범위 추출
        factor_range = extract_year_range_regex(clean_year_str)
        factor_start, factor_end = factor_range

        # 직접 테스트로 확인
        # 연식 조건에 대해 여러 연도로 테스트
        test_years = []

        # 특정 연도만 가리키는 조건일 경우
        if '=Y' in clean_year_str:
            match = re.search(r'(\d+)=Y', clean_year_str)
            if match:
                test_year = int(match.group(1))
                test_years = [test_year]
        else:
            # 범위를 테스트하기 위한 샘플 연도 생성
            if factor_start is not None and factor_end is not None:
                # 범위가 명확한 경우
                step = max(1, (factor_end - factor_start) // 5)  # 최대 5개 포인트
                test_years = list(range(factor_start, factor_end + 1, step))
            elif factor_start is not None:
                # 하한만 있는 경우
                test_years = [factor_start, factor_start + 10, factor_start + 20]
            elif factor_end is not None:
                # 상한만 있는 경우
                test_years = [factor_end - 20, factor_end - 10, factor_end]
            else:
                # 모든 범위가 없는 경우, 대표적인 연도 사용
                test_years = [1990, 2000, 2010, 2020]

        # 테스트 연도 중 하나라도 조건 범위에 포함되면 True
        for year in test_years:
            # 조건 범위 확인
            if condition_start is not None and year < condition_start:
                continue
            if condition_end is not None and year > condition_end:
                continue

            # factor_year_str 조건에 맞는지 확인
            if check_year_range(clean_year_str, year):
                return True

        # 기본 조건 확인 (factor_range와 condition 범위의 겹침 확인)
        # factor_range가 범위가 없으면 무조건 포함
        if factor_start is None and factor_end is None:
            return True

        # condition에 범위가 없으면 모든 연도 포함
        if condition_start is None and condition_end is None:
            return True

        # 조건 시작 연도만 있는 경우 (예: >=2010)
        if condition_start is not None and condition_end is None:
            # 계수파일 종료 연도가 없거나 조건 시작 연도보다 크거나 같으면 포함
            return factor_end is None or factor_end >= condition_start

        # 조건 종료 연도만 있는 경우 (예: <2010)
        if condition_start is None and condition_end is not None:
            # 계수파일 시작 연도가 없거나 조건 종료 연도보다 작거나 같으면 포함
            return factor_start is None or factor_start <= condition_end

        # 조건 범위가 모두 있는 경우
        # 계수파일 연도 범위와 조건 연도 범위가 겹치는지 확인
        if factor_start is not None and factor_end is not None:
            # 범위가 겹치는지 확인
            return not (factor_end < condition_start or factor_start > condition_end)

        # 계수파일 시작 연도만 있는 경우
        if factor_start is not None and factor_end is None:
            return factor_start <= condition_end

        # 계수파일 종료 연도만 있는 경우
        if factor_start is None and factor_end is not None:
            return factor_end >= condition_start

        return False

    except Exception as e:
        print(f"연도 범위 확인 중 오류 발생: {factor_year_str} vs {condition_start}-{condition_end} - {str(e)}")
        # 오류 발생 시 값이 입력되도록 함
        return True


# 추가데이터 변환 및 구조화 함수
def parse_additional_data(df):
    # 빈 열 찾기
    empty_cols = df.columns[df.columns.str.strip() == '']
    empty_col_indices = [df.columns.get_loc(col) for col in empty_cols]

    # 물질군 구분
    groups = []
    start_idx = 0

    for idx in empty_col_indices:
        if idx > start_idx:  # 비어있지 않은 열 그룹이 있으면
            group_cols = df.columns[start_idx:idx]
            groups.append((group_cols[0], group_cols[1:]))
        start_idx = idx + 1

    # 마지막 그룹 추가
    if start_idx < len(df.columns):
        group_cols = df.columns[start_idx:]
        groups.append((group_cols[0], group_cols[1:]))

    return groups


def get_conditions_from_header(header):
    """헤더에서 조건 추출 - 개선된 버전"""
    fuel_type = None
    year_range = (None, None)
    vehicle_types = []

    # 문자열이 아니면 처리하지 않음
    if not isinstance(header, str):
        return {
            'fuel_type': fuel_type,
            'year_range': year_range,
            'vehicle_types': vehicle_types
        }

    # 문자열 정규화
    clean_header = normalize_str(header)
    
    # 헤더에 .1, .2 등의 접미사 제거 (중복 열로 인해 pandas가 자동으로 추가)
    clean_header = re.sub(r'\.\d+$', '', clean_header)

    # 연료 타입 및 연도 정보 추출
    if '휘발유' in clean_header:
        fuel_type = '휘발유'
        # 연도 정보 추출
        if '(' in clean_header and ')' in clean_header:
            year_info = clean_header.split('(')[1].split(')')[0].strip()
            year_range = extract_year_range_regex(year_info)
    elif '경유' in clean_header:
        fuel_type = '경유'
        # 연도 정보 추출
        if '(' in clean_header and ')' in clean_header:
            year_info = clean_header.split('(')[1].split(')')[0].strip()
            year_range = extract_year_range_regex(year_info)
        elif ',' in clean_header:
            year_info = clean_header.split(',')[1].strip()
            year_range = extract_year_range_regex(year_info)
    elif 'LPG' in clean_header:
        fuel_type = 'LPG'
    elif 'CNG' in clean_header:
        fuel_type = 'CNG'

    # 차종 정보 추출
    if '디젤_승용,벤,소형화물' in clean_header or (
            '디젤' in clean_header and ('승용' in clean_header or '벤' in clean_header or '소형화물' in clean_header)):
        fuel_type = '경유'  # 디젤은 경유와 동일하게 처리
        vehicle_types = [
            ('승용차', None),
            ('승합차', None),
            ('RV', None),
            ('화물차', '경형'),
            ('화물차', '소형')
        ]
    elif ('중대형화물' in clean_header) or ('특수' in clean_header) or ('버스' in clean_header):
        fuel_type = '경유'  # 중대형화물, 특수, 버스는 경유 차량으로 가정
        vehicle_types = [
            ('화물차', '중형'),
            ('화물차', '대형'),
            ('특수차', None),
            ('버스', None)
        ]

    return {
        'fuel_type': fuel_type,
        'year_range': year_range,
        'vehicle_types': vehicle_types
    }


def create_new_column_name(group_name, substance):
    """새로운 열 이름 생성 - 요구사항에 맞게 수정"""
    # 물질 이름에서 특수 문자 제거
    clean_substance = substance.replace(',', '').replace('[', '').replace(']', '')
    clean_substance = clean_substance.replace('(', '').replace(')', '')
    
    # 물질군 이름에 따라 다른 열 이름 형식 사용
    if 'VOCs' in group_name:
        return f"VOC_frac_{clean_substance}"
    elif '카보닐화합물' in group_name:
        # 카보닐화합물도 NMVOC 분율이므로 동일 접두사 사용
        return f"VOC_frac_{clean_substance}"
    elif '중금속' in group_name:
        if 'g/km' in group_name:
            return f"HM_gkm_{clean_substance}"
        elif 'ppm/wt fuel' in group_name:
            return f"HM_ppm_{clean_substance}"
    elif 'PAHs' in group_name:
        return f"PAH_gkm_{clean_substance}"
    else:
        # 단위 정보가 없는 경우
        return f"{clean_substance}"


def convert_coefficient(coefficient):
    """계수값 변환 함수 - 오류 처리 향상"""
    # 물질명인지 확인
    if isinstance(coefficient, str) and any(x.isalpha() for x in coefficient):
        return 0.0
        
    # 비어있는 값 처리
    if pd.isna(coefficient):
        return 0.0
        
    # 문자열을 숫자로 변환
    if isinstance(coefficient, str):
        try:
            # 퍼센트 처리
            if '%' in coefficient:
                return float(coefficient.replace('%', '')) / 100
                
            # 과학적 표기법 처리
            elif 'E-' in coefficient or 'E+' in coefficient:
                # 6.1.E-04와 같은 형식 처리
                cleaned = coefficient.replace('.E', 'E').replace('E', 'e')
                return float(cleaned)
                
            # 일반 숫자
            else:
                return float(coefficient)
                
        except Exception as e:
            print(f"    경고: 값을 숫자로 변환할 수 없음: {coefficient} - {str(e)}")
            return 0.0
            
    # 이미 숫자면 그대로 반환
    return float(coefficient)


def add_coefficients_to_factor_file():
    """계수파일에 계수 추가"""
    global factor_df, error_count

    # 추가데이터 구조화
    groups = parse_additional_data(additional_data_df)

    print(f"추출된 물질군 수: {len(groups)}")

    # 오류 감소를 위한 메시지 횟수 제한
    error_count = 0
    max_error_messages = 10

    # 각 물질군과 물질별로 계수 추가
    all_missing_data = {}
    
    # 실제 물질군 수 확인
    real_group_count = 0
    
    for group_name, condition_cols in groups:
        # 실제로 처리된 물질군 수 카운트
        real_group_count += 1
        print(f"\n물질군 처리 중 ({real_group_count}/{len(groups)}): {group_name}")

        # 물질군 이름에서 단위 정보 추출
        unit_info = "단위 없음"
        if '단위:' in group_name:
            unit_info = group_name.split('단위:')[1].strip(')')

        # 각 물질(행)에 대해 반복
        for _, row in additional_data_df.iterrows():
            substance = row.iloc[0]

            # 빈 값이거나 물질명이 아닌 경우 건너뛰기
            if pd.isna(substance) or substance in groups:
                continue

            print(f"  물질 처리 중: {substance}")

            # 이 물질에 대한 오류 메시지 제한 초기화
            error_count = 0

            # 새 열 이름 생성
            new_column_name = create_new_column_name(group_name, substance)
            print(f"    새 열 이름: {new_column_name}")

            # 초기값은 0으로 설정
            factor_df[new_column_name] = 0.0  # 명시적으로 float 타입 지정

            # 각 차종 조건에 맞는 행에 값 설정
            missing_data = []

            for idx, condition_col in enumerate(condition_cols):
                try:
                    coefficient = row.iloc[idx + 1]  # 첫 번째 열은 물질명이므로 +1

                    # 계수값 변환 (개선된 함수 사용)
                    coefficient = convert_coefficient(coefficient)
                    
                    # 0인 경우 건너뛰기 (선택적)
                    # if coefficient == 0:
                    #     continue

                    # 헤더에서 조건 추출
                    conditions = get_conditions_from_header(condition_col)
                    fuel_type = conditions['fuel_type']
                    year_range = conditions['year_range']
                    vehicle_types = conditions['vehicle_types']

                    print(f"    조건 처리 중: {condition_col}")
                    if error_count < max_error_messages:
                        print(f"      연료: {fuel_type}, 연도범위: {year_range}, 차종: {len(vehicle_types)}개")
                        error_count += 1

                    # 조건에 맞는 행 찾기
                    matching_rows_count = 0
                    non_matching_rows = []

                    for i, factor_row in factor_df.iterrows():
                        is_match = True

                        # 연료 조건 확인
                        if fuel_type is not None and factor_row['연료'] != fuel_type:
                            is_match = False
                            if error_count < max_error_messages:
                                non_matching_rows.append(f"연료 불일치: {factor_row['연료']} != {fuel_type}")
                                error_count += 1
                            continue

                        # 연식 조건 확인 - 개선된 함수 사용
                        if year_range != (None, None) and not is_year_in_range_direct(factor_row['연식'], year_range[0],
                                                                                       year_range[1]):
                            is_match = False
                            if error_count < max_error_messages:
                                non_matching_rows.append(f"연식 불일치: {factor_row['연식']} 범위 밖: {year_range}")
                                error_count += 1
                            continue

                        # 차종 조건 확인
                        if vehicle_types:
                            vehicle_match = False
                            for vehicle_type, sub_type in vehicle_types:
                                if factor_row['차종'] == vehicle_type:
                                    if sub_type is None or factor_row['소분류'] == sub_type:
                                        vehicle_match = True
                                        break

                            if not vehicle_match:
                                is_match = False
                                if error_count < max_error_messages:
                                    non_matching_rows.append(f"차종 불일치: {factor_row['차종']}/{factor_row['소분류']} 조건에 없음")
                                    error_count += 1
                                continue

                        # 모든 조건 만족시 계수값 설정
                        if is_match:
                            factor_df.at[i, new_column_name] = coefficient
                            matching_rows_count += 1

                    print(f"      매칭된 행 수: {matching_rows_count}")

                    if matching_rows_count == 0:
                        # 매칭된 행이 없는 경우 기록
                        details = f"물질: {substance}, 조건: {condition_col}, 연료: {fuel_type}, 연도범위: {year_range}, 차종: {vehicle_types}"
                        missing_data.append(details)
                        print(f"    경고: 매칭된 행이 없음! {details}")

                except Exception as e:
                    print(f"    오류 발생: {substance} / {condition_col} - {str(e)}")

            # 빈 셀 정보 저장
            if len(missing_data) > 0:
                all_missing_data[f"{new_column_name}"] = missing_data

    return all_missing_data


# 메인 실행
try:
    print("추가데이터 계수 추가 시작...")

    missing_data = add_coefficients_to_factor_file()

    # 결과 저장
    print(f"업데이트된 계수파일 저장 중... ({output_path})")
    # object 타입의 열 검사 및 처리
    for col in factor_df.columns:
        if factor_df[col].dtype == 'object' and col not in ['차종', '소분류', '연료', '연식']:
            try:
                factor_df[col] = pd.to_numeric(factor_df[col], errors='coerce').fillna(0)
            except Exception as e:
                print(f"열 변환 중 오류 발생: {col} - {str(e)}")
    
    # 결과 저장
    factor_df.to_excel(output_path, index=False)

    # 빈 셀 정보 저장
    if missing_data:
        print(f"빈 셀 정보 저장 중... ({missing_data_file})")
        missing_df = pd.DataFrame({
            '물질': [key for key in missing_data.keys() for _ in missing_data[key]],
            '미매칭 조건': [detail for key in missing_data.keys() for detail in missing_data[key]]
        })
        missing_df.to_excel(missing_data_file, index=False)

    print("작업 완료!")
    print(f"업데이트된 계수파일: {output_path}")
    if missing_data:
        print(f"빈 셀 정보 파일: {missing_data_file}")

except Exception as e:
    print(f"오류 발생: {str(e)}")