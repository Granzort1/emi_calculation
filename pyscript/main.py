import os
import pandas as pd
import re
import shutil

# === [ 기존 모듈 임포트(수정 없음) ] ===
from emi_calculator.data_io import load_input_data, make_output_directory
from emi_calculator.emission_calculation import calculate_emissions

# === [ 후처리 1: 폴더 이름 변경을 위한 유틸 함수 ] ===
def get_direction_code(folder_name):
    """폴더명에서 동측/서측/남측/북측 등을 찾아서 E/W/S/N 코드 반환"""
    direction_mapping = {
        '동측': 'E',
        '서측': 'W',
        '남측': 'S',
        '북측': 'N',
    }
    for kr_dir, en_dir in direction_mapping.items():
        if kr_dir in folder_name:
            return en_dir
    return ''

def rename_folders(root_path):
    """
    기존 후처리 스크립트 #1 기능:
    (root_path) 내에서 '_emi'가 포함된 폴더를 찾아
    ex) '경기동로_여름_동측도로_emi' -> '경기동로_E_여름_배출량'
    으로 이름을 바꿔주는 함수.
    """
    renamed_folders = []  # 변경 결과 저장
    
    # root_path 내의 모든 폴더를 검사
    for folder_name in os.listdir(root_path):
        old_path = os.path.join(root_path, folder_name)
        if os.path.isdir(old_path) and '_emi' in folder_name:
            # 도로명 추출 (첫 번째 '_' 전까지)
            road_name = folder_name.split('_')[0]
            
            # 계절 추출
            seasons = ['봄', '여름', '가을', '겨울']
            season = next((s for s in seasons if s in folder_name), '')

            # 방향 코드 추출
            direction = get_direction_code(folder_name)
            
            # 새 폴더명 생성
            if direction:
                new_name = f"{road_name}_{direction}_{season}_배출량"
            else:
                new_name = f"{road_name}_{season}_배출량"

            new_path = os.path.join(root_path, new_name)

            try:
                os.rename(old_path, new_path)
                renamed_folders.append({
                    'old_name': folder_name,
                    'new_name': new_name
                })
                print(f"폴더명 변경 완료: {folder_name} → {new_name}")
            except Exception as e:
                print(f"폴더명 변경 실패: {folder_name} - {str(e)}")
    
    return renamed_folders


# === [ 후처리 2: 엑셀파일 이름 변경 및 missing_emission_factors 이동 함수 ] ===
def get_substance_name(filename):
    """파일명에서 물질명을 추출 (ex: emission_results_PM10_4n.xlsx -> 'PM10', 재비산 PM10 등)"""
    if filename.startswith('resuspension_dust'):
        # resuspension_dust_TSP_4n.xlsx 같은 형식
        parts = filename.split('_')
        # parts[0] = 'resuspension'
        # parts[1] = 'dust'
        # parts[2] = 'TSP'
        # parts[3] = '4n.xlsx'
        substance = parts[2]  # 'TSP'
        return f"재비산_{substance}"

    # emission_results_물질_4n.xlsx 처리
    parts = filename.split('_')
    # 재비산 포함된 파일이면 "PM10_재비산" 형태 추출
    if '재비산' in filename:
        # 예: emission_results_PM10_재비산_4n.xlsx -> parts중 "PM10" 뒤에 '재비산'
        resuspension_idx = parts.index('재비산')
        return f"{parts[resuspension_idx - 1]}_재비산"

    # 일반 케이스: emission_results_PM10_4n.xlsx -> 'PM10'
    return parts[-2]


def process_excel_files(root_path):
    """
    기존 후처리 스크립트 #2 기능:
    1) 각 폴더 내 엑셀파일들을 새 이름으로 변경
       - emission_results_* -> 폴더명_물질.xlsx
       - resuspension_dust_* -> 폴더명_재비산_TSP등.xlsx
    2) missing_emission_factors.xlsx 는 root_path/missing_emission_factors 폴더로 이동
    """
    # missing_emission_factors 파일들을 한 데 모을 폴더 생성
    missing_factors_dir = os.path.join(root_path, "missing_emission_factors")
    if not os.path.exists(missing_factors_dir):
        os.makedirs(missing_factors_dir)

    processed_files = []

    # root_path 내 모든 하위 폴더 순회
    for folder_name in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder_name)

        # 폴더가 아니거나, missing_emission_factors 폴더는 건너뛰기
        if not os.path.isdir(folder_path) or folder_name == "missing_emission_factors":
            continue

        # 폴더 내의 .xlsx 파일들 처리
        for filename in os.listdir(folder_path):
            if not filename.endswith('.xlsx'):
                continue

            old_file_path = os.path.join(folder_path, filename)

            # missing_emission_factors.xlsx 파일
            if filename == "missing_emission_factors.xlsx":
                new_filename = f"{folder_name}_missing_emission_factors.xlsx"
                new_file_path = os.path.join(missing_factors_dir, new_filename)
                try:
                    shutil.move(old_file_path, new_file_path)
                    processed_files.append({
                        'old_name': filename,
                        'new_name': new_filename,
                        'action': 'moved to missing_factors folder'
                    })
                except Exception as e:
                    print(f"missing_emission_factors.xlsx 이동 실패: {str(e)}")
                continue

            # emission_results_*, resuspension_dust_* 형태 파일
            if filename.startswith('emission_results') or filename.startswith('resuspension_dust'):
                substance_name = get_substance_name(filename)
                # 새 파일명 = "폴더명_물질.xlsx" 예: "경기동로_E_여름_배출량_PM10.xlsx"
                new_filename = f"{folder_name}_{substance_name}.xlsx"
                new_file_path = os.path.join(folder_path, new_filename)

                try:
                    os.rename(old_file_path, new_file_path)
                    processed_files.append({
                        'old_name': filename,
                        'new_name': new_filename,
                        'action': 'renamed'
                    })
                except Exception as e:
                    print(f"엑셀 파일명 변경 실패: {filename} - {str(e)}")

    return processed_files


# === [ 메인 실행부: 기존 배출량 계산 → 후처리 통합 ] ===
if __name__ == '__main__':
    # ----- 기존 main.py 내용 (데이터 처리 파트) -----
    경기동로_동측_VKT = 2.067
    경기동로_서측_VKT = 1.476
    경기동로_북측_VKT = 1.618
    경기대로_VKT     = 1.667
    동부대로_남측_VKT = 1.29
    동부대로_북측_VKT = 1.96

    # 여름 데이터
    input_files = {
        '경기동로_여름_동측도로_차량수.xlsx': {
            'VKT': 경기동로_동측_VKT,
            'V': 64,
            'T': 31.1,
            'ta_min': 28.1,
            'ta_rise': 8.3
        },
        '경기동로_여름_서측도로_차량수.xlsx': {
            'VKT': 경기동로_서측_VKT,
            'V': 64,
            'T': 31.1,
            'ta_min': 28.1,
            'ta_rise': 8.3
        },
        '경기동로_여름_북측도로_차량수.xlsx': {
            'VKT': 경기동로_북측_VKT,
            'V': 64,
            'T': 31.1,
            'ta_min': 28.1,
            'ta_rise': 8.3
        },
        '동부대로_여름_남측도로_차량수.xlsx': {
            'VKT': 동부대로_남측_VKT,
            'V': 64,
            'T': 31.1,
            'ta_min': 28.1,
            'ta_rise': 8.3
        },
        '동부대로_여름_북측도로_차량수.xlsx': {
            'VKT': 동부대로_북측_VKT,
            'V': 64,
            'T': 31.1,
            'ta_min': 28.1,
            'ta_rise': 8.3
        },
        '경기대로_여름_도로_차량수.xlsx': {
            'VKT': 경기대로_VKT,
            'V': 64,
            'T': 31.1,
            'ta_min': 28.1,
            'ta_rise': 8.3
        },
    }

    input_files_fall = {
        '경기동로_가을_동측도로_차량수.xlsx': {
            'VKT': 경기동로_동측_VKT,
            'V': 64,
            'T': 18.4,
            'ta_min': 15.2,
            'ta_rise': 8
        },
        '경기동로_가을_서측도로_차량수.xlsx': {
            'VKT': 경기동로_서측_VKT,
            'V': 64,
            'T': 18.4,
            'ta_min': 15.2,
            'ta_rise': 8
        },
        '경기동로_가을_북측도로_차량수.xlsx': {
            'VKT': 경기동로_북측_VKT,
            'V': 64,
            'T': 18.4,
            'ta_min': 15.2,
            'ta_rise': 8
        },
        '동부대로_가을_남측도로_차량수.xlsx': {
            'VKT': 동부대로_남측_VKT,
            'V': 64,
            'T': 18.4,
            'ta_min': 15.2,
            'ta_rise': 8
        },
        '동부대로_가을_북측도로_차량수.xlsx': {
            'VKT': 동부대로_북측_VKT,
            'V': 64,
            'T': 18.4,
            'ta_min': 15.2,
            'ta_rise': 8
        },
        '경기대로_가을_도로_차량수.xlsx': {
            'VKT': 경기대로_VKT,
            'V': 64,
            'T': 18.4,
            'ta_min': 15.2,
            'ta_rise': 8
        },
    }

    input_files_winter = {
        '경기동로_겨울_동측도로_차량수.xlsx': {
            'VKT': 경기동로_동측_VKT,
            'V': 64,
            'T': 0.4,
            'ta_min': -4.3,
            'ta_rise': 8.5
        },
        '경기동로_겨울_서측도로_차량수.xlsx': {
            'VKT': 경기동로_서측_VKT,
            'V': 64,
            'T': 0.4,
            'ta_min': -4.3,
            'ta_rise': 8.5
        },
        '경기동로_겨울_북측도로_차량수.xlsx': {
            'VKT': 경기동로_북측_VKT,
            'V': 64,
            'T': 0.4,
            'ta_min': -4.3,
            'ta_rise': 8.5
        },
        '동부대로_겨울_남측도로_차량수.xlsx': {
            'VKT': 동부대로_남측_VKT,
            'V': 64,
            'T': 0.4,
            'ta_min': -4.3,
            'ta_rise': 8.5
        },
        '동부대로_겨울_북측도로_차량수.xlsx': {
            'VKT': 동부대로_북측_VKT,
            'V': 64,
            'T': 0.4,
            'ta_min': -4.3,
            'ta_rise': 8.5
        },
        '경기대로_겨울_도로_차량수.xlsx': {
            'VKT': 경기대로_VKT,
            'V': 64,
            'T': 0.4,
            'ta_min': -4.3,
            'ta_rise': 8.5
        },
    }

    input_files_seasons = [
        input_files,       # 여름
        input_files_fall,  # 가을
        input_files_winter # 겨울
    ]

    input_base_dir = 'C:/emi_calculation/input'
    output_base_dir = 'C:/emi_calculation/output'

    sL = 0.06
    P_4N = 0

    # === 각 시즌별 배출량 계산 ===
    for season_idx, season_files in enumerate(input_files_seasons):
        if season_idx == 0:
            print("=== 여름 데이터 처리 시작 ===")
        elif season_idx == 1:
            print("=== 가을 데이터 처리 시작 ===")
        else:
            print("=== 겨울 데이터 처리 시작 ===")

        # 시즌별 파일을 순회하며 배출량 계산
        for input_file, params in season_files.items():
            full_input_path = os.path.join(input_base_dir, input_file)
            output_dir = make_output_directory(output_base_dir, input_file)

            try:
                inputdata = load_input_data(full_input_path)
                print(f"처리 중: {input_file}")
                print(f"매개변수: VKT={params['VKT']}, V={params['V']}, T={params['T']}, "
                      f"ta_min={params['ta_min']}, ta_rise={params['ta_rise']}")

                calculate_emissions(
                    inputdata,
                    VKT=params['VKT'],
                    V=params['V'],
                    T=params['T'],
                    ta_min=params['ta_min'],
                    ta_rise=params['ta_rise'],
                    P_4N=P_4N,
                    sL=sL,
                    output_dir=output_dir
                )

                print(f"완료: {input_file}\n")
            except Exception as e:
                print(f"오류 발생 ({input_file}): {str(e)}\n")

        if season_idx == 0:
            print("=== 여름 데이터 처리 완료 ===\n")
        elif season_idx == 1:
            print("=== 가을 데이터 처리 완료 ===\n")
        else:
            print("=== 겨울 데이터 처리 완료 ===\n")

    print("=== 모든 시즌 데이터 처리(배출량 계산) 완료 ===\n")

    # === [후처리 1] 폴더명 변경 실행 여부 확인 ===
    print(f"다음 경로에서 폴더명을 변경합니다: {output_base_dir}")
    confirm = input("폴더명 변경을 진행하시겠습니까? (y/n): ")
    if confirm.lower() == 'y':
        renamed_list = rename_folders(output_base_dir)
        print("\n=== 변경된 폴더 목록 ===")
        for item in renamed_list:
            print(f"이전: {item['old_name']} -> 이후: {item['new_name']}")
        print("-" * 50)
        print("폴더명 변경 후처리 완료\n")
    else:
        print("폴더명 변경 작업을 건너뜁니다.\n")

    # === [후처리 2] 엑셀파일 처리 실행 여부 확인 ===
    print(f"다음 경로의 엑셀파일들을 처리합니다: {output_base_dir}")
    confirm2 = input("엑셀파일 후처리를 진행하시겠습니까? (y/n): ")
    if confirm2.lower() == 'y':
        processed = process_excel_files(output_base_dir)
        print("\n=== 처리된 파일 목록 ===")
        for item in processed:
            print(f"이전: {item['old_name']}")
            print(f"이후: {item['new_name']}")
            print(f"작업: {item['action']}")
            print("-" * 50)
        print(f"\n총 {len(processed)}개 파일이 처리되었습니다.")
        print("엑셀파일 후처리 완료\n")
    else:
        print("엑셀파일 후처리 작업을 건너뜁니다.\n")

    print("=== 모든 후처리 작업이 완료되었습니다. ===")
