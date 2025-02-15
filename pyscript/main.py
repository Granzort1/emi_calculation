import os
import pandas as pd

# 모듈 임포트
from emi_calculator.data_io import load_input_data, make_output_directory
from emi_calculator.emission_calculation import calculate_emissions

if __name__ == '__main__':
    # ----- 입력 파일과 매개변수 매핑 -----
    경기동로_동측_VKT = 2.067
    경기동로_서측_VKT = 1.476
    경기동로_북측_VKT = 1.618
    경기대로_VKT     = 1.667
    동부대로_남측_VKT = 3.21

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
        '경기대로_겨울_도로_차량수.xlsx': {
            'VKT': 경기대로_VKT,
            'V': 64,
            'T': 0.4,
            'ta_min': -4.3,
            'ta_rise': 8.5
        },
    }

    # 시즌별 input 파일 리스트
    input_files_seasons = [
        input_files,       # 여름
        input_files_fall,  # 가을
        input_files_winter # 겨울
    ]

    # 입력/출력 기본 경로
    input_base_dir = 'C:/emi_calculation/input'
    output_base_dir = 'C:/emi_calculation/output'

    # 공통 파라미터
    sL = 0.06
    P_4N = 0

    # 시즌별로 처리
    for season_idx, season_files in enumerate(input_files_seasons):
        if season_idx == 0:
            print("=== 여름 데이터 처리 시작 ===")
        elif season_idx == 1:
            print("=== 가을 데이터 처리 시작 ===")
        elif season_idx == 2:
            print("=== 겨울 데이터 처리 시작 ===")

        # 각 파일별 처리
        for input_file, params in season_files.items():
            full_input_path = os.path.join(input_base_dir, input_file)
            output_dir = make_output_directory(output_base_dir, input_file)

            try:
                inputdata = load_input_data(full_input_path)
                print(f"처리 중: {input_file}")
                print(f"매개변수: VKT={params['VKT']}, V={params['V']}, T={params['T']}, ta_min={params['ta_min']}, ta_rise={params['ta_rise']}")

                # 배출량 계산
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

        # 시즌별 완료 메시지
        if season_idx == 0:
            print("=== 여름 데이터 처리 완료 ===\n")
        elif season_idx == 1:
            print("=== 가을 데이터 처리 완료 ===\n")
        elif season_idx == 2:
            print("=== 겨울 데이터 처리 완료 ===\n")
