import pandas as pd


def split_rows_by_linebreak(file_path, sheet_name, target_column, output_file):
    # 엑셀 파일 읽기
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # 새로운 데이터 저장을 위한 리스트 생성
    new_rows = []

    # 각 행을 순회하면서 줄바꿈 기준으로 분리
    for _, row in df.iterrows():
        # 대상 열에서 줄바꿈으로 분리
        split_values = str(row[target_column]).split('\n')

        # 분리된 값들 각각에 대해 새로운 행 생성
        for value in split_values:
            value = value.strip()  # 줄바꿈 후 앞뒤 공백 제거
            if value:  # 값이 비어있지 않은 경우에만 처리
                new_row = row.copy()
                new_row[target_column] = value
                new_rows.append(new_row)

    # 새로운 데이터프레임 생성
    new_df = pd.DataFrame(new_rows)

    # 결과를 엑셀 파일로 저장
    new_df.to_excel(output_file, index=False)

# 파라미터 설정
file_path = r'C:\dense_traffic_emi\output\계수통합분석.xlsx'  # 입력 파일 경로
sheet_name = 'Sheet'  # 시트 이름
target_column = 'SCC'  # 줄바꿈으로 분리할 대상 열 이름
output_file = r'C:\dense_traffic_emi\output\계수통합분석_줄바꿈새행분리.xlsx'  # 출력 파일 경로

# 함수 호출
split_rows_by_linebreak(file_path, sheet_name, target_column, output_file)

print("작업이 완료되었습니다.")
