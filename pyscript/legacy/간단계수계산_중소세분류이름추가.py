import pandas as pd

# 파일 경로 설정
voc_file = r"C:\dense_traffic_emi\factor\factor_calculated_VOC_only.xlsx"
scc_file = r"C:\dense_traffic_emi\factor\scc_test.xlsx"

# 두 파일 읽기
voc_df = pd.read_excel(voc_file)
scc_df = pd.read_excel(scc_file)

# 새로운 분류 열 추가
voc_df['중분류'] = ''
voc_df['소분류'] = ''
voc_df['세분류'] = ''

# SCC 매칭 및 분류 정보 입력
for idx, row in voc_df.iterrows():
    # 현재 VOC 데이터의 SCC 값과 일치하는 SCC 데이터 찾기
    matching_scc = scc_df[scc_df['SCC'] == row['SCC']]

    # 매칭되는 데이터가 있으면 분류 정보 입력
    if not matching_scc.empty:
        voc_df.at[idx, '중분류'] = matching_scc.iloc[0]['중분류']
        voc_df.at[idx, '소분류'] = matching_scc.iloc[0]['소분류']
        voc_df.at[idx, '세분류'] = matching_scc.iloc[0]['세분류']

# 결과 저장
output_path = r"C:\dense_traffic_emi\factor\factor_calculated_VOC_with_categories.xlsx"
voc_df.to_excel(output_path, index=False)

# 결과 확인을 위한 출력
print("매칭 완료되었습니다.")
print(f"\n전체 VOC 데이터 수: {len(voc_df)} 행")
print(f"분류 정보가 매칭된 데이터 수: {len(voc_df[voc_df['중분류'] != ''])} 행")
print("\n처음 5개 행의 매칭 결과:")
print(voc_df[['SCC', '중분류', '소분류', '세분류']].head())