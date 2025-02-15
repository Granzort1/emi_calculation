import os
import shutil
import re


def get_substance_name(filename):
    """파일명에서 물질명 추출"""
    # resuspension_dust 파일 처리
    if filename.startswith('resuspension_dust'):
        parts = filename.split('_')
        # TSP나 다른 물질명 추출
        substance = parts[2]  # TSP 위치
        return f"재비산_{substance}"

    # emission_results 파일 처리
    parts = filename.split('_')

    # 재비산이 포함된 경우
    if '재비산' in filename:
        resuspension_idx = parts.index('재비산')
        return f"{parts[resuspension_idx - 1]}_재비산"

    # 일반적인 경우 (emission_results_물질명_4n.xlsx 형식)
    return parts[-2]


def process_excel_files(root_path):
    # missing_emission_factors 파일을 모을 새 폴더 생성
    missing_factors_dir = os.path.join(root_path, "missing_emission_factors")
    if not os.path.exists(missing_factors_dir):
        os.makedirs(missing_factors_dir)

    # 처리된 파일 기록
    processed_files = []

    # 모든 하위 폴더 순회
    for folder_name in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder_name)

        # 폴더가 아니거나 missing_emission_factors 폴더인 경우 건너뛰기
        if not os.path.isdir(folder_path) or folder_name == "missing_emission_factors":
            continue

        # 폴더 내 모든 엑셀 파일 처리
        for filename in os.listdir(folder_path):
            if not filename.endswith('.xlsx'):
                continue

            old_file_path = os.path.join(folder_path, filename)

            # missing_emission_factors.xlsx 파일 처리
            if filename == "missing_emission_factors.xlsx":
                new_filename = f"{folder_name}_missing_emission_factors.xlsx"
                new_file_path = os.path.join(missing_factors_dir, new_filename)
                shutil.move(old_file_path, new_file_path)
                processed_files.append({
                    'old_name': filename,
                    'new_name': new_filename,
                    'action': 'moved to missing_factors folder'
                })
                continue

            # emission_results 또는 resuspension_dust로 시작하는 파일 처리
            if filename.startswith('emission_results') or filename.startswith('resuspension_dust'):
                substance_name = get_substance_name(filename)
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
                    print(f"오류 발생: {filename} 변경 실패 - {str(e)}")

    return processed_files


if __name__ == "__main__":
    # 대상 폴더 경로 설정
    target_path = r"C:\emi_calculation\output"

    # 실행 전 확인
    print(f"다음 경로의 엑셀 파일들을 처리합니다: {target_path}")
    confirm = input("계속하시겠습니까? (y/n): ")

    if confirm.lower() == 'y':
        # 파일 처리 실행
        processed = process_excel_files(target_path)

        # 처리 결과 출력
        print("\n=== 처리된 파일 목록 ===")
        for item in processed:
            print(f"이전: {item['old_name']}")
            print(f"이후: {item['new_name']}")
            print(f"작업: {item['action']}")
            print("-" * 50)

        print(f"\n총 {len(processed)}개 파일이 처리되었습니다.")
    else:
        print("작업이 취소되었습니다.")