import os
import re

def get_direction_code(folder_name):
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
    # 변경된 폴더 이름을 저장할 리스트
    renamed_folders = []
    
    # root_path 내의 모든 폴더를 검사
    for folder_name in os.listdir(root_path):
        if os.path.isdir(os.path.join(root_path, folder_name)) and '_emi' in folder_name:
            # 기존 폴더 전체 경로
            old_path = os.path.join(root_path, folder_name)
            
            # 도로명 추출 (첫 번째 언더스코어까지)
            road_name = folder_name.split('_')[0]
            
            # 계절 추출
            seasons = ['봄', '여름', '가을', '겨울']
            season = next((s for s in seasons if s in folder_name), '')
            
            # 방향 코드 가져오기
            direction = get_direction_code(folder_name)
            
            # 새 폴더명 생성
            if direction:
                new_name = f"{road_name}_{direction}_{season}_배출량"
            else:
                new_name = f"{road_name}_{season}_배출량"
            
            # 새 폴더 전체 경로
            new_path = os.path.join(root_path, new_name)
            
            try:
                # 폴더명 변경
                os.rename(old_path, new_path)
                renamed_folders.append({
                    'old_name': folder_name,
                    'new_name': new_name
                })
                print(f"변경 완료: {folder_name} → {new_name}")
            except Exception as e:
                print(f"오류 발생: {folder_name} 변경 실패 - {str(e)}")
    
    return renamed_folders

if __name__ == "__main__":
    # 대상 폴더 경로 설정
    target_path = r"C:\dense_traffic_emi\output"
    
    # 실행 전 확인
    print(f"다음 경로의 폴더명을 변경합니다: {target_path}")
    confirm = input("계속하시겠습니까? (y/n): ")
    
    if confirm.lower() == 'y':
        # 폴더명 변경 실행
        renamed = rename_folders(target_path)
        
        # 변경 결과 출력
        print("\n=== 변경된 폴더 목록 ===")
        for item in renamed:
            print(f"이전: {item['old_name']}")
            print(f"이후: {item['new_name']}")
            print("-" * 50)
    else:
        print("작업이 취소되었습니다.")