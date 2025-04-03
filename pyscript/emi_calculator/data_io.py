import pandas as pd
import os

def load_input_data(file_path):
    """
    사용자가 지정한 input 파일(Excel)을 읽어와서 DataFrame으로 반환
    """
    return pd.read_excel(file_path)

def make_output_directory(base_dir, input_file):
    """
    input_file 이름(확장자 제외)에 _emi를 붙여서 디렉토리를 만든 뒤 경로 반환
    """
    output_dir_name = os.path.splitext(input_file)[0]
    output_dir = os.path.join(base_dir, f'{output_dir_name}_emi')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return output_dir