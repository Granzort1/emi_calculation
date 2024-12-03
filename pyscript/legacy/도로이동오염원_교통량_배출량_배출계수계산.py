import pandas as pd
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
from math import exp


@dataclass
class VehicleInfo:
    category: str
    scc: str
    fuel: str


def normalize_formula(formula: str) -> str:
    """수식을 파이썬에서 계산 가능한 형태로 정규화"""
    formula = formula.replace('^', '**')
    formula = formula.replace('×', '*')
    formula = formula.replace('E-', 'e-')
    formula = formula.replace('E+', 'e+')
    formula = formula.replace('Exp', 'exp')
    return formula


def parse_year_range(year_text: str) -> tuple:
    """연도 범위 파싱"""
    if '이전' in year_text:
        end_year = int('19' + year_text.split('년')[0].strip("'"))
        return 1900, end_year
    elif '이후' in year_text:
        start_year = int('20' + year_text.split('년')[0].strip("'"))
        return start_year, 2050
    elif '~' in year_text:
        start, end = year_text.split('~')
        if '.' in start:  # '15.9월~'17년' 형태 처리
            start_year = int('20' + start.split('.')[0].strip("'"))
            end_year = int('20' + end.split('년')[0].strip("'`"))
        else:
            start_year = int('20' + start.strip("'년"))
            end_year = int('20' + end.strip("'년"))
        return start_year, end_year
    else:
        year = int('20' + year_text.strip("'년"))
        return year, year


def parse_speed_condition(condition: str) -> tuple:
    """속도 조건 파싱"""
    if pd.isna(condition):
        return None, None

    if '≦' in condition or '<' in condition:
        speed = float(condition.split('km/h')[0].split('V')[-1].strip('≦<'))
        return None, speed
    elif '>' in condition or '≧' in condition:
        speed = float(condition.split('km/h')[0].split('V')[-1].strip('>≧'))
        return speed, None

    return None, None


def create_record(vehicle: VehicleInfo, pollutant: str, year_range: str,
                  speed_cond: str, temp_cond: str, formula: str) -> Dict[str, Any]:
    """단일 배출계수 레코드 생성"""
    year_start, year_end = parse_year_range(year_range)
    speed_min, speed_max = parse_speed_condition(speed_cond)

    return {
        'Category': vehicle.category,
        'SCC': vehicle.scc,
        'Fuel': vehicle.fuel,
        'Pollutant': pollutant,
        'Year_Start': year_start,
        'Year_End': year_end,
        'Temperature_Condition': temp_cond,
        'Speed_Min': speed_min,
        'Speed_Max': speed_max,
        'Formula': normalize_formula(formula),
        'Parameters': 'V = 속도(km/h)',
        'Notes': '배출계수 단위: g/km'
    }


def get_gasoline_light_passenger_data() -> List[Dict[str, Any]]:
    """휘발유 승용차 경형 데이터"""
    records = []
    vehicle = VehicleInfo('승용차 경형', '07010101', '휘발유')

    # CO 데이터
    co_data = [
        ("'99년 이전", 'V≦65km/h', None, '59.783×V^(-1.0007)'),
        ("'99년 이전", 'V>65km/h', None, '0.0874×V-3.5618'),
        ("'00~'05년", 'V≦65km/h', None, '60.556×V^(-1.2501)'),
        ("'00~'05년", 'V>65km/h', None, '-0.0006×V+0.5753'),
        ("'06~'08년", 'V≦45km/h', None, '4.9952×V^(-0.8461)'),
        ("'06~'08년", 'V>45km/h', None, '-0.0001×V^2+0.0229×V-0.5701'),
        ("'09~'11년", 'V≦45km/h', None, '4.5956×V^(-0.8461)'),
        ("'09~'11년", 'V>45km/h', None, '-9.2000E-05×V^2+2.1068E-02×V-5.2449E-01'),
        ("'12~'13년", 'V≦45km/h', None, '4.4517×V^(-0.8461)'),
        ("'12~'13년", 'V>45km/h', None, '-8.9120E-05×V^2+2.0408E-02×V-5.0807E-01'),
        ("'14년", 'V≦45km/h', None, '4.3079×V^(-0.8461)'),
        ("'14년", 'V>45km/h', None, '-8.6240E-05×V^2+1.9749E-02×V-4.9165E-01'),
        ("'15년", 'V≦45km/h', None, '4.164×V^(-0.8461)'),
        ("'15년", 'V>45km/h', None, '-8.3360E-05×V^2+1.9089E-02×V-4.7524E-01'),
        ("'16년", None, None, '2.9500E-05×V^2-3.2721E-03×V+2.2115E-01'),
        ("'17년", None, None, '2.9500E-05×V^2-3.2721E-03×V+2.2115E-01'),
        ("'18년 이후", None, None, '2.9500E-05×V^2-3.2721E-03×V+2.2115E-01')
    ]

    for year_range, speed_cond, temp_cond, formula in co_data:
        records.append(create_record(vehicle, 'CO', year_range, speed_cond, temp_cond, formula))

    # VOC 데이터
    voc_data = [
        ("'96년 이전", None, None, '7.6244×V^(-0.8364)'),
        ("'97~'99년", None, None, '8.6275×V^(-1.0722)'),
        ("'00~'02년", None, None, '5.1835×V^(-1.1889)'),
        ("'03~'05년", None, None, '0.7446×V^(-0.9392)'),
        ("'06~'08년", None, None, '0.2958×V^(-0.7830)'),
        ("'09~'11년", None, None, '0.2662×V^(-0.7830)'),
        ("'12~'13년", None, None, '0.2556×V^(-0.7830)'),
        ("'14년", None, None, '0.2449×V^(-0.7830)'),
        ("'15년", None, None, '0.2343×V^(-0.7830)'),
        ("'16년", 'V≦65.4km/h', None, '5.7153E-02×V^(-1.0656E+00)'),
        ("'16년", 'V>65.4km/h', None, '1.4308E-06×V^2-1.3307E-04×V+3.2987E-03'),
        ("'17년", 'V≦65.4km/h', None, '6.2620E-02×V^(-1.1016E+00)'),
        ("'17년", 'V>65.4km/h', None, '1.3125E-06×V^2-1.2579E-04×V+3.3016E-03'),
        ("'18년 이후", 'V≦65.4km/h', None, '6.7287E-02×V^(-1.1300E+00)'),
        ("'18년 이후", 'V>65.4km/h', None, '1.2232E-06×V^2-1.2029E-04×V+3.3037E-03')
    ]

    for year_range, speed_cond, temp_cond, formula in voc_data:
        records.append(create_record(vehicle, 'VOC', year_range, speed_cond, temp_cond, formula))

    # NOx 데이터
    nox_data = [
        ("'96년 이전", None, None, '2.6754×V^(-0.3236)'),
        ("'97~'99년", None, None, '3.2294×V^(-0.5763)'),
        ("'00~'02년", None, None, '1.7525×V^(-0.6481)'),
        ("'03~'05년", None, None, '0.3403×V^(-0.5455)'),
        ("'06~'08년", None, None, '0.4819×V^(-0.9198)'),
        ("'09~'11년", None, None, '0.4476×V^(-0.9198)'),
        ("'12~'13년", None, None, '0.4353×V^(-0.9198)'),
        ("'14년", None, None, '0.4230×V^(-0.9198)'),
        ("'15년", None, None, '0.4106×V^(-0.9198)'),
        ("'16년", None, None, '5.5718E-07×V^2-1.4999E-04×V+1.3699E-02'),
        ("'17년", None, None, '5.7226E-07×V^2-1.4891E-04×V+1.3178E-02'),
        ("'18년 이후", None, None, '5.8363E-07×V^2-1.4810E-04×V+1.2786E-02')
    ]

    for year_range, speed_cond, temp_cond, formula in nox_data:
        records.append(create_record(vehicle, 'NOx', year_range, speed_cond, temp_cond, formula))

    # PM-10 데이터 (MPI)
    records.append(create_record(vehicle, 'MPI-PM-10', '-', 'V<85km/h', None, '0.00030'))
    records.append(create_record(vehicle, 'MPI-PM-10', '-', 'V≧85km/h', None, '0.00075'))

    # PM-10 데이터 (GDI)
    records.append(create_record(vehicle, 'GDI-PM-10', "'10년 이후", 'V<85km/h', None, '0.0010'))
    records.append(create_record(vehicle, 'GDI-PM-10', "'10년 이후", 'V≧85km/h', None, '0.0025'))

    # PM-2.5 데이터 (MPI)
    records.append(create_record(vehicle, 'MPI-PM-2.5', '-', 'V<85km/h', None, 'k×0.00030'))
    records.append(create_record(vehicle, 'MPI-PM-2.5', '-', 'V≧85km/h', None, 'k×0.00075'))

    # PM-2.5 데이터 (GDI)
    records.append(create_record(vehicle, 'GDI-PM-2.5', "'10년 이후", 'V<85km/h', None, 'k×0.0010'))
    records.append(create_record(vehicle, 'GDI-PM-2.5', "'10년 이후", 'V≧85km/h', None, 'k×0.0025'))

    return records


def get_gasoline_small_passenger_data() -> List[Dict[str, Any]]:
    """휘발유 승용차 소형 데이터"""
    records = []
    vehicle = VehicleInfo('승용차 소형', '07010201', '휘발유')

    # CO 데이터
    co_data = [
        ("'86년 이전", None, None, '247.00×V^(-0.6651)'),
        ("'87~'90년", None, None, '36.169×V^(-0.7587)'),
        ("'91~'99년", None, None, '111.67×V^(-1.1566)'),
        ("'00~'02년", None, None, '22.356×V^(-0.9068)'),
        ("'03~'05년", None, None, '1.4898×V^(-0.3837)'),
        ("'06~'08년", None, None, '1.0000E-04×V^2-7.1000E-03×V+2.2450E-01'),
        ("'09~'11년", None, None, '9.2000E-05×V^2-6.5320E-03×V+2.0654E-01'),
        ("'12~'13년", None, None, '8.9120E-05×V^2-6.3275E-03×V+2.0007E-01'),
        ("'14년", None, None, '8.6240E-05×V^2-6.1230E-03×V+1.9361E-01'),
        ("'15년", None, None, '8.3360E-05×V^2-5.9186E-03×V+1.8714E-01'),
        ("'16년", None, None, '2.9500E-05×V^2-3.2721E-03×V+2.2115E-01'),
        ("'17년", None, None, '2.9500E-05×V^2-3.2721E-03×V+2.2115E-01'),
        ("'18년 이후", None, None, '2.9500E-05×V^2-3.2721E-03×V+2.2115E-01')
    ]

    for year_range, speed_cond, temp_cond, formula in co_data:
        records.append(create_record(vehicle, 'CO', year_range, speed_cond, temp_cond, formula))

# VOC 데이터
    voc_data = [
        ("'86년 이전", None, None, '15.953×V^(-0.5059)'),
        ("'87~'90년", None, None, '15.607×V^(-1.0423)'),
        ("'91~'99년", None, None, '32.017×V^(-1.4171)'),
        ("'00~'02년", None, None, '0.8428×V^(-0.8829)'),
        ("'03~'05년", None, None, '0.1738×V^(-0.7268)'),
        ("'06~'08년", 'V≦65.4km/h', None, '0.0633×V^(-1.0484)'),
        ("'06~'08년", 'V>65.4km/h', None, '1.3200E-06×V^2-1.8800E-04×V+7.7000E-03'),
        ("'09~'11년", 'V≦65.4km/h', None, '0.0570×V^(-1.0484)'),
        ("'09~'11년", 'V>65.4km/h', None, '1.1880E-06×V^2-1.6920E-04×V+6.9300E-03'),
        ("'12~'13년", 'V≦65.4km/h', None, '0.0547×V^(-1.0484)'),
        ("'12~'13년", 'V>65.4km/h', None, '1.1405E-06×V^2-1.6243E-04×V+6.6528E-03'),
        ("'14년", 'V≦65.4km/h', None, '0.0524×V^(-1.0484)'),
        ("'14년", 'V>65.4km/h', None, '1.0930E-06×V^2-1.5566E-04×V+6.3756E-03'),
        ("'15년", 'V≦65.4km/h', None, '0.0501×V^(-1.0484)'),
        ("'15년", 'V>65.4km/h', None, '1.0500E-06×V^2-1.4890E-04×V+6.09840E-03'),
        ("'16년", 'V≦65.4km/h', None, '5.7153E-02×V^-1.0656E+00'),
        ("'16년", 'V>65.4km/h', None, '1.4308E-06×V^2-1.3307E-04×V+3.2987E-03'),
        ("'17년", 'V≦65.4km/h', None, '6.2620E-02×V^-1.1016E+00'),
        ("'17년", 'V>65.4km/h', None, '1.3125E-06×V^2-1.2579E-04×V+3.3016E-03'),
        ("'18년 이후", 'V≦65.4km/h', None, '6.7287E-02×V^-1.1300E+00'),
        ("'18년 이후", 'V>65.4km/h', None, '1.2232E-06×V^2-1.2029E-04×V+3.3037E-03')
    ]

    for year_range, speed_cond, temp_cond, formula in voc_data:
        records.append(create_record(vehicle, 'VOC', year_range, speed_cond, temp_cond, formula))

    # NOx 데이터
    nox_data = [
        ("'86년 이전", None, None, '3.1140×V^(-0.2278)'),
        ("'87~'90년", None, None, '6.2007×V^(-0.6781)'),
        ("'91~'99년", None, None, '7.5244×V^(-0.7634)'),
        ("'00~'02년", None, None, '1.2613×V^(-0.3873)'),
        ("'03~'05년", None, None, '0.1563×V^(-0.2671)'),
        ("'06~'08년", None, None, '-3.5000E-06×V^2+3.3000E-04×V+1.1200E-02'),
        ("'09~'11년", None, None, '-3.2511E-06×V^2+3.0653E-04×V+1.0404E-02'),
        ("'12~'13년", None, None, '-3.1615E-06×V^2+2.9809E-04×V+1.0117E-02'),
        ("'14년", None, None, '-3.0719E-06×V^2+2.8964E-04×V+9.8301E-03'),
        ("'15년", None, None, '-2.9823E-06×V^2+2.8119E-04×V+9.5434E-03'),
        ("'16년", None, None, '5.5718E-07×V^2-1.4999E-04×V+1.3699E-02'),
        ("'17년", None, None, '5.7226E-07×V^2-1.4891E-04×V+1.3178E-02'),
        ("'18년 이후", None, None, '5.8363E-07×V^2-1.4810E-04×V+1.2786E-02')
    ]

    for year_range, speed_cond, temp_cond, formula in nox_data:
        records.append(create_record(vehicle, 'NOx', year_range, speed_cond, temp_cond, formula))

    # PM 데이터 추가
    pm_data = [
        ('MPI-PM-10', '-', 'V<85km/h', '0.00030'),
        ('MPI-PM-10', '-', 'V≧85km/h', '0.00075'),
        ('GDI-PM-10', "'10년 이후", 'V<85km/h', '0.0010'),
        ('GDI-PM-10', "'10년 이후", 'V≧85km/h', '0.0025'),
        ('MPI-PM-2.5', '-', 'V<85km/h', 'k×0.00030'),
        ('MPI-PM-2.5', '-', 'V≧85km/h', 'k×0.00075'),
        ('GDI-PM-2.5', "'10년 이후", 'V<85km/h', 'k×0.0010'),
        ('GDI-PM-2.5', "'10년 이후", 'V≧85km/h', 'k×0.0025')
    ]

    for pollutant, year_range, speed_cond, formula in pm_data:
        records.append(create_record(vehicle, pollutant, year_range, speed_cond, None, formula))

    return records

def get_gasoline_medium_passenger_data() -> List[Dict[str, Any]]:
    """휘발유 승용차 중형 데이터"""
    records = []
    vehicle = VehicleInfo('승용차 중형', '07010301', '휘발유')

    # CO 데이터
    co_data = [
        ("'86년 이전", None, None, '247.00×V^(-0.6651)'),
        ("'87~'90년", None, None, '36.169×V^(-0.7587)'),
        ("'91~'99년", None, None, '51.555×V^(-0.9531)'),
        ("'00~'02년", None, None, '29.921×V^(-0.8868)'),
        ("'03~'05년", None, None, '2.4938×V^(-0.6106)'),
        ("'06~'08년", None, None, '2.2900E-05×V^2-1.6300E-03×V+5.8300E-02'),
        ("'09~'11년", None, None, '2.1068E-05×V^2-1.4996E-03×V+5.3636E-02'),
        ("'12~'13년", None, None, '2.0408E-05×V^2-1.4527E-03×V+5.1957E-02'),
        ("'14년", None, None, '1.9749E-05×V^2-1.4057E-03×V+5.0278E-02'),
        ("'15년", None, None, '1.9089E-05×V^2-1.3588E-03×V+4.8599E-02'),
        ("'16년", None, None, '2.9500E-05×V^2-3.2721E-03×V+2.2115E-01'),
        ("'17년", None, None, '2.9500E-05×V^2-3.2721E-03×V+2.2115E-01'),
        ("'18년 이후", None, None, '2.9500E-05×V^2-3.2721E-03×V+2.2115E-01')
    ]

    for year_range, speed_cond, temp_cond, formula in co_data:
        records.append(create_record(vehicle, 'CO', year_range, speed_cond, temp_cond, formula))

    # VOC 데이터
    voc_data = [
        ("'86년 이전", None, None, '15.953×V^(-0.5059)'),
        ("'87~'90년", None, None, '15.607×V^(-1.0423)'),
        ("'91~'99년", None, None, '31.816×V^(-1.4804)'),
        ("'00~'02년", None, None, '7.9374×V^(-1.3041)'),
        ("'03~'05년", None, None, '0.4262×V^(-1.0122)'),
        ("'06~'08년", 'V≦65.4km/h', None, '0.0633×V^(-1.0484)'),
        ("'06~'08년", 'V>65.4km/h', None, '1.3200E-06×V^2-1.8800E-04×V+7.7000E-03'),
        ("'09~'11년", 'V≦65.4km/h', None, '0.0570×V^(-1.0484)'),
        ("'09~'11년", 'V>65.4km/h', None, '1.1880E-06×V^2-1.6920E-04×V+6.9300E-03'),
        ("'12~'13년", 'V≦65.4km/h', None, '0.0547×V^(-1.0484)'),
        ("'12~'13년", 'V>65.4km/h', None, '1.1405E-06×V^2-1.6243E-04×V+6.6528E-03'),
        ("'14년", 'V≦65.4km/h', None, '0.0524×V^(-1.0484)'),
        ("'14년", 'V>65.4km/h', None, '1.0930E-06×V^2-1.5566E-04×V+6.3756E-03'),
        ("'15년", 'V≦65.4km/h', None, '0.0501×V^(-1.0484)'),
        ("'15년", 'V>65.4km/h', None, '1.0500E-06×V^2-1.4890E-04×V+6.0984E-03'),
        ("'16년", 'V≦65.4km/h', None, '5.7153E-02×V^-1.0656E+00'),
        ("'16년", 'V>65.4km/h', None, '1.4308E-06×V^2-1.3307E-04×V+3.2987E-03'),
        ("'17년", 'V≦65.4km/h', None, '6.2620E-02×V^-1.1016E+00'),
        ("'17년", 'V>65.4km/h', None, '1.3125E-06×V^2-1.2579E-04×V+3.3016E-03'),
        ("'18년 이후", 'V≦65.4km/h', None, '6.7287E-02×V^-1.1300E+00'),
        ("'18년 이후", 'V>65.4km/h', None, '1.2232E-06×V^2-1.2029E-04×V+3.3037E-03')
    ]

    for year_range, speed_cond, temp_cond, formula in voc_data:
        records.append(create_record(vehicle, 'VOC', year_range, speed_cond, temp_cond, formula))

    # NOx 데이터
    nox_data = [
        ("'86년 이전", None, None, '3.1140×V^(-0.2278)'),
        ("'87~'90년", None, None, '6.2007×V^(-0.6781)')
    ]

    for year_range, speed_cond, temp_cond, formula in nox_data:
        records.append(create_record(vehicle, 'NOx', year_range, speed_cond, temp_cond, formula))

    return records


def get_diesel_light_passenger_data() -> List[Dict[str, Any]]:
    """경유 승용차 경형 데이터"""
    records = []
    vehicle = VehicleInfo('승용차 경형', '07010101', '경유')

    # CO 데이터
    co_data = [
        ("'05년 이전", None, None, '0.7392×V^(-0.7524)'),
        ("'06~'10년", None, None, '0.5775×V^(-0.7524)'),
        ("'11년~'15.8월", None, None, '0.5141×V^(-0.6792)'),
        ("'15.9월~'17년", None, None, '0.4574×V^(-0.5215)'),
        ("'18년 이후", None, None, '4.5878E-01×V^(-5.6934E-01)')
    ]

    for year_range, speed_cond, temp_cond, formula in co_data:
        records.append(create_record(vehicle, 'CO', year_range, speed_cond, temp_cond, formula))

    # VOC 데이터
    voc_data = [
        ("'05년 이전", None, None, '0.0989×V^(-0.6848)'),
        ("'06~'10년", None, None, '0.0825×V^(-0.6848)'),
        ("'11년~'15.8월", None, None, '0.3713×V^(-0.7513)'),
        ("'15.9월 이후", None, None, '0.1300×V^(-0.7265)')
    ]

    for year_range, speed_cond, temp_cond, formula in voc_data:
        records.append(create_record(vehicle, 'VOC', year_range, speed_cond, temp_cond, formula))

    # NOx 데이터 (온도 조건 포함)
    nox_data = [
        ("'05년 이전", None, "외기온도 20℃이상 (에어컨 가동)", '32.4104×V^(-0.6377)'),
        ("'05년 이전", None, "외기온도 10~20℃ (standard)", '24.3491×V^(-0.7277)'),
        ("'05년 이전", None, "외기온도 0~10℃", '17.2988×V^(-0.5818)'),
        ("'05년 이전", None, "외기온도 0℃미만", '12.4051×V^(-0.4960)'),

        ("'06~'10년", None, "외기온도 20℃이상 (에어컨 가동)", '32.4104×V^(-0.6377)'),
        ("'06~'10년", None, "외기온도 10~20℃ (standard)", '24.3491×V^(-0.7277)'),
        ("'06~'10년", None, "외기온도 0~10℃", '17.2988×V^(-0.5818)'),
        ("'06~'10년", None, "외기온도 0℃미만", '12.4051×V^(-0.4960)'),

        ("'11년~'15.8월", None, "외기온도 20℃이상 (에어컨 가동)", '0.0003×V^2-0.0324×V+1.8035'),
        ("'11년~'15.8월", None, "외기온도 10~20℃ (standard)", '0.0003×V^2-0.0324×V+1.4773'),
        ("'11년~'15.8월", None, "외기온도 10℃ 미만", '0.0003×V^2-0.0324×V+2.0031'),

        ("'15.9월~'17년", None, "외기온도 20℃이상 (에어컨 가동)", '2.7144×V^(-0.3437)'),
        ("'15.9월~'17년", None, "외기온도 10~20℃ (standard)", '2.7702×V^(-0.3869)'),
        ("'15.9월~'17년", None, "외기온도 10℃ 미만", '2.7241×V^(-0.2743)'),

        ("'18년 이후", None, None, '1.0031E-01×Exp(-3.0196E-02×V)')
    ]

    for year_range, speed_cond, temp_cond, formula in nox_data:
        records.append(create_record(vehicle, 'NOx', year_range, speed_cond, temp_cond, formula))

    # PM-10 데이터
    pm10_data = [
        ("'05년 이전", None, None, '0.0839×V^(-0.3420)'),
        ("'06~'10년", None, None, '0.0420×V^(-0.3420)'),
        ("'11년~'15.8월", 'V≦65.4km/h', None, '0.0225×V^(-0.7264)'),
        ("'11년~'15.8월", 'V>65.4km/h', None, '0.0009×V^(0.0416)'),
        ("'15.9월 이후", 'V≦65.4km/h', None, '0.0225×V^(-0.7264)'),
        ("'15.9월 이후", 'V>65.4km/h', None, '0.0009×V^(0.0416)')
    ]

    for year_range, speed_cond, temp_cond, formula in pm10_data:
        records.append(create_record(vehicle, 'PM-10', year_range, speed_cond, temp_cond, formula))

    # PM-2.5 데이터
    pm25_data = [
        ("'05년 이전", None, None, 'k×0.0839×V^(-0.3420)'),
        ("'06~'10년", None, None, 'k×0.0420×V^(-0.3420)'),
        ("'11년~'15.8월", 'V≦65.4km/h', None, 'k×0.0225×V^(-0.7264)'),
        ("'11년~'15.8월", 'V>65.4km/h', None, 'k×0.0009×V^(0.0416)'),
        ("'15.9월 이후", 'V≦65.4km/h', None, 'k×0.0225×V^(-0.7264)'),
        ("'15.9월 이후", 'V>65.4km/h', None, 'k×0.0009×V^(0.0416)')
    ]

    for year_range, speed_cond, temp_cond, formula in pm25_data:
        records.append(create_record(vehicle, 'PM-2.5', year_range, speed_cond, temp_cond, formula))

    return records

def get_lpg_passenger_data() -> List[Dict[str, Any]]:
    """LPG 승용차 데이터"""
    records = []
    vehicle = VehicleInfo('승용차', '07010101', 'LPG')

    # CO 데이터
    co_data = [
        ("'96년 이전", 'V≦45km/h', None, '22.498×V^(-0.6579)'),
        ("'96년 이전", 'V>45km/h', None, '0.0006×V^2+0.0004×V+0.8272'),
        ("'97~'05년", 'V≦45km/h', None, '19.887×V^(-0.8461)'),
        ("'97~'05년", 'V>45km/h', None, '-0.0004×V^2+0.0911×V-2.2698'),
        ("'06~'07년", 'V≦45km/h', None, '8.9904×V^(-0.8461)'),
        ("'06~'07년", 'V>45km/h', None, '-0.0002×V^2+0.0457×V-1.1403'),
        ("'08년", 'V≦79.6km/h', None, '0.7693×V^(-0.7666)'),
        ("'08년", 'V>79.6km/h', None, '5.0000E-16×V^(7.2766)'),
        ("'09~'11년", 'V≦79.6km/h', None, '0.7059×V^(-0.7666)'),
        ("'09~'11년", 'V>79.6km/h', None, '4.5878E-16×V^(7.2766)'),
        ("'12~'13년", 'V≦79.6km/h', None, '0.6830×V^(-0.7666)'),
        ("'12~'13년", 'V>79.6km/h', None, '4.4393E-16×V^(7.2766)'),
        ("'14년", 'V≦79.6km/h', None, '0.6602×V^(-0.7666)'),
        ("'14년", 'V>79.6km/h', None, '4.2909E-16×V^(7.2766)'),
        ("'15년", 'V≦79.6km/h', None, '0.6374×V^(-0.7666)'),
        ("'15년", 'V>79.6km/h', None, '4.1425E-16×V^(7.2766)'),
        ("'16년", None, None, '2.9500E-05×V^2-3.2721E-03×V+2.2115E-01'),
        ("'17년", None, None, '2.9500E-05×V^2-3.2721E-03×V+2.2115E-01'),
        ("'18년 이후", None, None, '2.9500E-05×V^2-3.2721E-03×V+2.2115E-01')
    ]

    for year_range, speed_cond, temp_cond, formula in co_data:
        records.append(create_record(vehicle, 'CO', year_range, speed_cond, temp_cond, formula))

    # VOC 데이터
    voc_data = [
        ("'96년 이전", None, None, '12.961×V^(-0.8364)'),
        ("'97~'02년", None, None, '2.2714×V^(-0.7830)'),
        ("'03~'05년", None, None, '1.1073×V^(-0.7830)'),
        ("'06~'07년", None, None, '0.3549×V^(-0.7830)'),
        ("'08년", 'V≦79.6km/h', None, '0.1063×V^(-1.0745)'),
        ("'08년", 'V>79.6km/h', None, '1.0000E-15×V^(6.2696)')
    ]

    for year_range, speed_cond, temp_cond, formula in voc_data:
        records.append(create_record(vehicle, 'VOC', year_range, speed_cond, temp_cond, formula))

    # NOx 데이터
    nox_data = [
        ("'96년 이전", None, None, '4.0131×V^(-0.3236)'),
        ("'97~'99년", None, None, '1.8528×V^(-0.3889)'),
        ("'00~'05년", None, None, '5.8289×V^(-0.9198)'),
        ("'06~'07년", None, None, '0.7228×V^(-0.9198)'),
        ("'08년", None, None, '-4.0000E-06×V^2+6.0000E-04×V+5.5000E-03'),
        ("'09~'11년", None, None, '-3.7333E-06×V^2+5.6000E-04×V+5.1333E-03')
    ]

    for year_range, speed_cond, temp_cond, formula in nox_data:
        records.append(create_record(vehicle, 'NOx', year_range, speed_cond, temp_cond, formula))

    # PM 데이터
    pm_data = [
        ('PM-10', '-', 'V<85km/h', '0.0002'),
        ('PM-10', '-', 'V≧85km/h', '0.0005'),
        ('PM-2.5', '-', 'V<85km/h', 'k×0.0002'),
        ('PM-2.5', '-', 'V≧85km/h', 'k×0.0005')
    ]

    for pollutant, year_range, speed_cond, formula in pm_data:
        records.append(create_record(vehicle, pollutant, year_range, speed_cond, None, formula))

    return records


def get_cng_data() -> List[Dict[str, Any]]:
    """CNG 버스 데이터"""
    records = []
    vehicle = VehicleInfo('버스 시내', '07040101', 'CNG')

    # CNG 데이터
    cng_data = [
        # CO 데이터
        ('CO', "'05년 이전", None, None, '18.235×V^(-0.3767)'),
        ('CO', "'06년~'10.6월", None, None, '2.4653×V^(-1.1470)'),
        ('CO', "'10.7월~'13년", None, None, '0.7403×V^(-0.6307)'),
        ('CO', "'14년 이후", None, None, '0.935'),

        # VOC 데이터
        ('VOC', "'05년 이전", None, None, '8.0544×EXP(-0.0174×V)'),
        ('VOC', "'06년~'10.6월", None, None, '70.5483×V^(-0.8432)'),
        ('VOC', "'10.7월~'13년", None, None, '74.1362×V^(-0.7291)'),
        ('VOC', "'14년 이후", None, None, '0.0004×V^2-0.0563×V+2.4117'),

        # NOx 데이터
        ('NOx', "'05년 이전", None, None, '8.6972×EXP(-0.0130×V)'),
        ('NOx', "'06년~'10.6월", None, None, '33.1287×V^(-0.5966)'),
        ('NOx', "'10.7월~'13년", None, None, '14.9841×V^(-0.4592)'),
        ('NOx', "'14년 이후", None, None, '0.0003×V^2-0.041×V+1.4756'),

        # PM-10 데이터
        ('PM-10', '-', '≦55 km/h', None, '0.0038'),
        ('PM-10', '-', '>55 km/h', None, '0.0011'),

        # PM-2.5 데이터
        ('PM-2.5', '-', '≦55 km/h', None, 'k×0.0038'),
        ('PM-2.5', '-', '>55 km/h', None, 'k×0.0011')
    ]

    for pollutant, year_range, speed_cond, temp_cond, formula in cng_data:
        records.append(create_record(vehicle, pollutant, year_range, speed_cond, temp_cond, formula))

    return records


def get_hybrid_data() -> List[Dict[str, Any]]:
    """하이브리드 차량 데이터"""
    records = []
    vehicle = VehicleInfo('승용차', '07010101', '하이브리드')

    hybrid_data = [
        ('CO', '-', None, None, '3.8807E-5×V^2-4.5279E-3×V+1.6196E-1'),
        ('VOC', '-', None, None, '1.0124E-2×V^(-1.0584)'),
        ('NOx', '-', None, None, '2.4521E-2×Exp(-0.0221×V)'),
        ('PM-10', '-', 'V≤85 km/h', None, 'r×0.0010'),
        ('PM-10', '-', 'V>85 km/h', None, 'r×0.0025'),
        ('PM-2.5', '-', 'V≤85 km/h', None, 'k×r×0.0010'),
        ('PM-2.5', '-', 'V>85 km/h', None, 'k×r×0.0025')
    ]

    for pollutant, year_range, speed_cond, temp_cond, formula in hybrid_data:
        records.append(create_record(vehicle, pollutant, year_range, speed_cond, temp_cond, formula))

    return records


def create_full_database():
    """전체 데이터베이스 생성"""
    all_records = []

    # 휘발유 차량 데이터
    all_records.extend(get_gasoline_light_passenger_data())
    all_records.extend(get_gasoline_small_passenger_data())
    all_records.extend(get_gasoline_medium_passenger_data())

    # 경유 차량 데이터
    all_records.extend(get_diesel_light_passenger_data())

    # LPG 차량 데이터
    all_records.extend(get_lpg_passenger_data())

    # CNG 차량 데이터
    all_records.extend(get_cng_data())

    # 하이브리드 차량 데이터
    all_records.extend(get_hybrid_data())

    # DataFrame 생성
    df = pd.DataFrame(all_records)

    # 컬럼 순서 정리
    columns = [
        'Category', 'SCC', 'Fuel', 'Pollutant',
        'Year_Start', 'Year_End', 'Temperature_Condition',
        'Speed_Min', 'Speed_Max', 'Formula',
        'Parameters', 'Notes'
    ]

    return df[columns]


# 메인 실행 코드
if __name__ == "__main__":
    print("배출계수 데이터베이스 생성 시작...")

    try:
        # 데이터베이스 생성
        df = create_full_database()

        # 데이터 검증
        print("\n데이터 검증 중...")
        print(f"총 레코드 수: {len(df)}")
        print(f"차종 수: {df['Category'].nunique()}")
        print(f"연료 유형 수: {df['Fuel'].nunique()}")
        print(f"오염물질 수: {df['Pollutant'].nunique()}")

        # 엑셀 파일로 저장
        output_file = 'emission_factors_database.xlsx'
        df.to_excel(output_file, index=False)
        print(f"\n데이터베이스가 성공적으로 생성되었습니다: {output_file}")

        # CSV 백업 파일 생성
        backup_file = 'emission_factors_database_backup.csv'
        df.to_csv(backup_file, index=False)
        print(f"백업 파일이 생성되었습니다: {backup_file}")

    except Exception as e:
        print(f"오류 발생: {str(e)}")


# 사용 예시
def calculate_emission(vehicle_type: str, fuel: str, pollutant: str,
                       year: int, speed: float, temp_condition: str = None) -> float:
    """
    주어진 조건에 맞는 배출계수 계산

    Parameters:
    -----------
    vehicle_type : str
        차종 (예: '승용차 경형')
    fuel : str
        연료 종류 (예: '휘발유')
    pollutant : str
        오염물질 (예: 'CO')
    year : int
        연식
    speed : float
        속도 (km/h)
    temp_condition : str, optional
        온도 조건

    Returns:
    --------
    float : 계산된 배출계수 (g/km)
    """

    # 엑셀 파일에서 데이터 로드
    df = pd.read_excel('emission_factors_database.xlsx')

    # 기본 조건으로 필터링
    mask = (df['Category'] == vehicle_type) & \
           (df['Fuel'] == fuel) & \
           (df['Pollutant'] == pollutant) & \
           (df['Year_Start'] <= year) & \
           ((df['Year_End'] >= year) | (df['Year_End'].isna()))

    if temp_condition:
        mask &= (df['Temperature_Condition'] == temp_condition)

    filtered_df = df[mask]

    for _, row in filtered_df.iterrows():
        # 속도 조건 확인
        speed_min = row['Speed_Min'] if pd.notna(row['Speed_Min']) else float('-inf')
        speed_max = row['Speed_Max'] if pd.notna(row['Speed_Max']) else float('inf')

        if speed_min <= speed <= speed_max:
            # 수식에서 변수 대체
            formula = row['Formula']
            formula = formula.replace('V', str(speed))
            formula = formula.replace('k', '0.92')  # PM-2.5 계산용 상수
            formula = formula.replace('r', '0.59')  # 하이브리드용 상수

            try:
                # 수식 계산
                result = eval(formula)
                return result
            except Exception as e:
                print(f"수식 계산 오류: {formula}")
                print(f"오류 메시지: {str(e)}")
                return None

    return None


# 사용 예시
if __name__ == "__main__":
    # 예시: 2020년식 휘발유 경형 승용차의 CO 배출계수 계산 (속도 60km/h)
    result = calculate_emission(
        vehicle_type='승용차 경형',
        fuel='휘발유',
        pollutant='CO',
        year=2020,
        speed=60
    )

    if result is not None:
        print(f"\n계산 예시:")
        print(f"2020년식 휘발유 경형 승용차")
        print(f"속도: 60 km/h")
        print(f"CO 배출계수: {result:.4f} g/km")














