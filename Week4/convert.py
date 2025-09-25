import xml.etree.ElementTree as ET
import json
import os

def convert_xml_to_full_json(xml_path):
    """
    XML 어노테이션 파일의 모든 정보를 포함하여 JSON으로 변환하고,
    'suspicious' event 항목을 추가합니다.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except (FileNotFoundError, ET.ParseError) as e:
        print(f"오류: XML 파일을 읽을 수 없습니다 - {e}")
        return None

    # --- XML 태그를 순서대로 읽어 딕셔너리에 추가 ---
    data = {}

    # 1. 최상위 기본 정보
    data['folder'] = root.findtext('folder')
    data['filename'] = root.findtext('filename')

    # 2. <source> 정보
    source_node = root.find('source')
    data['source'] = {
        "database": source_node.findtext('database'),
        "annotation": source_node.findtext('annotation')
    }

    # 3. <size> 정보
    size_node = root.find('size')
    data['size'] = {
        "width": int(size_node.findtext('width')),
        "height": int(size_node.findtext('height')),
        "depth": int(size_node.findtext('depth'))
    }

    # 4. <header>의 모든 정보
    header_node = root.find('header')
    data['header'] = {
        "duration": header_node.findtext('duration'),
        "fps": int(header_node.findtext('fps')),
        "frames": int(header_node.findtext('frames')),
        "inout": header_node.findtext('inout'),
        "location": header_node.findtext('location'),
        "season": header_node.findtext('season'),
        "weather": header_node.findtext('weather'),
        "time": header_node.findtext('time'),
        "population": int(header_node.findtext('population')),
        "character": header_node.findtext('character')
    }

    # 5. <event> 정보 및 'suspicious' 항목 추가
    event_node = root.find('event')
    event_name = event_node.findtext('eventname')
    data['event'] = {
        event_name: {
            "start_time_str": event_node.findtext('starttime'),
            "duration_str": event_node.findtext('duration')
        },
        "suspicious": {
            "start_time_str": "",
            "duration_str": ""
        }
    }

    # 6. <object>의 모든 정보 (position 포함)
    data['objects'] = []
    for obj_node in root.findall('object'):
        pos_node = obj_node.find('position')
        keypoint_node = pos_node.find('keypoint')
        
        obj_data = {
            "objectname": obj_node.findtext('objectname'),
            "position": {
                "keyframe": int(pos_node.findtext('keyframe')),
                "keypoint": {
                    "x": int(keypoint_node.findtext('x')),
                    "y": int(keypoint_node.findtext('y'))
                }
            },
            "actions": []
        }

        for action_node in obj_node.findall('action'):
            action_data = {
                "actionname": action_node.findtext('actionname'),
                "frames": []
            }
            for frame_node in action_node.findall('frame'):
                action_data['frames'].append({
                    "start": int(frame_node.findtext('start')),
                    "end": int(frame_node.findtext('end'))
                })
            obj_data['actions'].append(action_data)
        data['objects'].append(obj_data)

    return data

# --- 스크립트 실행 부분 ---
if __name__ == "__main__":
    # ❗ 여기에 변환할 XML 파일 경로를 입력하세요.
    xml_file = "10-1_cam01_assault03_place07_night_spring.xml"
    
    # 파일명 규칙에 따라 JSON 파일 이름 자동 생성
    base_filename = os.path.splitext(xml_file)[0]
    json_file = f"plus_{base_filename}.json"

    # 변환 함수 실행
    converted_data = convert_xml_to_full_json(xml_file)

    if converted_data:
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, indent=4, ensure_ascii=False)
        
        print(f"✅ 변환 완료! '{xml_file}' -> '{json_file}'")
