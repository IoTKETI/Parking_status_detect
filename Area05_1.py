import json
from ultralytics import YOLO
import cv2
import numpy as np
import torch
import time
import paho.mqtt.client as mqtt
import requests
import threading
from datetime import datetime, timezone

# MQTT 설정
MQTT_HOST = "127.0.0.1"
MQTT_PORT = 1883
MQTT_CLIENT_ID = "area05_1_camera"
MQTT_SUBSCRIBE_TOPIC = "AIparking/area05"

# MQTT 자동 재연결 설정
FIRST_RECONNECT_DELAY = 1 
RECONNECT_RATE = 2 
MAX_RECONNECT_COUNT = 5 
MAX_RECONNECT_DELAY = 30 

# Mobius 설정
mobiusUrl = "http://203.253.128.164:7579/Mobius"
cntUrl = f"{mobiusUrl}/AIparkingblock"
cinUrl = f"{cntUrl}/KETI_Block_E"

# 데이터 통합을 위한 전역 변수
integrated_parking_status = {}
integrated_diou_scores = {}
mqtt_data_lock = threading.Lock()
last_mobius_send = {}

mqtt_client = None
previous_parking_status = None
previous_diou_scores = None

# Mobius 연동 함수들
def createAe():
    """AE 생성"""
    payload = {
        "m2m:ae": {
            "rn": "AIparkingblock",
            "api": "Yolo11_parking_detector",
            "lbl": ["key1", "key2"],
            "rr": True
        }
    }
    headers = {
        'Accept': 'application/json',
        'X-M2M-RI': '1234',
        'X-M2M-Origin': 'SAIparkingblock',
        'Content-Type': 'application/json;ty=2'
    }
    response = requests.post(mobiusUrl, headers=headers, json=payload)
    if response.status_code == 201:
        print("AE created successfully")
    elif response.status_code == 409:
        print("AE already exists")
    else:
        raise Exception(f"Failed to Create AE: {response.status_code}, {response.text}")

def createCnt():
    """Container 생성"""
    payload = {
        "m2m:cnt": {
            "rn": "KETI_Block_E",
            "lbl": ["KETI_Block_E"]
        }
    }
    headers = {
        'Accept': 'application/json',
        'X-M2M-RI': '1234',
        'X-M2M-Origin': 'SAIparkingblock',
        'Content-Type': 'application/json;ty=3'
    }
    response = requests.post(cntUrl, headers=headers, json=payload)
    if response.status_code == 201:
        print("Container created successfully")
    elif response.status_code == 409:
        print("Container already exists")
    else:
        raise Exception(f"Failed to Create Container: {response.status_code}, {response.text}")

def createCin(parking_status):
    """ContentInstance 생성"""
    payload = {
        "m2m:cin": {
            "con": parking_status
        }
    }
    headers = {
        'Accept': 'application/json',
        'X-M2M-RI': '1234',
        'X-M2M-Origin': 'SAIparkingblock',
        'Content-Type': 'application/json;ty=4'
    }    
    response = requests.post(cinUrl, headers=headers, json=payload)
    if response.status_code == 201:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return True
    else:
        print(f"Failed to Create ContentInstance: {response.status_code}, {response.text}")
        return False

def getAe():
    """AE 조회"""
    headers = {
        'Accept': 'application/json',
        'X-M2M-RI': '1234',
        'X-M2M-Origin': 'SAIparkingblock'
    }
    response = requests.get(cntUrl, headers=headers)
    if response.status_code == 200:
        print("AE found successfully")
    elif response.status_code == 404:
        print(f"AE not found, creating...")
        createAe()
    else:
        raise Exception(f"Failed to Get AE: {response.status_code}, {response.text}")

def getCnt():
    """Container 조회"""
    headers = {
        'Accept': 'application/json',
        'X-M2M-RI': '1234',
        'X-M2M-Origin': 'SAIparkingblock'
    }
    response = requests.get(cinUrl, headers=headers)
    if response.status_code == 200:
        print("Container found successfully")
    elif response.status_code == 404:
        print(f"Container not found, creating...")
        createCnt()
    else:
        raise Exception(f"Failed to Get Cnt: {response.status_code}, {response.text}")

# MQTT 브로커 연결 함수들
def on_connect(client, userdata, flags, rc):
    """MQTT 브로커 연결 시 호출"""
    if rc == 0:
        print(f"[MQTT] Connected to MQTT Broker")
        client.subscribe(MQTT_SUBSCRIBE_TOPIC)
        # print(f"[MQTT] Subscribed to {MQTT_SUBSCRIBE_TOPIC}")
    else:
        print(f"[MQTT] Failed to connect, return code {rc}")

def on_disconnect(client, userdata, rc):
    """MQTT 브로커 연결 해제 시 자동 재연결"""
    print(f"[MQTT] Disconnected with result code: {rc}")
    reconnect_count, reconnect_delay = 0, FIRST_RECONNECT_DELAY
    while reconnect_count < MAX_RECONNECT_COUNT:
        print(f"[MQTT] Reconnecting in {reconnect_delay} seconds...")
        time.sleep(reconnect_delay)

        try:
            client.reconnect()
            print(f"[MQTT] Reconnected successfully.")
            return
        except Exception as err:
            print(f"[MQTT] {err}. Reconnect failed. Retrying...")

        reconnect_delay *= RECONNECT_RATE
        reconnect_delay = min(reconnect_delay, MAX_RECONNECT_DELAY)
        reconnect_count += 1
    print(f"[MQTT] Reconnect failed after {reconnect_count} attempts. Exiting...")
    
def on_message(client, userdata, msg):
    """Area05_2로부터 area05 토픽 수신"""
    if msg.topic != MQTT_SUBSCRIBE_TOPIC:
        return
    
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
        parking_status = payload.get("parking_status", {})
        diou_scores = payload.get("diou_scores", {})
        
        # Area05_2에서 받은 area05 토픽 데이터 저장
        with mqtt_data_lock:
            for spot_id, status in parking_status.items():
                integrated_parking_status[spot_id] = status
            
            # DIoU 점수도 저장
            for spot_id, diou in diou_scores.items():
                integrated_diou_scores[spot_id] = diou
        
    except Exception as e:
        print(f"[MQTT] Message processing error: {e}")
        
# 데이터 통합 및 Mobius 전송 함수들
def integrate_parking_data(area05_1_parking_status, area05_1_diou_scores=None):
    """
    Area05_1과 Area05_2의 데이터를 통합
    - area05_1_parking_status: Area05_1 카메라로 감지한 주차 상태(128~133, 136~139)
    - area05_1_diou_scores: Area05_1의 DIoU 점수
    - integrated_parking_status: topic area05 에서 받은 데이터(126~130, 134~138)
    - integrated_diou_scores: Area05_2 DIoU 점수
    - 겹치는 주차면 (128, 129, 130, 133, 136, 137, 138): DIoU 점수 비교하여 높은 값 선택
    """
    global integrated_parking_status, integrated_diou_scores
    
    if area05_1_diou_scores is None:
        area05_1_diou_scores = {}
    
    complete_status = {}
    
    # Area05_1 데이터 추가 (131~132, 139)
    for spot_id, status in area05_1_parking_status.items():
        if (131 <= int(spot_id) <= 132) or (int(spot_id) == 139):
            complete_status[spot_id] = status
    
    with mqtt_data_lock:
        # topic area05 데이터 추가 (126~127, 134~135)
        for spot_id, status in integrated_parking_status.items():
            if (126 <= int(spot_id) <= 127) or (134 <= int(spot_id) <= 135):
                complete_status[spot_id] = status
        
        # 128, 129, 130, 133, 136, 137, 138번: 두 카메라의 DIoU를 비교해서 더 높은 값 선택
        for overlap_spot in ["128", "129", "130", "133", "136", "137", "138"]:
            area05_1_diou = area05_1_diou_scores.get(overlap_spot, 0.0)
            area05_2_diou = integrated_diou_scores.get(overlap_spot, 0.0)
            
            if area05_1_diou >= area05_2_diou:
                if overlap_spot in area05_1_parking_status:
                    complete_status[overlap_spot] = area05_1_parking_status[overlap_spot]
            else:
                if overlap_spot in integrated_parking_status:
                    complete_status[overlap_spot] = integrated_parking_status[overlap_spot]
    
    return complete_status

def send_to_mobius(parking_status):
    """
    통합된 주차 상태를 Mobius로 전송
    - 126~139 주차면 데이터 전송
    """
    sorted_items = sorted(parking_status.items(), key=lambda x: int(x[0]))
    mobius_payload = {f"{int(s):03d}": status for s, status in sorted_items}
    
    # Mobius로 전송
    success = createCin(mobius_payload)
    
    if success:
        # 전송 성공 시 마지막 전송 시간 업데이트
        last_mobius_send.update(parking_status)
    else:
        print(f"[Mobius] Failed to send Area05 data")
    
    return success

def init_mqtt():
    """MQTT 클라이언트 초기화"""
    global mqtt_client
    
    client = mqtt.Client(client_id=MQTT_CLIENT_ID, clean_session=True)
    
    # 콜백 함수 등록
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message
    
    # 브로커 접속
    try:
        client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
    except Exception as e:
        print(f"[MQTT] Connect error: {e}")
        return None
    
    client.loop_start()
    mqtt_client = client
    return client

def preprocess_image(image):
    """
    이미지 전처리 함수
    """
    # 1. 렌즈 왜곡 보정
    camera_matrix = np.array([[1000, 0, 960],
                              [0, 1000, 540],
                              [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array([-0.15, 0.03, 0.003, -0.003], dtype=np.float32)
    h, w = image.shape[:2]
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # 원근 변환
    pts = np.array([[620, 95], [1690, 187], [1754, 694], [665, 934]], dtype=np.float32)
    sm = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    topLeft = pts[np.argmin(sm)]
    bottomRight = pts[np.argmax(sm)]
    topRight = pts[np.argmin(diff)]
    bottomLeft = pts[np.argmax(diff)]

    pts1 = np.float32([topLeft, topRight, bottomRight, bottomLeft])
    pts2 = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    mtrx = cv2.getPerspectiveTransform(pts1, pts2)
    final_transformed_image = cv2.warpPerspective(undistorted_image, mtrx, (w, h))

    return final_transformed_image

def calculate_diou(box_a, box_b):
    """
    DIoU(Distance Intersection over Union) 계산 함수
    """
    x1_a, y1_a, x2_a, y2_a = box_a
    x1_b, y1_b, x2_b, y2_b = box_b

    inter_x1 = max(x1_a, x1_b)
    inter_y1 = max(y1_a, y1_b)
    inter_x2 = min(x2_a, x2_b)
    inter_y2 = min(y2_a, y2_b)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    area_a = (x2_a - x1_a) * (y2_a - y1_a)
    area_b = (x2_b - x1_b) * (y2_b - y1_b)
    union_area = area_a + area_b - inter_area
    iou = inter_area / union_area if union_area > 0 else 0

    # 중심점 거리 계산
    center_a = [(x1_a + x2_a) / 2, (y1_a + y2_a) / 2]
    center_b = [(x1_b + x2_b) / 2, (y1_b + y2_b) / 2]
    center_distance = (center_a[0] - center_b[0])**2 + (center_a[1] - center_b[1])**2

    # 최소 외접 박스 C 계산
    c_x1 = min(x1_a, x1_b)
    c_y1 = min(y1_a, y1_b)
    c_x2 = max(x2_a, x2_b)
    c_y2 = max(y2_a, y2_b)
    c_diagonal = (c_x2 - c_x1)**2 + (c_y2 - c_y1)**2

    diou = iou - (center_distance / c_diagonal) if c_diagonal > 0 else iou
    return diou

def load_parking_lines(file_path):
    """
    주차 공간의 좌표 파일을 읽어 YOLO 형식의 좌표를 OpenCV 형식의 좌표로 변환하는 함수
    """
    parking_lines = []
    img_width, img_height = 1920, 1080
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip().split()
                
                # YOLO 형식 좌표 (cx, cy, w, h)
                cx, cy, w, h = map(float, line[1:5])
                spot_id = line[5]  # 주차 공간 ID
                
                # YOLO 형식 좌표 (cx, cy, w, h) -> 픽셀 단위인 OpenCV 형식 좌표 (x1, y1, x2, y2) 변환
                x1, y1 = int((cx - w / 2) * img_width), int((cy - h / 2) * img_height)
                x2, y2 = int((cx + w / 2) * img_width), int((cy + h / 2) * img_height)
                parking_lines.append(((x1, y1, x2, y2), spot_id))  # (좌표 정보, spot_id) 형태로 저장
        return parking_lines
    except FileNotFoundError:
        print(f"Error: Could not find mapping file at {file_path}")
        return None

def load_original_parking_lines(file_path, image_width=1920, image_height=1080):
    """
    YOLO OBB 형식 좌표 파일을 읽어 꼭지점 4개의 정규화된 좌표 (x1~y4)를
    픽셀 기준의 꼭지점 좌표 [(x1,y1), (x2,y2), ...]로 변환하여 꼭지점 4개 좌표와 주차칸 ID를 반환함
    """
    parking_lines = []

    try:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 10:    # 총 10개의 값(class, x1~y4, spot_id)이 아닌 경우 무시
                    continue
                
                # YOLO OBB 형식 좌표 (x1, y1, x2, y2, x3, y3, x4, y4)
                coords = list(map(float, parts[1:9]))

                # 정규화된 좌표를 이미지 크기에 맞게 픽셀 좌표로 변환
                pixels = []
                for i in range(8):
                    if i % 2 == 0:  # 짝수 인덱스 = x좌표
                        pixel_value = int(round(coords[i] * image_width))
                    else:  # 홀수 인덱스 = y좌표
                        pixel_value = int(round(coords[i] * image_height))
                    pixels.append(pixel_value)

                points = [(pixels[i], pixels[i + 1]) for i in range(0, 8, 2)]
                spot_id = parts[9]

                parking_lines.append((points, spot_id))

        return parking_lines

    except FileNotFoundError:
        print(f"[Error] mapping_original.txt not found: {file_path}")
        return None

def analyze_parking_status(boxes, parking_lines): 
    """
    주차 공간과 차량 탐지 정보를 활용하여 DIoU 기반으로 주차 상태를 분석하는 함수
    """
    parking_status = {}
    diou_scores = {}

    for (px1, py1, px2, py2), spot_id in parking_lines:
        is_occupied = False
        best_diou = 0

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # 차량 바운딩 박스와 주차칸 박스가 겹치는지 확인
            if px1 < x2 and px2 > x1 and py1 < y2 and py2 > y1:
                diou = calculate_diou([x1, y1, x2, y2], [px1, py1, px2, py2])
                
                if diou > best_diou:
                    best_diou = diou

                # DIoU 값이 임계값(0.35) 이상이면 주차된 것으로 간주
                if diou >= 0.35:
                    is_occupied = True
                    break

        parking_status[f"{spot_id}"] = "occupied" if is_occupied else "free"
        diou_scores[f"{spot_id}"] = best_diou

    return parking_status, diou_scores

def draw_status_on_image(image, parking_lines, parking_status, diou_scores):
    """
    DIoU 기반 주차 상태 분석 결과를 이미지에 시각화하는 함수 (폴리곤 기반)
    """
    result_image = image.copy()
    overlay = image.copy()
    alpha = 0.3  # 투명도 조절 (0.0 ~ 1.0)

    for points, spot_id in parking_lines:
        status = parking_status.get(spot_id, "free")
        diou = diou_scores.get(spot_id, 0)

        # 색상 설정: free = 연한 초록색, occupied = 연한 빨간색
        color = (100, 255, 100) if status == "free" else (100, 100, 255)

        # 꼭지점 리스트를 OpenCV용 다각형 형태로 변환
        polygon = np.array(points, np.int32).reshape((-1, 1, 2))

        if status == "free":
            cv2.fillPoly(overlay, [polygon], color)

    cv2.addWeighted(overlay, alpha, result_image, 1 - alpha, 0, result_image)

    for points, spot_id in parking_lines:
        status = parking_status.get(spot_id, "free")
        diou = diou_scores.get(spot_id, 0)
        color = (100, 255, 100) if status == "free" else (100, 100, 255)
        polygon = np.array(points, np.int32).reshape((-1, 1, 2))

        cv2.polylines(result_image, [polygon], isClosed=True, color=color, thickness=3)

        # 주차 구역 번호, 주차 상태, DIoU 값 표시 (첫 꼭지점 위치 기준)
        tx, ty = points[0]
        # cv2.putText(result_image, f"{spot_id}", (tx + 5, ty + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # cv2.putText(result_image, status, (tx + 5, ty + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # cv2.putText(result_image, f"DIoU: {diou:.2f}", (tx + 5, ty + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    return result_image

def reconnect_rtsp(rtsp_url, retry_delay=3):
    """
    RTSP 스트림 재연결 함수
    """
    print(f"Attempting to reconnect to RTSP stream in {retry_delay} seconds...")
    time.sleep(retry_delay)
    cap = cv2.VideoCapture(rtsp_url)
    if cap.isOpened():
        print("Reconnected to RTSP stream.")
    else:
        print("Failed to reconnect.")
    return cap

def is_frame_valid(frame):
    """
    프레임이 정상인지 체크하는 함수
    """
    if frame is None:
        return False
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    std_val = np.std(gray)

    if std_val < 5:
        return False
    return True

def run_detection_loop(model, cap, parking_lines, original_parking_lines, rtsp_url):
    """
    주차 탐지 메인 루프 실행
    RTSP 스트림 수신-> YOLO 차량 탐지 -> DIoU 기반 주차 상태 판단 -> MQTT 데이터와 통합한 후 Mobius 전송 -> 시각화
    """
    global previous_parking_status
    
    retry_count = 0
    MAX_RETRY_COUNT = 10  # 최대 재시도 횟수
    frame_counter = 0
    FRAME_SKIP = 90  # 90프레임마다 1번만 YOLO 차량 탐지
    
    # 초기 설정
    initial_frame_read = False
    parking_status = {}
    diou_scores = {}

    while True:
        ret, frame = cap.read()
        if not ret or not is_frame_valid(frame):
            retry_count += 1
            print(f"Error: Failed to read frame from RTSP stream or bad frame received. Reconnecting... (Attempt #{retry_count}/{MAX_RETRY_COUNT})")
            
            # 최대 재시도 횟수 초과 시 종료
            if retry_count >= MAX_RETRY_COUNT:
                print(f"Error: Maximum retry count ({MAX_RETRY_COUNT}) reached. Exiting...")
                cap.release()
                cv2.destroyAllWindows()
                return
            
            cap.release()
            cap = reconnect_rtsp(rtsp_url)
            continue
        else:
            if retry_count > 0:
                # print(f"Reconnection successful after {retry_count} attempts.")
                pass
            retry_count = 0

        frame_counter += 1
        
        if not initial_frame_read or (frame_counter % FRAME_SKIP == 0):
            # 차량 탐지 -> DIoU 판단
            results = model(frame)
            boxes = results[0].boxes
            parking_status, diou_scores = analyze_parking_status(boxes, parking_lines)
            
            # MQTT 데이터와 통합
            complete_status = integrate_parking_data(parking_status, diou_scores)
            
            # Mobius 전송 (통합 데이터가 변경된 경우에만)
            if complete_status != last_mobius_send:
                send_to_mobius(complete_status)
                previous_parking_status = parking_status.copy()
                
                if not initial_frame_read:
                    # print("Initial parking detection completed and sent to Mobius.")
                    pass
                else:
                    print("Parking status changed, updated to Mobius.")
            
            if not initial_frame_read:
                initial_frame_read = True

        result_image = draw_status_on_image(frame, original_parking_lines, parking_status, diou_scores)
        cv2.imshow('Area05_1 Detection', result_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    global previous_parking_status
    
    # MQTT 클라이언트 초기화
    mqtt_client = init_mqtt()
    if mqtt_client is None:
        print("Failed to initialize MQTT client. Exiting...")
        return
    
    # AE 및 Container 확인 및 생성
    while True:
        try:
            getAe()
            getCnt()
            break
        
        except Exception as e:
            print("Retrying in 10 seconds...")
            time.sleep(10)
            continue

    # YOLO모델 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO('./pt/Area05_1.pt')
    model.to(device)
    
    # RTSP 연결
    rtsp_url = 'rtsp://admin:1234556@10.252.11.34/stream0'
    cap = cv2.VideoCapture(rtsp_url)
    
    # 전처리 매핑 파일 로딩(DIoU 판단용)
    mapping_file_path = './mapping/Area05_1_mapping.txt'
    parking_lines = load_parking_lines(mapping_file_path)
    
    # 원본 시각화 매핑 파일 로딩
    original_mapping_file_path = './mapping/Area05_1_mapping_original.txt'
    original_parking_lines = load_original_parking_lines(original_mapping_file_path)

    if not cap.isOpened() or parking_lines is None or original_parking_lines is None:
        print("Error: Could not open RTSP stream or load mapping file.")
        return

    # 주차 탐지 루프 실행
    run_detection_loop(model, cap, parking_lines, original_parking_lines, rtsp_url)
    
    # 종료 시 MQTT 클라이언트 정리
    if mqtt_client is not None:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
        print("MQTT client disconnected.")

if __name__ == "__main__":
    main()