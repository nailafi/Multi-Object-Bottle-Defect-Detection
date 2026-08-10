import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from inspector import DefectInspector 
import time
import winsound

# CONFIG & SESSION STATE
st.set_page_config(page_title="Bottle Inspection", layout="wide")

if 'inspectors' not in st.session_state:
    st.session_state.inspectors = {
        'botol': DefectInspector('cae_model_botol.keras', 0.0025, (128, 128), max_frames=5, name="Botol"),
        'label': DefectInspector('cae_model_label.keras', 0.0035, (128, 128), max_frames=5, name="Label"),
        'tutup': DefectInspector('cae_model_tutup.keras', 0.0035, (64, 64), max_frames=5, name="Tutup")
    }
    st.session_state.yolo = YOLO('best_last.pt')
    st.session_state.result_log = [] 

# UI LAYOUT
st.title("Automated Bottle Inspection System")

col_left, col_right = st.columns([6, 4])

with col_left:
    st.subheader("Live Camera Feed")
    frame_placeholder = st.empty() 

    st.subheader("Threshold Active (Real-Time)")
    t1, t2, t3 = st.columns(3)
    thresh_bottle = t1.number_input("Threshold Botol", value=0.0025, format="%.4f")
    thresh_cap = t2.number_input("Threshold Tutup", value=0.0035, format="%.4f")
    thresh_label = t3.number_input("Threshold Label", value=0.0035, format="%.4f")
    
    st.session_state.inspectors['botol'].threshold = thresh_bottle
    st.session_state.inspectors['tutup'].threshold = thresh_cap
    st.session_state.inspectors['label'].threshold = thresh_label

    st.subheader("Inspection Log")
    table_placeholder = st.empty()
    download_placeholder = st.empty()

with col_right:
    st.subheader("Counter")
    count_cols = st.columns(3)
    total_m = count_cols[0].empty()
    normal_m = count_cols[1].empty()
    defect_m = count_cols[2].empty()
    
    st.subheader("Real-time MSE Charts")
    st.text("MSE Botol")
    chart_bottle = st.empty()
    st.text("MSE Tutup")
    chart_cap = st.empty()
    st.text("MSE Label")
    chart_label = st.empty()

# MAIN LOOP
st.sidebar.markdown("### Control Panel")
cam_idx = st.sidebar.selectbox("Camera Index", [0, 1, 2, 3], index=2)
run_system = st.sidebar.checkbox("▶ Start Inspection")

if run_system:
    cap = cv2.VideoCapture(cam_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    history_mse = {'botol': [], 'tutup': [], 'label': []}
    log_length = 0 
    frame_counter = 0

    while cap.isOpened():
        start_frame_time = time.perf_counter()
        ret, frame = cap.read()
        
        if not ret: 
            st.error("Gagal mengambil gambar dari kamera.")
            break
        
        frame_counter += 1

        # YOLO Tracking
        results = st.session_state.yolo.track(
            frame, 
            persist=True, 
            tracker="botsort.yaml", 
            imgsz=320, 
            verbose=False
        )
        
        yolo_time_ms = results[0].speed['inference']
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes
            track_ids = boxes.id.int().cpu().tolist()
            classes = boxes.cls.int().cpu().tolist()
            xyxy_list = boxes.xyxy.int().cpu().tolist()
            
            # ZONA INSPEKSI
            screen_width = frame.shape[1]
            batas_kiri = int(screen_width * 0.3)
            batas_kanan = int(screen_width * 0.7)
            
            # Garis Zona Virtual
            #cv2.line(frame, (batas_kiri, 0), (batas_kiri, frame.shape[0]), (255, 0, 0), 2)
            #cv2.line(frame, (batas_kanan, 0), (batas_kanan, frame.shape[0]), (255, 0, 0), 2)

            # Kategori
            list_botol = []
            list_status_extra = [] 

            for i in range(len(classes)):
                x1, y1, x2, y2 = xyxy_list[i]
                cx = (x1 + x2) // 2

                if cx < batas_kiri or cx > batas_kanan: continue

                t_id = track_ids[i]
                cls = classes[i]
                box = xyxy_list[i]

                if cls == 0: 
                    list_botol.append({'id': t_id, 'box': box, 'final_cls': 0, 'crop_box': box})
                else: 
                    list_status_extra.append({'cls': cls, 'box': box, 'id': t_id})

            # LOGIKA ASOSIASI PER OBJEK
            def is_inside(small_box, big_box):
                sm_x1, sm_y1, sm_x2, sm_y2 = small_box
                bg_x1, bg_y1, bg_x2, bg_y2 = big_box
                sm_cx = (sm_x1 + sm_x2) // 2
                sm_cy = (sm_y1 + sm_y2) // 2
                return (bg_x1 < sm_cx < bg_x2) and (bg_y1 < sm_cy < bg_y2)

            for botol in list_botol:
                has_label = False
                has_cap = False
                label_obj = None
                cap_obj = None

                for extra in list_status_extra:
                    if is_inside(extra['box'], botol['box']):
                        if extra['cls'] == 1:
                            has_label = True
                            label_obj = extra
                        elif extra['cls'] == 2:
                            has_cap = True
                            cap_obj = extra

                if has_label:
                    botol['final_cls'] = 1
                    botol['crop_box'] = label_obj['box'] 
                    botol['eval_id'] = label_obj['id']
                elif has_cap:
                    botol['final_cls'] = 2 
                    botol['crop_box'] = cap_obj['box'] 
                    botol['eval_id'] = cap_obj['id']
                else:
                    botol['final_cls'] = 0 
                    botol['crop_box'] = botol['box'] 
                    botol['eval_id'] = botol['id']

            # JALANKAN INSPEKSI
            for botol in list_botol:
                f_cls = botol['final_cls']
                x1, y1, x2, y2 = botol['crop_box']

                x1, y1 = max(0, x1), max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0: continue

                if f_cls == 0:
                    insp, roi_name = st.session_state.inspectors['botol'], "botol"
                elif f_cls == 1:
                    insp, roi_name = st.session_state.inspectors['label'], "label"
                else:
                    insp, roi_name = st.session_state.inspectors['tutup'], "tutup"

                disp_id, status, color, mse, cae_time = insp.evaluate(botol['eval_id'], crop)
                
                key = "botol" if f_cls == 0 else ("label" if f_cls == 1 else "tutup")
                history_mse[key].append(mse)
                if len(history_mse[key]) > 20: history_mse[key].pop(0)

                current_fps = 1.0 / (time.perf_counter() - start_frame_time)

                if insp.object_data[disp_id]['locked'] and len(insp.object_data[disp_id]['mse_list']) == insp.max_frames:
                    if status == "CACAT":
                        winsound.Beep(1000, 500)
                    
                    st.session_state.result_log.append({
                        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "ID": disp_id, 
                        "ROI": roi_name.upper(), 
                        "MSE": round(float(mse), 6),
                        "FPS": round(float(current_fps), 1),
                        "YOLO (ms)": round(float(yolo_time_ms), 1),
                        "CAE (ms)": round(float(cae_time), 1), 
                        "STATUS": status
                    })
                    insp.object_data[disp_id]['mse_list'].append(0) 

                # VISUALISASI BOUNDING BOX
                cx1, cy1, cx2, cy2 = botol['crop_box'] # Target hanya kotak yang diinspeksi
                
                cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (color[2], color[1], color[0]), 3)
                
                # ID | STATUS
                text_bbox = f"{disp_id} | {status}"
                cv2.putText(frame, text_bbox, (cx1, cy1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (color[2], color[1], color[0]), 2)

        # UPDATE UI
        if frame_counter % 3 == 0:
            frame_placeholder.image(frame, channels="BGR", use_container_width=True)
            
        if len(st.session_state.result_log) > log_length:
            df_full = pd.DataFrame(st.session_state.result_log)
            total_m.metric("Total", len(df_full))
            normal_m.metric("Normal", len(df_full[df_full['STATUS'] == 'NORMAL']))
            defect_m.metric("Cacat", len(df_full[df_full['STATUS'] == 'CACAT']))
            
            table_placeholder.dataframe(df_full, use_container_width=True, height=300)
            log_length = len(st.session_state.result_log)

        if frame_counter % 5 == 0:
            if history_mse['botol']: chart_bottle.line_chart(history_mse['botol'], height=100)
            if history_mse['tutup']: chart_cap.line_chart(history_mse['tutup'], height=100)
            if history_mse['label']: chart_label.line_chart(history_mse['label'], height=100)

    cap.release()

# TOMBOL DOWNLOAD
if st.session_state.result_log:
    df_download = pd.DataFrame(st.session_state.result_log)
    csv_bytes = df_download.to_csv(index=False).encode('utf-8')
    
    download_placeholder.download_button(
        label=f"💾 Download Seluruh Data Inspeksi ({len(df_download)} Deteksi)",
        data=csv_bytes,
        file_name=f"log_inspeksi_{time.strftime('%Y%m%d')}.csv",
        mime="text/csv",
        key="download_csv_full"
    )

if not run_system:
    st.info("Menunggu inspeksi dimulai... Centang 'Start Inspection' di sidebar.")