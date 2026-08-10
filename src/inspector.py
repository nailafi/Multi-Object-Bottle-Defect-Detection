import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time

class DefectInspector:
    def __init__(self, model_path, threshold, input_size=(128, 128), max_frames=15, name="Inspector"):
        self.name = name
        self.threshold = threshold
        self.input_size = input_size
        self.max_frames = max_frames
        self.model = load_model(model_path) 
        self.mapped_ids = {}
        self.next_id = 1
        self.object_data = {} 

    def evaluate(self, track_id, crop_img):
        if track_id not in self.mapped_ids:
            self.mapped_ids[track_id] = self.next_id
            self.next_id += 1
        
        display_id = self.mapped_ids[track_id]

        if display_id not in self.object_data:
            self.object_data[display_id] = {
                'mse_list': [],
                'status': 'SCANNING',
                'color': (255, 255, 0), # Kuning (RGB untuk Streamlit)
                'locked': False,
                'final_mse': 0.0
            }

        data = self.object_data[display_id]
        
        # Preprocessing
        crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        crop_resized = cv2.resize(crop_rgb, self.input_size)
        img = crop_resized.astype(np.float32) / 255.0
        img_input = np.expand_dims(img, axis=0)

        # Predict
        #reconstructed = self.model.predict(img_input, verbose=0)[0]
        start_cae = time.perf_counter() # Mulai hitung waktu
        reconstructed = self.model(img_input, training=False).numpy()[0]
        mse = np.mean((img - reconstructed) ** 2)
        end_cae = time.perf_counter() # Berhenti hitung

        cae_time_ms = (end_cae - start_cae) * 1000 # Ubah ms

        if not data['locked']:
            data['mse_list'].append(mse)
            if len(data['mse_list']) >= self.max_frames:
                avg_mse = sum(data['mse_list']) / self.max_frames
                data['final_mse'] = avg_mse
                data['status'] = "CACAT" if avg_mse > self.threshold else "NORMAL"
                data['color'] = (255, 0, 0) if data['status'] == "CACAT" else (0, 255, 0)
                data['locked'] = True

        #return display_id, data['status'], data['color'], mse if not data['locked'] else data['final_mse']
    
        # Ambil hasil
        current_status = data['status']
        current_color = data['color']

        if data['locked']:
            display_mse = data['final_mse']
        else:
            display_mse = data['mse_list'][-1]

        return display_id,current_status, current_color, display_mse, cae_time_ms