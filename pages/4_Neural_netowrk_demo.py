import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os

st.set_page_config(page_title="Neural Network Demo", page_icon="", layout="centered")

# ==========================================
# 1. กำหนด Path ของไฟล์ต่างๆ ตามโครงสร้างโฟลเดอร์
# ==========================================
# Path สำหรับอ่านข้อมูล
DATA_PATH = os.path.join("datasets", "neural_network", "customer_churn_data.csv")

# Path สำหรับเซฟ/โหลด โมเดล
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "custom_nn_model.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# สร้างโฟลเดอร์ models เตรียมไว้ล่วงหน้า (ถ้ายังไม่มี)
os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================================
# 2. ฟังก์ชันสำหรับเทรนโมเดล (อ่านจาก datasets -> เซฟลง models)
# ==========================================
def train_and_save_model():
    if not os.path.exists(DATA_PATH):
        st.error(f"ไม่พบไฟล์ข้อมูลที่: {DATA_PATH} กรุณาตรวจสอบโฟลเดอร์")
        return False
        
    with st.spinner("กำลังเตรียมข้อมูลและเทรนโมเดล (อาจใช้เวลาสักครู่)..."):
        # 1. โหลดข้อมูล
        df = pd.read_csv(DATA_PATH)
        
        # 2. จัดการ Missing Values
        if 'gender' in df.columns:
            df['gender'].fillna(df['gender'].mode()[0], inplace=True)
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in num_cols:
            df[col].fillna(df[col].mean(), inplace=True)
            
        # 3. แปลงข้อมูล
        df = pd.get_dummies(df, drop_first=True)
        X = df.drop('target', axis=1) # สมมติว่าคอลัมน์ผลลัพธ์ชื่อ target
        y = df['target']
        
        # 4. Train-Test Split & Scaling
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        
        # 5. สร้างและเทรน Neural Network
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32, callbacks=[early_stop], verbose=0)
        
        # 6. บันทึกโมเดลลงโฟลเดอร์ models/
        model.save(MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        
    st.success(f"เทรนโมเดลสำเร็จ! บันทึกไฟล์ไว้ที่โฟลเดอร์ `{MODEL_DIR}/` แล้ว")
    return True

# ==========================================
# 3. หน้าเว็บ Streamlit (UI)
# ==========================================
st.title("Neural Network Demo")
st.markdown("ทดสอบการทำนายโอกาสยกเลิกบริการ (Customer Churn)")

# เช็คว่ามีโมเดลอยู่ในโฟลเดอร์ models หรือยัง
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.warning("ยังไม่มีไฟล์โมเดลที่ถูกเทรนในโฟลเดอร์ `models/`")
    if st.button("เริ่มเทรนโมเดลเดี๋ยวนี้"):
        train_and_save_model()
        st.rerun() # รีเฟรชหน้าเว็บหลังเทรนเสร็จ
else:
    # --- ส่วนของการทดสอบ (Demo) ---
    st.success("โหลดโมเดลพร้อมใช้งานแล้ว")
    
    # โหลด Model และ Scaler จากโฟลเดอร์ models/
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    st.markdown("---")
    st.header("กรอกข้อมูลลูกค้าเพื่อทำนาย")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("อายุ (Age)", 18, 100, 30)
        income = st.number_input("รายได้ (Income)", 0, 200000, 35000, step=1000)
    with col2:
        gender = st.selectbox("เพศ (Gender)", ["Male", "Female"])
        score = st.slider("คะแนนความพึงพอใจ (Score)", 0.0, 100.0, 50.0)
        
    if st.button("ประมวลผล", type="primary", use_container_width=True):
        # เตรียมข้อมูลสำหรับทำนาย (ต้องให้โครงสร้างเหมือนตอนเทรน)
        gender_Male = 1 if gender == "Male" else 0
        input_data = pd.DataFrame({'age': [age], 'income': [income], 'score': [score], 'gender_Male': [gender_Male]})
        
        # Scale ข้อมูลและทำนาย
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0][0]
        
        st.markdown("---")
        st.subheader("ผลการทำนาย")
        if prediction > 0.5:
            st.error(f"มีโอกาส **ยกเลิกบริการ** (ความน่าจะเป็น: {prediction*100:.2f}%)")
        else:
            st.success(f"มีโอกาส **ใช้บริการต่อ** (ความน่าจะเป็นที่จะยกเลิก: {prediction*100:.2f}%)")

    st.markdown("---")
    # ปุ่มสำหรับบังคับเทรนใหม่ เผื่อมีการอัปเดตไฟล์ CSV
    with st.expander("ตั้งค่าผู้พัฒนา (Developer Options)"):
        if st.button("บังคับเทรนโมเดลใหม่ (Retrain Model)"):
            train_and_save_model()
            st.rerun()