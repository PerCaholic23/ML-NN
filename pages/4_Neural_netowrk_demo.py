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

# ตั้งค่าหน้าเว็บให้ดูทันสมัย
st.set_page_config(page_title="AI Customer Insight", page_icon="👤", layout="wide")

# ==========================================
# 1. กำหนด Path
# ==========================================
DATA_PATH = os.path.join("datasets", "neural_network", "customer_churn_data.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "custom_nn_model.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# ฟังก์ชันเทรนโมเดล (คงเดิมแต่ปรับปรุงข้อความแสดงผล)
def train_and_save_model():
    if not os.path.exists(DATA_PATH):
        st.error(f"❌ ไม่พบไฟล์ข้อมูลที่: {DATA_PATH}")
        return False
        
    with st.spinner("🧠 AI กำลังเรียนรู้พฤติกรรมลูกค้าจากฐานข้อมูล..."):
        df = pd.read_csv(DATA_PATH)
        if 'gender' in df.columns:
            df['gender'] = df['gender'].fillna(df['gender'].mode()[0])
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in num_cols:
            df[col] = df[col].fillna(df[col].mean())
            
        df = pd.get_dummies(df, drop_first=True)
        X = df.drop('target', axis=1)
        y = df['target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        
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
        
        model.save(MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
    return True

# ==========================================
# 2. หน้าเว็บหลัก (UI)
# ==========================================
st.title("👤 ระบบวิเคราะห์และทำนายการรักษาลูกค้า")
st.markdown("""
    **สถานการณ์:** ร้านฟิตเนส 'Active Plus' ต้องการทราบว่าลูกค้าคนไหนมีโอกาสจะเลิกเป็นสมาชิก 
    เพื่อให้ทีมการตลาดสามารถมอบส่วนลดหรือโปรโมชั่นดึงดูดลูกค้าได้ทันเวลา
""")
st.info("💡 **Neural Network** จะคำนวณจาก อายุ, รายได้ และคะแนนความพึงพอใจ เพื่อหาความเสี่ยง")

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.warning("🚨 ระบบยังไม่มีฐานความรู้ (Model)")
    if st.button("🔌 เริ่มสอน AI (Train Model)"):
        if train_and_save_model():
            st.success("✅ AI เรียนรู้เสร็จสิ้น!")
            st.rerun()
else:
    # โหลดโมเดล
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # ส่วนรับข้อมูล
    with st.container(border=True):
        st.subheader("📝 ข้อมูลลูกค้าที่ต้องการตรวจสอบ")
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("📅 อายุ (ปี)", 18, 100, 30)
            income = st.number_input("💰 รายได้ต่อเดือน (บาท)", 0, 200000, 35000)
        with c2:
            gender = st.selectbox("🚻 เพศ", ["Male", "Female"])
            score = st.slider("⭐️ คะแนนความพึงพอใจต่อบริการ (0-100)", 0.0, 100.0, 75.0)

    if st.button("🚀 วิเคราะห์แนวโน้ม", type="primary", use_container_width=True):
        # ทำนายผล
        gender_Male = 1 if gender == "Male" else 0
        input_data = pd.DataFrame({'age': [age], 'income': [income], 'score': [score], 'gender_Male': [gender_Male]})
        input_scaled = scaler.transform(input_data)
        prob = model.predict(input_scaled)[0][0]

        st.markdown("---")
        st.subheader("📊 ผลการวิเคราะห์")
        
        left, right = st.columns([1, 2])
        
        with left:
            st.metric("โอกาสการยกเลิก (Churn Rate)", f"{prob*100:.2f}%")
        
        with right:
            if prob > 0.5:
                st.error("⚠️ **สถานะ: เสี่ยงต่อการเสียลูกค้า**")
                st.write("🔴 ลูกค้าคนนี้มีแนวโน้มจะเลิกใช้บริการสูงมาก")
                st.markdown("**ข้อแนะนำ:** ทีมขายควรเสนอส่วนลด 20% หรือให้ทดลองใช้คลาสพิเศษฟรี เพื่อรักษาสมาชิก")
            else:
                st.success("✅ **สถานะ: ลูกค้าเหนียวแน่น**")
                st.write("🟢 ลูกค้ามีความพึงพอใจและมีแนวโน้มจะต่ออายุสมาชิก")
                st.markdown("**ข้อแนะนำ:** รักษามาตรฐานการบริการ และส่งข่าวสารกิจกรรมใหม่ๆ ตามปกติ")

    # ส่วนล่าง
    with st.expander("🛠 สำหรับผู้ดูแลระบบ"):
        if st.button("🔄 อัปเดตสมอง AI (Retrain Model)"):
            if train_and_save_model():
                st.rerun()