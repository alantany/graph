import csv
import random
from datetime import datetime, timedelta
import os

# 创建healthcare目录（如果不存在）
if not os.path.exists('healthcare'):
    os.makedirs('healthcare')

# 生成随机日期
def random_date(start, end):
    return start + timedelta(
        seconds=random.randint(0, int((end - start).total_seconds())))

# 患者数据
patients = [
    {"id": f"P{i}", "name": f"Patient{i}", "age": random.randint(18, 80), "gender": random.choice(["Male", "Female"])}
    for i in range(1, 1001)
]

# 医生数据
doctors = [
    {"id": f"D{i}", "name": f"Dr. {random.choice(['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez'])}", "speciality": random.choice(["Cardiology", "Neurology", "Oncology", "Pediatrics", "Surgery"])}
    for i in range(1, 101)
]

# 疾病数据
diseases = [
    {"id": "DIS1", "name": "Hypertension"},
    {"id": "DIS2", "name": "Diabetes"},
    {"id": "DIS3", "name": "Asthma"},
    {"id": "DIS4", "name": "Arthritis"},
    {"id": "DIS5", "name": "Depression"}
]

# 症状数据
symptoms = [
    {"id": "SYM1", "name": "Headache"},
    {"id": "SYM2", "name": "Fever"},
    {"id": "SYM3", "name": "Cough"},
    {"id": "SYM4", "name": "Fatigue"},
    {"id": "SYM5", "name": "Nausea"}
]

# 药物数据
medications = [
    {"id": "MED1", "name": "Lisinopril"},
    {"id": "MED2", "name": "Metformin"},
    {"id": "MED3", "name": "Albuterol"},
    {"id": "MED4", "name": "Ibuprofen"},
    {"id": "MED5", "name": "Sertraline"}
]

# 医院数据
hospitals = [
    {"id": "H1", "name": "General Hospital"},
    {"id": "H2", "name": "City Medical Center"},
    {"id": "H3", "name": "Community Health Hospital"}
]

# 生成诊断记录
diagnoses = []
for _ in range(2000):
    patient = random.choice(patients)
    doctor = random.choice(doctors)
    disease = random.choice(diseases)
    diagnoses.append({
        "patient_id": patient["id"],
        "doctor_id": doctor["id"],
        "disease_id": disease["id"],
        "date": random_date(datetime(2020, 1, 1), datetime(2023, 5, 31)).strftime("%Y-%m-%d")
    })

# 生成症状记录
symptom_records = []
for diagnosis in diagnoses:
    for _ in range(random.randint(1, 3)):
        symptom = random.choice(symptoms)
        symptom_records.append({
            "patient_id": diagnosis["patient_id"],
            "symptom_id": symptom["id"],
            "date": diagnosis["date"]
        })

# 生成处方记录
prescriptions = []
for diagnosis in diagnoses:
    for _ in range(random.randint(1, 2)):
        medication = random.choice(medications)
        prescriptions.append({
            "patient_id": diagnosis["patient_id"],
            "doctor_id": diagnosis["doctor_id"],
            "medication_id": medication["id"],
            "date": diagnosis["date"],
            "dosage": f"{random.randint(1, 3)} times daily"
        })

# 生成医生工作记录
doctor_hospital = []
for doctor in doctors:
    hospital = random.choice(hospitals)
    doctor_hospital.append({
        "doctor_id": doctor["id"],
        "hospital_id": hospital["id"],
        "start_date": random_date(datetime(2015, 1, 1), datetime(2020, 1, 1)).strftime("%Y-%m-%d")
    })

# 生成保险理赔记录
insurance_claims = []
for _ in range(500):
    patient = random.choice(patients)
    insurance_claims.append({
        "claim_id": f"C{_+1}",
        "patient_id": patient["id"],
        "amount": round(random.uniform(100, 10000), 2),
        "date": random_date(datetime(2020, 1, 1), datetime(2023, 5, 31)).strftime("%Y-%m-%d"),
        "status": random.choice(["Approved", "Pending", "Rejected"])
    })

# 将数据保存为CSV文件
def save_to_csv(data, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

save_to_csv(patients, 'healthcare/patients.csv')
save_to_csv(doctors, 'healthcare/doctors.csv')
save_to_csv(diseases, 'healthcare/diseases.csv')
save_to_csv(symptoms, 'healthcare/symptoms.csv')
save_to_csv(medications, 'healthcare/medications.csv')
save_to_csv(hospitals, 'healthcare/hospitals.csv')
save_to_csv(diagnoses, 'healthcare/diagnoses.csv')
save_to_csv(symptom_records, 'healthcare/symptom_records.csv')
save_to_csv(prescriptions, 'healthcare/prescriptions.csv')
save_to_csv(doctor_hospital, 'healthcare/doctor_hospital.csv')
save_to_csv(insurance_claims, 'healthcare/insurance_claims.csv')

print("医疗健康数据生成完成。")