import random
import csv
from datetime import datetime, timedelta
import os

def generate_patient(patient_id):
    return {
        'id': f'P{patient_id:05d}',
        'name': f'Patient{patient_id}',
        'age': random.randint(18, 90),
        'gender': random.choice(['Male', 'Female']),
        'blood_type': random.choice(['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-'])
    }

def generate_doctor(doctor_id):
    specialties = ['Cardiology', 'Neurology', 'Oncology', 'Pediatrics', 'Surgery', 'Internal Medicine']
    return {
        'id': f'D{doctor_id:05d}',
        'name': f'Dr. {random.choice(["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis"])}',
        'specialty': random.choice(specialties)
    }

def generate_diagnosis(diagnosis_id, patient_id, doctor_id):
    diseases = ['Hypertension', 'Diabetes', 'Asthma', 'Arthritis', 'Depression', 'Cancer', 'Heart Disease']
    return {
        'id': f'DG{diagnosis_id:07d}',
        'patient_id': f'P{patient_id:05d}',
        'doctor_id': f'D{doctor_id:05d}',
        'disease': random.choice(diseases),
        'date': (datetime.now() - timedelta(days=random.randint(0, 365))).isoformat()
    }

def generate_medication(medication_id):
    medications = ['Aspirin', 'Ibuprofen', 'Amoxicillin', 'Lisinopril', 'Metformin', 'Atorvastatin', 'Levothyroxine']
    return {
        'id': f'M{medication_id:05d}',
        'name': random.choice(medications),
        'dosage': f'{random.randint(1, 1000)} mg'
    }

def generate_prescription(prescription_id, diagnosis_id, medication_id):
    return {
        'id': f'PR{prescription_id:07d}',
        'diagnosis_id': f'DG{diagnosis_id:07d}',
        'medication_id': f'M{medication_id:05d}',
        'duration': f'{random.randint(1, 30)} days'
    }

def generate_healthcare_data(num_patients=1000, num_doctors=50, num_diagnoses=5000, num_medications=100):
    patients = [generate_patient(i) for i in range(num_patients)]
    doctors = [generate_doctor(i) for i in range(num_doctors)]
    medications = [generate_medication(i) for i in range(num_medications)]
    
    diagnoses = [generate_diagnosis(i, random.randint(0, num_patients-1), random.randint(0, num_doctors-1)) 
                 for i in range(num_diagnoses)]
    
    prescriptions = [generate_prescription(i, i, random.randint(0, num_medications-1)) 
                     for i in range(num_diagnoses)]

    return patients, doctors, diagnoses, medications, prescriptions

def write_to_csv(data, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

if __name__ == "__main__":
    healthcare_dir = 'healthcare'
    os.makedirs(healthcare_dir, exist_ok=True)

    patients, doctors, diagnoses, medications, prescriptions = generate_healthcare_data()
    
    write_to_csv(patients, os.path.join(healthcare_dir, 'patients.csv'))
    write_to_csv(doctors, os.path.join(healthcare_dir, 'doctors.csv'))
    write_to_csv(diagnoses, os.path.join(healthcare_dir, 'diagnoses.csv'))
    write_to_csv(medications, os.path.join(healthcare_dir, 'medications.csv'))
    write_to_csv(prescriptions, os.path.join(healthcare_dir, 'prescriptions.csv'))

    print(f"Healthcare data generation complete. CSV files have been created in the '{healthcare_dir}' directory.")
    print(f"Total patients: {len(patients)}")
    print(f"Total doctors: {len(doctors)}")
    print(f"Total diagnoses: {len(diagnoses)}")
    print(f"Total medications: {len(medications)}")
    print(f"Total prescriptions: {len(prescriptions)}")