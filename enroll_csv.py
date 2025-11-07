# enroll_csv.py
import csv
import os
from datetime import datetime

CSV_FILE = "students.csv"

def read_rfid_keyboard():
    """If your reader acts like a keyboard."""
    uid = input("Scan card now (or type UID): ").strip()
    return uid

def enroll(uid, student_id, name, course, gender):
    file_exists = os.path.exists(CSV_FILE)
    row = {
        "uid": uid,
        "student_id": student_id,
        "name": name,
        "course": course.upper(),
        "gender": gender.capitalize(),
        "enrolled_at": datetime.utcnow().isoformat()
    }
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print("✅ Enrolled:", row)

def lookup(uid):
    if not os.path.exists(CSV_FILE):
        return None
    with open(CSV_FILE, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["uid"] == uid:
                return r
    return None

def main():
    print("=== Enrollment tool ===")
    while True:
        uid = read_rfid_keyboard()
        if not uid:
            continue
        existing = lookup(uid)
        if existing:
            print("⚠️ UID already enrolled:", existing)
            if input("Overwrite? (y/N): ").strip().lower() != "y":
                continue
        student_id = input("Student ID: ").strip()
        name = input("Name: ").strip()
        course = input("Course (e.g. BSBA): ").strip()
        gender = input("Gender (Male/Female): ").strip().capitalize()
        enroll(uid, student_id, name, course, gender)
        print("Enroll another? (Ctrl+C to stop)")

if __name__ == "__main__":
    main()
