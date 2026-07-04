from flask import Flask, render_template, request, jsonify, send_file
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import cv2
import os
import sqlite3
from datetime import datetime
from openpyxl import Workbook

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

model = YOLO("yolov8n.pt")

def init_db():
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            filename TEXT,
            pizza_count INTEGER
        )
    """)
    conn.commit()
    conn.close()

init_db()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process_image():
    file = request.files["image"]

    filename = secure_filename(file.filename)
    upload_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(upload_path)

    img = cv2.imread(upload_path)
    results = model(img, classes=[53])

    pizza_count = 0

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        if class_name == "pizza":
            pizza_count += 1

    output_img = results[0].plot()
    result_filename = "result_" + filename
    result_path = os.path.join(RESULT_FOLDER, result_filename)
    cv2.imwrite(result_path, output_img)

    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO requests (timestamp, filename, pizza_count) VALUES (?, ?, ?)",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), filename, pizza_count)
    )
    conn.commit()
    conn.close()

    return jsonify({
        "count": pizza_count,
        "result_image": "/" + result_path
    })

@app.route("/export_excel")
def export_excel():
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, filename, pizza_count FROM requests")
    rows = cursor.fetchall()
    conn.close()

    wb = Workbook()
    ws = wb.active
    ws.title = "История обработки"

    ws.append(["Дата и время", "Файл", "Количество пицц"])

    for row in rows:
        ws.append(row)

    report_path = "pizza_report.xlsx"
    wb.save(report_path)

    return send_file(report_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)