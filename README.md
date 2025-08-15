# Face Recognition Attendance System (Flask)

A simple web-based attendance system using face recognition, built with Flask, OpenCV, and scikit-learn.

## Features

- Register new users with face images via webcam
- Train a face recognition model (KNN)
- Mark attendance automatically by recognizing faces
- Attendance records saved as CSV files by date
- Simple web interface

## Requirements

- Python 3.7+
- Flask
- OpenCV (`opencv-python`)
- scikit-learn
- pandas
- numpy
- joblib

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/face_recognition_flask.git
    cd face_recognition_flask
    ```

2. **Create and activate a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    venv\Scripts\activate  # On Windows
    # source venv/bin/activate  # On Linux/Mac
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Download required files:**
    - Place `haarcascade_frontalface_default.xml` in the project directory.
    - Place a `background.png` image in the project directory (optional, for UI).

## Usage

1. **Run the Flask app:**
    ```bash
    python app.py
    ```

2. **Open your browser and go to:**
    ```
    http://127.0.0.1:5000/
    ```

3. **Register a new user:**
    - Enter a name and ID, then capture face images via webcam.

4. **Mark attendance:**
    - Click "Start Attendance" and look at the webcam.

5. **View attendance:**
    - Attendance is saved in the `Attendance` folder as CSV files.

## Folder Structure

```
face_recognition_flask/
│
├── app.py
├── requirements.txt
├── README.md
├── haarcascade_frontalface_default.xml
├── background.png
├── Attendance/
│   └── Attendance-<date>.csv
└── static/
    └── faces/
        └── <username_id>/
            └── <face_images>.jpg
```

## Notes

- Make sure your webcam is working and accessible.
- For better accuracy, ensure good lighting and a clear view of faces.
- You can customize the HTML/CSS in the `templates` and `static` folders for a personalized interface.