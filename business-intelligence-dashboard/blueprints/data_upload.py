from flask import Blueprint, render_template, request
from ui_config import get_current_user
data_upload_bp = Blueprint("data_upload", __name__)


def _base_context():
    return {
        "active_page": "data_upload",
        "processing_status": [],
        "user": get_current_user(),
    }


@data_upload_bp.route("/data-upload")
def index():
    return render_template("data_upload.html", **_base_context())


@data_upload_bp.route("/data-upload/submit", methods=["POST"])
def submit():
    uploaded_file = request.files.get("csv_file")
    context = _base_context()

    if uploaded_file is None or uploaded_file.filename == "":
        context["upload_result"] = {
            "level": "error",
            "title": "No file selected",
            "message": "Choose a CSV file before submitting the ingestion job.",
        }
        return render_template("data_upload.html", **context), 400

    context["upload_result"] = {
        "level": "success",
        "title": "Upload queued",
        "message": f"{uploaded_file.filename} was received and queued for processing.",
    }
    return render_template("data_upload.html", **context)


@data_upload_bp.route("/data-upload/connect", methods=["POST"])
def connect():
    context = _base_context()
    host = request.form.get("host", "").strip()
    database = request.form.get("database", "").strip()
    port = request.form.get("port", "").strip()
    username = request.form.get("username", "").strip()

    if not host or not database or not port or not username:
        context["connection_result"] = {
            "level": "error",
            "title": "Missing connection details",
            "message": "Fill in host, database, port, and username before validating the connection.",
        }
        return render_template("data_upload.html", **context), 400

    context["connection_result"] = {
        "level": "success",
        "title": "Connection validated",
        "message": f"Prepared a connection attempt for {username}@{host}:{port}/{database}.",
    }
    return render_template("data_upload.html", **context)
