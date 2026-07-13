from flask import Blueprint, render_template
from ui_config import get_current_user
settings_bp = Blueprint("settings", __name__)


@settings_bp.route("/settings")
def index():
    context = {
        "active_page": "settings",
        "user": get_current_user(),
        "settings_items": [
            {"label": "Theme", "value": "Light"},
            {"label": "Notifications", "value": "Enabled"},
            {"label": "Data refresh", "value": "Hourly"},
            {"label": "Role", "value": "Admin"},
        ],
    }
    return render_template("settings.html", **context)
