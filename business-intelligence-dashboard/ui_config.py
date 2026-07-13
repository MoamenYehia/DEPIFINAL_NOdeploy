# business-intelligence-dashboard/ui_config.py

REGIONS = ["All Regions", "Region X", "Region Y", "Region Z"]
TIME_RANGES = [("7d", "Next 7 Days"), ("30d", "Next 30 Days"), ("90d", "Next 90 Days")]
SEVERITIES = ["All", "Critical", "Warning", "Info"]
STATUSES = ["All", "Open", "Acknowledged", "Resolved"]
DATE_RANGES = ["Today", "This Week", "This Month", "All Time"]

def get_current_user():
    return {"name": "Admin User", "email": "admin@domain.com"}