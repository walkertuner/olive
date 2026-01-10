from pathlib import Path
import gspread
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

creds = Credentials.from_service_account_file(
    Path(__file__).parent / "railsback.json",
    scopes=SCOPES,
)

gc = gspread.authorize(creds)

worksheet = gc.open("railsback").sheet1

records = worksheet.get_all_records()

def get_record(instr_id):
    row = next((r for r in records if str(r["id"]) == str(instr_id)), None)
    return row