import requests
import datetime
import os
from dotenv import load_dotenv

load_dotenv()
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY")

BASE_URL = "https://v3.football.api-sports.io"

# ID chính xác của các giải đấu lớn
MAJOR_LEAGUE_IDS = {
    1: "FIFA World Cup",
    2: "UEFA Champions League",
    3: "UEFA Europa League",
    39: "Premier League (Anh)",
    61: "Ligue 1 (Pháp)",
    78: "Bundesliga (Đức)",
    135: "Serie A (Ý)",
    140: "La Liga (Tây Ban Nha)",
    848: "UEFA Conference League",
    4: "Euro Championship",
    9: "Copa America",
}

def get_headers():
    return {"x-apisports-key": API_FOOTBALL_KEY}

VN_OFFSET = datetime.timezone(datetime.timedelta(hours=7))  # GMT+7

def _utc_str_to_vn(date_str):
    """Chuyển chuỗi UTC sang datetime theo giờ Việt Nam (GMT+7)."""
    try:
        dt = datetime.datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return dt.astimezone(VN_OFFSET)
    except:
        return None

def _format_vn_time(date_str):
    """Trả về chuỗi giờ VN dạng HH:MM (VN)."""
    dt_vn = _utc_str_to_vn(date_str)
    return dt_vn.strftime("%H:%M") if dt_vn else "?"

def get_fixtures_by_date(date_str=None):
    """
    Lấy lịch/kết quả theo ngày. 
    Trả về dict với: success, wc_data, all_major_data, error
    """
    if not API_FOOTBALL_KEY or API_FOOTBALL_KEY == "YOUR_API_FOOTBALL_KEY_HERE":
        return {"success": False, "error": "API_FOOTBALL_KEY chưa được cấu hình."}

    if date_str is None:
        # Dùng ngày hiện tại theo giờ Việt Nam để tránh lệch ngày
        date_str = datetime.datetime.now(VN_OFFSET).strftime("%Y-%m-%d")

    try:
        resp = requests.get(
            f"{BASE_URL}/fixtures",
            headers=get_headers(),
            params={"date": date_str},
            timeout=10
        )
        data = resp.json()

        errors = data.get("errors")
        if errors and errors != []:
            return {"success": False, "error": str(errors), "date": date_str}

        all_fixtures = data.get("response", [])

        # Lọc theo league ID chính xác
        major_fixtures = [f for f in all_fixtures if f.get("league", {}).get("id") in MAJOR_LEAGUE_IDS]
        wc_fixtures = [f for f in all_fixtures if f.get("league", {}).get("id") == 1]

        def format_fixture(f):
            fix = f.get("fixture", {})
            teams = f.get("teams", {})
            goals = f.get("goals", {})
            league = f.get("league", {})
            status = fix.get("status", {})
            short = status.get("short", "")
            elapsed = status.get("elapsed")

            home = teams.get("home", {}).get("name", "?")
            away = teams.get("away", {}).get("name", "?")
            home_g = goals.get("home")
            away_g = goals.get("away")
            round_name = league.get("round", "")
            league_name = league.get("name", "")
            
            # Chuyển sang giờ VN đầy đủ (ngày + giờ)
            dt_vn = _utc_str_to_vn(fix.get("date", ""))
            if dt_vn:
                time_vn = dt_vn.strftime("%H:%M")
                date_vn = dt_vn.strftime("%d/%m")
                datetime_vn = f"{date_vn} lúc {time_vn}"
            else:
                time_vn = "?"
                datetime_vn = "?"

            if short == "FT":
                score_str = f"{home_g} - {away_g} (KT)"
            elif short in ("1H", "2H", "HT", "ET", "P"):
                score_str = f"{home_g} - {away_g} ({elapsed}')"
            elif short == "NS":
                score_str = f"vs  ⏰ {datetime_vn} (Giờ VN)"
            else:
                score_str = f"{home_g} - {away_g} ({short})"

            return f"⚽ {home} {score_str} {away}  [{round_name}]"

        return {
            "success": True,
            "date": date_str,
            "wc_fixtures": [format_fixture(f) for f in wc_fixtures],
            "major_fixtures": [format_fixture(f) for f in major_fixtures],
            "wc_count": len(wc_fixtures),
            "major_count": len(major_fixtures),
        }

    except Exception as e:
        return {"success": False, "error": str(e), "date": date_str}

def get_live_fixtures():
    """Lấy kết quả đang diễn ra (live) - chỉ giải lớn."""
    if not API_FOOTBALL_KEY or API_FOOTBALL_KEY == "YOUR_API_FOOTBALL_KEY_HERE":
        return {"success": False, "error": "API_FOOTBALL_KEY chưa được cấu hình."}

    try:
        resp = requests.get(
            f"{BASE_URL}/fixtures",
            headers=get_headers(),
            params={"live": "all"},
            timeout=10
        )
        data = resp.json()
        fixtures = data.get("response", [])

        major = [f for f in fixtures if f.get("league", {}).get("id") in MAJOR_LEAGUE_IDS]
        wc = [f for f in fixtures if f.get("league", {}).get("id") == 1]

        def fmt(f):
            t = f.get("teams", {})
            g = f.get("goals", {})
            el = f.get("fixture", {}).get("status", {}).get("elapsed", "?")
            lg = f.get("league", {}).get("name", "")
            return f"🔴 [{lg}] {t.get('home',{}).get('name')} {g.get('home',0)}-{g.get('away',0)} {t.get('away',{}).get('name')} | {el}'"

        return {
            "success": True,
            "live_count": len(fixtures),
            "wc_live": [fmt(f) for f in wc],
            "major_live": [fmt(f) for f in major],
        }

    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    print("=== TEST HÔM NAY ===")
    r = get_fixtures_by_date()
    print(f"WC: {r.get('wc_count')}, Major: {r.get('major_count')}")
    for line in r.get("wc_fixtures", []):
        print(" ", line)
