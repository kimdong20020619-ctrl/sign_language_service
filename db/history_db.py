import os
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional


class HistoryDB:
    """대화 내용을 SQLite에 저장하고 TXT/PDF로 내보낸다."""

    def __init__(self, db_path: str):
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self.db_path = db_path
        self._init_schema()

    # ------------------------------------------------------------------
    def _init_schema(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp   TEXT    NOT NULL,
                    role        TEXT    NOT NULL,
                    content     TEXT    NOT NULL,
                    mode        TEXT,
                    raw_sentence TEXT,
                    session_id  TEXT
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ts  ON conversations(timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sid ON conversations(session_id)"
            )

    # ------------------------------------------------------------------
    def save_message(self, role: str, content: str, mode: str,
                     raw_sentence: Optional[str] = None,
                     session_id: Optional[str] = None) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO conversations
                   (timestamp, role, content, mode, raw_sentence, session_id)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (datetime.now().isoformat(timespec="seconds"),
                 role, content, mode, raw_sentence, session_id),
            )

    def get_all(self) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM conversations ORDER BY timestamp"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_by_mode(self, mode: str) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM conversations WHERE mode=? ORDER BY timestamp",
                (mode,),
            ).fetchall()
            return [dict(r) for r in rows]

    def clear_all(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM conversations")

    # ------------------------------------------------------------------
    def export(self, fmt: str = "txt") -> str:
        records = self.get_all()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if fmt == "pdf":
            return self._export_pdf(records, stamp)
        return self._export_txt(records, stamp)

    # ── TXT 내보내기 ──────────────────────────────────────────────────
    def _export_txt(self, records: List[Dict], stamp: str) -> str:
        filename = f"conversation_{stamp}.txt"
        lines = [
            "수화 AI 소통 서비스 - 대화 기록",
            f"생성일시: {stamp}",
            "=" * 55,
            "",
        ]
        for r in records:
            role_ko = "청각장애인" if r["role"] == "deaf" else "비장애인"
            ts = r["timestamp"]
            mode = r.get("mode") or ""
            lines.append(f"[{ts}] [{mode}] {role_ko}")
            lines.append(f"  {r['content']}")
            if r.get("raw_sentence"):
                lines.append(f"  ↑ 원본 단어: {r['raw_sentence']}")
            lines.append("")

        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return filename

    # ── PDF 내보내기 ──────────────────────────────────────────────────
    def _export_pdf(self, records: List[Dict], stamp: str) -> str:
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.pdfgen import canvas as rl_canvas
        except ImportError:
            print("[HistoryDB] reportlab 미설치. TXT로 대신 내보냅니다.")
            return self._export_txt(records, stamp)

        filename = f"conversation_{stamp}.pdf"
        page_w, page_h = A4
        c = rl_canvas.Canvas(filename, pagesize=A4)
        margin = 50
        y = page_h - margin
        lh = 16  # line height

        def new_page():
            nonlocal y
            c.showPage()
            y = page_h - margin

        def write_line(text: str, indent: int = 0,
                       color=(0, 0, 0), size: int = 10) -> None:
            nonlocal y
            if y < margin + lh:
                new_page()
            c.setFillColorRGB(*color)
            c.setFont("Helvetica", size)
            # PDF는 기본 폰트에서 한글 지원 안 됨 → ASCII 변환
            safe = text.encode("ascii", errors="replace").decode("ascii")
            c.drawString(margin + indent, y, safe)
            y -= lh

        # 헤더
        c.setFont("Helvetica-Bold", 14)
        c.drawString(margin, y, "Sign Language AI Service - Conversation Log")
        y -= lh * 1.5
        write_line(f"Generated: {stamp}", size=9, color=(0.4, 0.4, 0.4))
        write_line("-" * 70, size=9, color=(0.5, 0.5, 0.5))
        y -= 4

        for r in records:
            role_en = "Deaf User" if r["role"] == "deaf" else "Hearing User"
            ts = r["timestamp"]
            mode = r.get("mode") or ""
            write_line(f"[{ts}] [{mode}] {role_en}",
                       color=(0.2, 0.3, 0.7), size=9)
            # 긴 내용 줄 분리
            content = r["content"]
            while content:
                write_line("  " + content[:90], indent=10, size=10)
                content = content[90:]
            if r.get("raw_sentence"):
                write_line(f"  (words: {r['raw_sentence'][:80]})",
                           indent=10, size=8, color=(0.5, 0.5, 0.5))
            y -= 4

        c.save()
        return filename
