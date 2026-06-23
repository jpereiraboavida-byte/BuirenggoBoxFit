"""
Database module — SQLite (lokal) ho MySQL-compatible SQL syntax
BuirenggoBoxFit | DIT Machine Learning 2026
"""
import sqlite3
import pandas as pd
from datetime import datetime
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "buirenggoboxfit.db")

MYSQL_SCHEMA = """-- ══════════════════════════════════════════
-- MySQL Schema — BuirenggoBoxFit Database
-- ══════════════════════════════════════════
CREATE DATABASE IF NOT EXISTS buirenggoboxfit;
USE buirenggoboxfit;

CREATE TABLE IF NOT EXISTS atleta (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    naran           VARCHAR(100) NOT NULL,
    data_regista    DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS avaliasaun (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    atleta_naran    VARCHAR(100),
    vo2max          FLOAT NOT NULL,
    pushup          INT   NOT NULL,
    situp           INT   NOT NULL,
    shuttle_run     FLOAT NOT NULL,
    sprint_30m      FLOAT NOT NULL,
    sit_reach       FLOAT NOT NULL,
    body_fat        FLOAT NOT NULL,
    bmi             FLOAT NOT NULL,
    grip_strength   FLOAT NOT NULL,
    vertical_jump   FLOAT NOT NULL,
    klasifikasaun   VARCHAR(20) NOT NULL,
    konfidensja     FLOAT,
    data_avaliasaun DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (atleta_naran) REFERENCES atleta(naran)
);

-- INSERT example:
INSERT INTO avaliasaun
    (atleta_naran, vo2max, pushup, situp, shuttle_run, sprint_30m,
     sit_reach, body_fat, bmi, grip_strength, vertical_jump,
     klasifikasaun, konfidensja)
VALUES
    ('Atleta A', 52.3, 45, 48, 10.8, 4.4,
     31.0, 14.2, 23.1, 49.5, 62.0,
     'Diak', 87.0);

-- SELECT history:
SELECT * FROM avaliasaun ORDER BY data_avaliasaun DESC;
"""


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS avaliasaun (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            atleta_naran    TEXT,
            vo2max          REAL, pushup INTEGER, situp INTEGER,
            shuttle_run     REAL, sprint_30m REAL, sit_reach REAL,
            body_fat        REAL, bmi REAL, grip_strength REAL,
            vertical_jump   REAL,
            klasifikasaun   TEXT,
            konfidensja     REAL,
            data_avaliasaun TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_result(atleta_naran, input_data, klasifikasaun, konfidensja):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO avaliasaun
            (atleta_naran, vo2max, pushup, situp, shuttle_run, sprint_30m,
             sit_reach, body_fat, bmi, grip_strength, vertical_jump,
             klasifikasaun, konfidensja, data_avaliasaun)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        atleta_naran or "Anonimu",
        input_data['vo2max'], input_data['pushup'], input_data['situp'],
        input_data['shuttle_run'], input_data['sprint_30m'], input_data['sit_reach'],
        input_data['body_fat'], input_data['bmi'], input_data['grip_strength'],
        input_data['vertical_jump'],
        klasifikasaun, round(konfidensja, 2),
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    conn.close()


def get_history():
    init_db()
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM avaliasaun ORDER BY data_avaliasaun DESC", conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df


def delete_record(record_id):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM avaliasaun WHERE id=?", (record_id,))
    conn.commit()
    conn.close()


def get_stats():
    init_db()
    conn = sqlite3.connect(DB_PATH)
    try:
        total = pd.read_sql("SELECT COUNT(*) as n FROM avaliasaun", conn).iloc[0, 0]
        dist  = pd.read_sql(
            "SELECT klasifikasaun, COUNT(*) as n FROM avaliasaun GROUP BY klasifikasaun", conn
        )
    except Exception:
        total = 0
        dist  = pd.DataFrame()
    conn.close()
    return total, dist
