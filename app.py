import logging
import pickle
import sqlite3
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from predict import predict_incident

BASE_DIR = Path(__file__).resolve().parent
DATABASE_DIR = BASE_DIR / "data"
DATABASE_PATH = DATABASE_DIR / "incidents.sqlite3"
INDEX_PATH = BASE_DIR / "static" / "index.html"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@contextmanager
def _database_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


@asynccontextmanager
async def lifespan(_: FastAPI):
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    with _database_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                incident_number TEXT UNIQUE,
                description TEXT NOT NULL,
                application TEXT NOT NULL,
                severity TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'Open',
                prediction_status TEXT NOT NULL DEFAULT 'PENDING',
                predicted_impact TEXT,
                impact_probability REAL,
                created_at TEXT NOT NULL
            )
            """
        )
    yield


app = FastAPI(
    title="Incident Impact Prediction",
    version="1.0.0",
    lifespan=lifespan,
)


class IncidentRequest(BaseModel):
    incident_id: str
    description: str = Field(min_length=5, max_length=2000)
    application: str
    severity: str


class IncidentCreate(BaseModel):
    description: str = Field(min_length=5, max_length=2000)
    application: str = Field(min_length=1, max_length=120)
    severity: str = Field(min_length=1, max_length=40)

    @field_validator("description", "application", "severity", mode="before")
    @classmethod
    def strip_text_fields(cls, value):
        return value.strip() if isinstance(value, str) else value


def _serialize_incident(row: sqlite3.Row) -> dict:
    return {
        "incident_id": row["incident_number"],
        "description": row["description"],
        "application": row["application"],
        "severity": row["severity"],
        "state": row["state"],
        "prediction_status": row["prediction_status"],
        "predicted_impact": row["predicted_impact"],
        "impact_probability": row["impact_probability"],
        "created_at": row["created_at"],
    }


def _get_incident(incident_number: str) -> dict:
    with _database_connection() as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT * FROM incidents WHERE incident_number = ?",
            (incident_number,),
        ).fetchone()
    if row is None:
        raise RuntimeError(f"Incident {incident_number} was not saved")
    return _serialize_incident(row)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(INDEX_PATH)


@app.get("/api/incidents")
def list_incidents():
    with _database_connection() as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT * FROM incidents ORDER BY id DESC"
        ).fetchall()
    return [_serialize_incident(row) for row in rows]


@app.post("/api/incidents")
def create_incident(request: IncidentCreate):
    created_at = datetime.now(timezone.utc).isoformat()

    # Save the system-of-record entry before attempting model inference.
    with _database_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO incidents (
                description, application, severity, created_at
            ) VALUES (?, ?, ?, ?)
            """,
            (
                request.description.strip(),
                request.application.strip(),
                request.severity.strip(),
                created_at,
            ),
        )
        incident_number = f"INC{cursor.lastrowid:06d}"
        connection.execute(
            "UPDATE incidents SET incident_number = ? WHERE id = ?",
            (incident_number, cursor.lastrowid),
        )

    try:
        prediction = predict_incident(request.description.strip())
    except (
        OSError,
        EOFError,
        pickle.UnpicklingError,
        ImportError,
        AttributeError,
        ValueError,
        TypeError,
        RuntimeError,
    ):
        logger.exception("Prediction failed for %s", incident_number)
        with _database_connection() as connection:
            connection.execute(
                """
                UPDATE incidents
                SET prediction_status = 'PREDICTION_UNAVAILABLE'
                WHERE incident_number = ?
                """,
                (incident_number,),
            )
    else:
        with _database_connection() as connection:
            connection.execute(
                """
                UPDATE incidents
                SET prediction_status = 'COMPLETED',
                    predicted_impact = ?,
                    impact_probability = ?
                WHERE incident_number = ?
                """,
                (
                    prediction["predicted_impact"],
                    prediction["impact_probability"],
                    incident_number,
                ),
            )

    return _get_incident(incident_number)


@app.post("/predict")
def predict(request: IncidentRequest):
    result = predict_incident(request.description)
    return {
        "incident_id": request.incident_id,
        "application": request.application,
        "severity": request.severity,
        "predicted_impact": result["predicted_impact"],
        "impact_probability": result["impact_probability"],
    }
