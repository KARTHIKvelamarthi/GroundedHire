import os
import logging
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker, declarative_base

logger = logging.getLogger("backend.database")

# Locate the database file at the root or within backend
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "grounded_hire.db"))
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False}  # Needed for SQLite multi-threading in FastAPI
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def run_migrations(target_engine=None):
    """
    Safe, idempotent startup migration for SQLite database.
    Checks for the existence of required new columns and executes ALTER TABLE
    if missing. Preserves all existing data.
    """
    db_engine = target_engine or engine
    inspector = inspect(db_engine)
    existing_tables = inspector.get_table_names()

    if "sessions" in existing_tables:
        session_columns = {col["name"].lower() for col in inspector.get_columns("sessions")}
        session_alterations = [
            ("company", "VARCHAR"),
            ("jd_text", "TEXT"),
            ("jd_profile", "JSON"),
            ("topic_plan", "JSON"),
        ]
        with db_engine.begin() as conn:
            for col_name, col_type in session_alterations:
                if col_name.lower() not in session_columns:
                    logger.info(f"Adding column '{col_name}' to 'sessions' table...")
                    conn.execute(text(f"ALTER TABLE sessions ADD COLUMN {col_name} {col_type}"))

    if "questions" in existing_tables:
        question_columns = {col["name"].lower() for col in inspector.get_columns("questions")}
        question_alterations = [
            ("source", "VARCHAR"),
            ("resume_ref", "VARCHAR"),
        ]
        with db_engine.begin() as conn:
            for col_name, col_type in question_alterations:
                if col_name.lower() not in question_columns:
                    logger.info(f"Adding column '{col_name}' to 'questions' table...")
                    conn.execute(text(f"ALTER TABLE questions ADD COLUMN {col_name} {col_type}"))

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
