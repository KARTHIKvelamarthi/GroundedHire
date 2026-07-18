import sqlite3
import json
import os

DB_PATH = "grounded_hire.db"

if not os.path.exists(DB_PATH):
    print(f"Error: Database file '{DB_PATH}' does not exist yet. Run an interview first to initialize it.")
    exit(1)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

def list_sessions():
    cursor.execute("""
        SELECT id, candidate_name, role, current_difficulty_tier, created_at
        FROM sessions
        ORDER BY created_at DESC
    """)
    rows = cursor.fetchall()
    if not rows:
        print("No interview sessions found in the database.")
        return

    print("\n" + "="*80)
    print(f"{'SESSION ID':<38} | {'CANDIDATE NAME':<20} | {'ROLE':<15} | {'DIFFICULTY'}")
    print("="*80)
    for row in rows:
        sid, name, role, difficulty, created = row
        role_disp = "AI/ML" if role == "ai_ml_engineer" else "Data Science"
        print(f"{sid:<38} | {name:<20} | {role_disp:<15} | {difficulty}")
    print("="*80)
    print("\nTo view detailed questions and answers for a session, run:")
    print("  python view_db.py <session_id>\n")

def view_session_details(session_id):
    # Get session info
    cursor.execute("""
        SELECT candidate_name, role, current_difficulty_tier, created_at, resume_profile_json
        FROM sessions
        WHERE id = ?
    """, (session_id,))
    session = cursor.fetchone()
    if not session:
        print(f"Error: Session '{session_id}' not found.")
        return

    name, role, difficulty, created, profile_json = session
    role_disp = "AI/ML Engineer" if role == "ai_ml_engineer" else "Data Scientist"
    
    print("\n" + "="*80)
    print(f"SESSION DETAILS")
    print("="*80)
    print(f"Candidate Name : {name}")
    print(f"Role           : {role_disp}")
    print(f"Difficulty     : {difficulty}")
    print(f"Created At     : {created}")
    print("="*80)

    # Get Q&As
    cursor.execute("""
        SELECT q.sequence_number, q.question_text, q.topic, q.difficulty_tier,
               a.answer_text, a.quality_signal, a.brief_feedback
        FROM questions q
        LEFT JOIN answers a ON q.id = a.question_id
        WHERE q.session_id = ?
        ORDER BY q.sequence_number ASC
    """, (session_id,))
    
    qas = cursor.fetchall()
    if not qas:
        print("No questions have been asked in this session yet.")
        return

    for qa in qas:
        seq, q_text, topic, q_diff, a_text, signal, feedback = qa
        print(f"\n[Question {seq}] (Topic: {topic} | Difficulty: {q_diff})")
        print(f"Q: {q_text}")
        if a_text:
            print(f"A: {a_text}")
            print(f"Evaluation: [{signal.upper()}] - {feedback}")
        else:
            print("A: [Pending Answer]")
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        view_session_details(sys.argv[1])
    else:
        list_sessions()

conn.close()
