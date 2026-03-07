"""
Data migration script: Import CSV exports from LearnAI Scope + GlassRoom Lab
into the unified ReasonLens Azure PostgreSQL database.

Usage: python3 db/004_data_migration.py
"""

import csv
import json
import psycopg
import os
from pathlib import Path

DB_URL = "postgresql://rladmin:plOYhiKsL2P4HS5e6WVs10HU@reasonlens-db.postgres.database.azure.com/reasonlens?sslmode=require"

LEARN_SCOPE_DIR = Path(os.path.expanduser("~/Downloads/tables learn scope"))
GLASSROOM_DIR = Path(os.path.expanduser("~/Downloads/reasonlens database tables"))


def find_csv(directory, prefix):
    """Find a CSV file by table name prefix."""
    for f in directory.glob(f"{prefix}-export-*.csv"):
        return f
    return None


def read_csv(path, delimiter=";"):
    """Read a semicolon-delimited CSV into a list of dicts."""
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        return [row for row in reader]


def empty_to_none(val):
    """Convert empty strings to None."""
    return val if val else None


def parse_pg_array(val: str) -> list:
    """Parse PostgreSQL array format like ["a","b"] or {a,b}."""
    if not val:
        return []
    val = val.strip()
    if val.startswith("[") or val.startswith('"['):
        try:
            cleaned = val.strip('"').replace('""', '"')
            return json.loads(cleaned)
        except:
            return []
    if val.startswith("{") and val.endswith("}"):
        inner = val[1:-1]
        if not inner:
            return []
        return [s.strip('"') for s in inner.split(",")]
    return []


def main():
    conn = psycopg.connect(DB_URL)
    conn.autocommit = True
    cur = conn.cursor()

    print("=" * 60)
    print("ReasonLens Data Migration")
    print("=" * 60)

    # ===== 1. PROFILES (from LearnAI Scope) =====
    print("\n--- Profiles (LearnAI Scope) ---")
    profiles_csv = find_csv(LEARN_SCOPE_DIR, "profiles")
    if profiles_csv:
        rows = read_csv(profiles_csv)
        for row in rows:
            cur.execute("""
                INSERT INTO profiles (id, auth_provider_id, email, full_name, institution, comfort_level, region, sector, institution_type, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
            """, (
                row["id"], f"supabase-learnscope:{row['id']}", row["email"],
                empty_to_none(row.get("full_name")), empty_to_none(row.get("institution")),
                int(row["comfort_level"]) if row.get("comfort_level") else None,
                empty_to_none(row.get("region")), empty_to_none(row.get("sector")),
                empty_to_none(row.get("institution_type")),
                row["created_at"], row["updated_at"]
            ))
        print(f"  Imported {len(rows)} profiles")

    # ===== 2. GLASSROOM USERS (create profile stubs) =====
    print("\n--- GlassRoom Users (profile stubs) ---")
    gl_roles_csv = find_csv(GLASSROOM_DIR, "user_roles")
    gl_user_ids = set()
    if gl_roles_csv:
        rows = read_csv(gl_roles_csv)
        for row in rows:
            gl_user_ids.add(row["user_id"])

    # Also get user_ids from runs
    runs_csv = find_csv(GLASSROOM_DIR, "runs")
    if runs_csv:
        for row in read_csv(runs_csv):
            gl_user_ids.add(row["created_by"])

    # Check which ones already exist (might overlap with LearnAI Scope users)
    for uid in gl_user_ids:
        cur.execute("SELECT id FROM profiles WHERE id = %s", (uid,))
        if not cur.fetchone():
            cur.execute("""
                INSERT INTO profiles (id, auth_provider_id, email, full_name, created_at)
                VALUES (%s, %s, %s, %s, now())
                ON CONFLICT (id) DO NOTHING
            """, (uid, f"supabase-glassroom:{uid}", f"user-{uid[:8]}@glassroom.local", "GlassRoom User"))
            print(f"  Created stub profile for {uid[:8]}...")
    print(f"  Processed {len(gl_user_ids)} GlassRoom users")

    # ===== 3. USER ROLES (both projects) =====
    print("\n--- User Roles ---")
    # LearnAI Scope roles (no granted_by column)
    ls_roles_csv = find_csv(LEARN_SCOPE_DIR, "user_roles")
    count = 0
    if ls_roles_csv:
        for row in read_csv(ls_roles_csv):
            cur.execute("""
                INSERT INTO user_roles (id, user_id, role, granted_at)
                VALUES (%s, %s, %s::app_role, %s)
                ON CONFLICT (user_id, role) DO NOTHING
            """, (row["id"], row["user_id"], row["role"], row["created_at"]))
            count += 1

    # GlassRoom roles
    if gl_roles_csv:
        for row in read_csv(gl_roles_csv):
            cur.execute("""
                INSERT INTO user_roles (id, user_id, role, granted_by, granted_at)
                VALUES (%s, %s, %s::app_role, %s, %s)
                ON CONFLICT (user_id, role) DO NOTHING
            """, (row["id"], row["user_id"], row["role"],
                  empty_to_none(row.get("granted_by")), row["granted_at"]))
            count += 1
    print(f"  Imported {count} roles")

    # ===== 4. USER GOALS =====
    print("\n--- User Goals ---")
    goals_csv = find_csv(LEARN_SCOPE_DIR, "user_goals")
    if goals_csv:
        rows = read_csv(goals_csv)
        for row in rows:
            cur.execute("""
                INSERT INTO user_goals (id, user_id, goal, created_at)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (user_id, goal) DO NOTHING
            """, (row["id"], row["user_id"], row["goal"], row["created_at"]))
        print(f"  Imported {len(rows)} goals")

    # ===== 5. ASSESSMENT RESULTS =====
    print("\n--- Assessment Results ---")
    ar_csv = find_csv(LEARN_SCOPE_DIR, "assessment_results")
    if ar_csv:
        rows = read_csv(ar_csv)
        for row in rows:
            cur.execute("""
                INSERT INTO assessment_results (id, user_id, framework_id, framework_name, question_id, dimension, selected_level, completed_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
            """, (row["id"], row["user_id"], row["framework_id"], row["framework_name"],
                  row["question_id"], row["dimension"], row["selected_level"], row["completed_at"]))
        print(f"  Imported {len(rows)} results")

    # ===== 6. FRAMEWORK PROGRESS =====
    print("\n--- Framework Progress ---")
    fp_csv = find_csv(LEARN_SCOPE_DIR, "framework_progress")
    if fp_csv:
        rows = read_csv(fp_csv)
        for row in rows:
            cur.execute("""
                INSERT INTO framework_progress (id, user_id, framework_id, framework_name, progress, completed_items, total_items, last_activity, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id, framework_id) DO NOTHING
            """, (row["id"], row["user_id"], row["framework_id"], row["framework_name"],
                  int(row["progress"]), int(row["completed_items"]), int(row["total_items"]),
                  row["last_activity"], row["created_at"], row["updated_at"]))
        print(f"  Imported {len(rows)} progress rows")

    # ===== 7. AUDIT RUNS (GlassRoom) =====
    print("\n--- Audit Runs ---")
    if runs_csv:
        rows = read_csv(runs_csv)
        for row in rows:
            cur.execute("""
                INSERT INTO audit_runs (id, created_by, scenario_pack, auditor_model, target_model, judge_model,
                    max_turns, cap_tokens, cap_cost, status, cost_tokens, cost_currency, error_message,
                    created_at, started_at, completed_at, samples_per_scenario, posthoc_packs, benchmark_packs,
                    mode, simple_mode_context)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::run_status, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
            """, (
                row["id"], row["created_by"], row["scenario_pack"],
                row["auditor_model"], row["target_model"], row["judge_model"],
                int(row["max_turns"]) if row.get("max_turns") else 10,
                int(row["cap_tokens"]) if row.get("cap_tokens") else None,
                float(row["cap_cost"]) if row.get("cap_cost") else None,
                row["status"] if row.get("status") else "completed",
                int(row["cost_tokens"]) if row.get("cost_tokens") else None,
                float(row["cost_currency"]) if row.get("cost_currency") else None,
                empty_to_none(row.get("error_message")),
                row["created_at"],
                empty_to_none(row.get("started_at")),
                empty_to_none(row.get("completed_at")),
                int(row["samples_per_scenario"]) if row.get("samples_per_scenario") else 1,
                parse_pg_array(row.get("posthoc_packs", "")) or None,
                parse_pg_array(row.get("benchmark_packs", "")) or None,
                empty_to_none(row.get("mode")),
                row.get("simple_mode_context") if row.get("simple_mode_context") else None,
            ))
        print(f"  Imported {len(rows)} runs")

    # ===== 8. POSTHOC PACK RESULTS =====
    print("\n--- Posthoc Pack Results ---")
    pp_csv = find_csv(GLASSROOM_DIR, "posthoc_pack_results")
    if pp_csv:
        rows = read_csv(pp_csv)
        for row in rows:
            cur.execute("""
                INSERT INTO posthoc_pack_results (id, run_id, pack_id, status, metrics_json, evidence_json, error_message, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (run_id, pack_id) DO NOTHING
            """, (
                row["id"], row["run_id"], row["pack_id"], row["status"],
                row.get("metrics_json") or None,
                row.get("evidence_json") or None,
                empty_to_none(row.get("error_message")),
                row["created_at"], row["updated_at"]
            ))
        print(f"  Imported {len(rows)} posthoc results")

    # ===== 9. BENCHMARK RUNS =====
    print("\n--- Benchmark Runs ---")
    br_csv = find_csv(GLASSROOM_DIR, "benchmark_runs")
    if br_csv:
        rows = read_csv(br_csv)
        seen = set()
        for row in rows:
            if row["id"] in seen:
                continue
            seen.add(row["id"])
            cur.execute("""
                INSERT INTO benchmark_runs (id, created_by, benchmark_type, target_model, status, config_json, results_json, error_message, petri_run_id, started_at, completed_at, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
            """, (
                row["id"], row["created_by"], row["benchmark_type"], row["target_model"],
                row["status"],
                row.get("config_json") or None,
                row.get("results_json") or None,
                empty_to_none(row.get("error_message")),
                empty_to_none(row.get("petri_run_id")),
                empty_to_none(row.get("started_at")),
                empty_to_none(row.get("completed_at")),
                row["created_at"]
            ))
        print(f"  Imported {len(seen)} benchmark runs")

    # ===== 10. REPORTS =====
    print("\n--- Audit Reports ---")
    reports_csv = find_csv(GLASSROOM_DIR, "reports")
    if reports_csv:
        rows = read_csv(reports_csv)
        for row in rows:
            cur.execute("""
                INSERT INTO audit_reports (id, run_id, content_markdown, visuals_json, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
            """, (
                row["id"], row["run_id"],
                empty_to_none(row.get("content_markdown")),
                row.get("visuals_json") or None,
                row["created_at"], row["updated_at"]
            ))
        print(f"  Imported {len(rows)} reports")

    # ===== 11. AUDIT LOG =====
    print("\n--- Audit Log ---")
    al_csv = find_csv(GLASSROOM_DIR, "audit_log")
    if al_csv:
        rows = read_csv(al_csv)
        for row in rows:
            cur.execute("""
                INSERT INTO audit_log (id, user_id, run_id, action, details, ts)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
            """, (
                row["id"],
                empty_to_none(row.get("user_id")),
                empty_to_none(row.get("run_id")),
                row["action"],
                row.get("details") or None,
                row["ts"]
            ))
        print(f"  Imported {len(rows)} audit log entries")

    # ===== 12. TRANSCRIPTS (SKIP for now — 1.4M lines, ~140MB) =====
    print("\n--- Transcripts ---")
    print("  SKIPPED: 1.4M lines (~140MB). Import separately if needed.")
    print("  Command: Use COPY or batch insert for large transcript import.")

    # ===== VERIFY =====
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    tables = [
        "profiles", "user_roles", "user_goals", "assessment_results",
        "framework_progress", "audit_runs", "posthoc_pack_results",
        "benchmark_runs", "audit_reports", "audit_log", "badges",
        "models", "scenarios"
    ]
    for table in tables:
        cur.execute(f"SELECT count(*) FROM {table}")
        count = cur.fetchone()[0]
        print(f"  {table}: {count} rows")

    conn.close()
    print("\nMigration complete!")


if __name__ == "__main__":
    main()
