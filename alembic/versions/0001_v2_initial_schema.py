"""Fortivex V2 schema migration (leads external_crm_ids_json, session metrics_json, crm_sync_jobs).

Revision ID: 0001_v2_schema
Revises:
Create Date: 2026-11-26
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0001_v2_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())

    if "leads" in tables:
        lead_cols = {c["name"] for c in inspector.get_columns("leads")}
        if "external_crm_ids_json" not in lead_cols:
            op.add_column("leads", sa.Column("external_crm_ids_json", sa.Text(), server_default="{}"))

    if "sessions" in tables:
        sess_cols = {c["name"] for c in inspector.get_columns("sessions")}
        if "metrics_json" not in sess_cols:
            op.add_column("sessions", sa.Column("metrics_json", sa.Text(), nullable=True))

    if "crm_sync_jobs" not in tables:
        op.create_table(
            "crm_sync_jobs",
            sa.Column("id", sa.Text(), primary_key=True),
            sa.Column("adapter", sa.Text(), nullable=False),
            sa.Column("operation", sa.Text(), nullable=False),
            sa.Column("entity_id", sa.Text(), nullable=False),
            sa.Column("idempotency_key", sa.Text(), nullable=False, unique=True),
            sa.Column("payload_json", sa.Text(), nullable=False),
            sa.Column("status", sa.Text(), nullable=False),
            sa.Column("retry_count", sa.Integer(), server_default="0"),
            sa.Column("http_status", sa.Integer(), nullable=True),
            sa.Column("external_record_id", sa.Text(), nullable=True),
            sa.Column("error_code", sa.Text(), nullable=True),
            sa.Column("failure_reason", sa.Text(), nullable=True),
            sa.Column("created_at", sa.Text(), nullable=False),
            sa.Column("updated_at", sa.Text(), nullable=False),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "crm_sync_jobs" in inspector.get_table_names():
        op.drop_table("crm_sync_jobs")
