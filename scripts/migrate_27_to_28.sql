-- Migration: schema 27 → 28.
--
-- Schema 28 reshapes the `translations` UNIQUE key:
--
--     OLD: UNIQUE (work_chapter_id, owner_id, target_lang)
--     NEW: UNIQUE (work_chapter_id, owner_id, draft_id)
--
-- Reason: the old key forced a single translation per (chapter,
-- target_lang, user), so a user couldn't keep an "AI VI dịch từ EN
-- MangaDex" và một "AI VI dịch từ KR Lezhin" song song — the second
-- spawn would overwrite `draft_id` on the existing row. The new key
-- pins uniqueness to (chapter, owner, draft) instead so different
-- source-language drafts coexist as separate translation rows under
-- the same target_lang, while a re-spawn from the exact same draft
-- still no-ops (idempotent).
--
-- Pre-existing rows are already compliant: the old constraint was
-- strictly stronger than the new one (a single row per
-- (wc, owner, target_lang) trivially satisfies uniqueness on
-- (wc, owner, draft_id) since `draft_id` is NOT NULL). No data
-- rewrite needed; just swap the constraint.
--
-- Idempotent: re-running after success is a no-op (meta version
-- already 28; constraint swap guarded by IF EXISTS / IF NOT EXISTS
-- on the index/constraint names).
--
-- Usage:
--   PGPASSWORD=… psql -h … -U … -d typoon -f scripts/migrate_27_to_28.sql

BEGIN;

-- Refuse to run if the DB is on the wrong baseline. Bail out loud
-- rather than silently dropping constraints on an unrelated schema.
DO $$
DECLARE
    cur TEXT;
BEGIN
    SELECT value INTO cur FROM meta WHERE key='schema_version';
    IF cur IS DISTINCT FROM '27' THEN
        RAISE EXCEPTION
            'expected schema_version=27, found %; abort migration',
            COALESCE(cur, '<missing>');
    END IF;
END $$;

-- The old UNIQUE was created inline with the table, so Postgres
-- auto-named the backing constraint `translations_work_chapter_id_owner_id_target_lang_key`.
-- We don't rely on the auto-name though — look up by columns and
-- drop whatever constraint backs that exact tuple, so this migration
-- survives a future where someone renamed the constraint.
DO $$
DECLARE
    con_name TEXT;
BEGIN
    SELECT conname INTO con_name
    FROM   pg_constraint c
    JOIN   pg_class      t ON t.oid = c.conrelid
    WHERE  t.relname = 'translations'
      AND  c.contype = 'u'
      AND  c.conkey  = (
          SELECT array_agg(a.attnum ORDER BY a.attnum)
          FROM   pg_attribute a
          WHERE  a.attrelid = t.oid
            AND  a.attname IN ('work_chapter_id', 'owner_id', 'target_lang')
      );
    IF con_name IS NOT NULL THEN
        EXECUTE format('ALTER TABLE translations DROP CONSTRAINT %I', con_name);
    END IF;
END $$;

-- Add the new constraint. Guarded with a duplicate-check so re-runs
-- after a partial apply don't error.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM   pg_constraint c
        JOIN   pg_class      t ON t.oid = c.conrelid
        WHERE  t.relname = 'translations'
          AND  c.contype = 'u'
          AND  c.conkey  = (
              SELECT array_agg(a.attnum ORDER BY a.attnum)
              FROM   pg_attribute a
              WHERE  a.attrelid = t.oid
                AND  a.attname IN ('work_chapter_id', 'owner_id', 'draft_id')
          )
    ) THEN
        ALTER TABLE translations
            ADD CONSTRAINT translations_work_chapter_owner_draft_key
            UNIQUE (work_chapter_id, owner_id, draft_id);
    END IF;
END $$;

UPDATE meta SET value='28' WHERE key='schema_version';

COMMIT;
