-- Migration: schema 26 → 27.
--
-- Schema 27 drops `library_entries.title` and `library_entries.cover_url`;
-- display fields now live on the attached materials and resolve
-- server-side via `_resolve_work_display`.
--
-- For existing entries that have no material we can read title/cover
-- from yet (typically a "Tạo trống" Work where the user never
-- uploaded), seed a placeholder `origin='upload'` material from the
-- entry's stored snapshot. The placeholder honours the UNIQUE
-- `(imported_by, work_id) WHERE origin='upload'` invariant — we only
-- create one if the same user doesn't already own an upload material
-- on this Work.
--
-- Idempotent: re-running after success is a no-op (selects find no
-- gap; meta version is already 27).
--
-- Usage:
--   PGPASSWORD=… psql -h … -U … -d typoon -f scripts/migrate_26_to_27.sql

BEGIN;

-- Refuse to run if the DB is on the wrong baseline. Bail out loud
-- rather than silently dropping cols on an unrelated schema.
DO $$
DECLARE
    cur TEXT;
BEGIN
    SELECT value INTO cur FROM meta WHERE key='schema_version';
    IF cur IS DISTINCT FROM '26' THEN
        RAISE EXCEPTION
            'expected schema_version=26, found %; abort migration',
            COALESCE(cur, '<missing>');
    END IF;
END $$;

-- Backfill placeholder upload materials for any entry whose Work
-- has no material the resolver can read title/cover from. We pull
-- title + cover from the entry snapshot itself — the only place the
-- value still lives at this point.
--
-- `NOT EXISTS` is per (user, work) to honour the partial UNIQUE
-- index; users whose Work already carries a sibling (source / ext /
-- upload-by-someone-else) plus their own upload material are skipped.
INSERT INTO materials (imported_by, origin, work_id, title, cover_url, languages)
SELECT
    le.user_id,
    'upload',
    le.work_id,
    le.title,
    le.cover_url,
    ARRAY['vi']::text[]
FROM library_entries le
WHERE NOT EXISTS (
    SELECT 1
    FROM   materials m
    WHERE  m.work_id = le.work_id
      AND  m.imported_by = le.user_id
      AND  m.origin = 'upload'
)
  AND NOT EXISTS (
    -- And the Work has no OTHER material the resolver could already
    -- use (source / extension / someone-else's upload). If it does,
    -- this user's library card already renders correctly without a
    -- per-user placeholder.
    SELECT 1
    FROM   materials m
    WHERE  m.work_id = le.work_id
);

-- Drop the now-redundant cache columns. Schema 27 onwards reads
-- title + cover from the attached materials at request time.
ALTER TABLE library_entries DROP COLUMN title;
ALTER TABLE library_entries DROP COLUMN cover_url;

UPDATE meta SET value='27' WHERE key='schema_version';

COMMIT;
