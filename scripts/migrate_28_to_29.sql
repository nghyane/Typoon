-- Migration: schema 28 → 29.
--
-- Schema 29 collapses bubble_geometry's redundant box fields. Previously
-- the table carried:
--     polygon JSONB, fit_box JSONB, erase_box JSONB, text_box JSONB
-- After the Lens-derived container refactor, fit_box / text_box are
-- always identical to the polygon's bounding rect, and erase_box is a
-- debug-only by-product of the pixel-level erase mask (which lives in
-- the per-page MaskStore, not here).  Render only consumes polygon.
--
-- Forward-only: re-running after success is a no-op.
--
-- Usage:
--   PGPASSWORD=… psql -h … -U … -d typoon -f scripts/migrate_28_to_29.sql

BEGIN;

DO $$
DECLARE
    cur text;
BEGIN
    SELECT value INTO cur FROM meta WHERE key='schema_version';
    IF cur IS DISTINCT FROM '28' AND cur IS DISTINCT FROM '29' THEN
        RAISE EXCEPTION
            'expected schema_version=28 (or already 29), found %; abort migration',
            cur;
    END IF;
END$$;

ALTER TABLE bubble_geometry DROP COLUMN IF EXISTS fit_box;
ALTER TABLE bubble_geometry DROP COLUMN IF EXISTS erase_box;
ALTER TABLE bubble_geometry DROP COLUMN IF EXISTS text_box;

UPDATE meta SET value='29' WHERE key='schema_version';

COMMIT;
