-- +goose Up
CREATE TABLE app_settings (
  id         BOOLEAN PRIMARY KEY DEFAULT true CHECK (id),
  value      JSONB NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_by UUID REFERENCES users(id)
);

INSERT INTO app_settings (id, value)
VALUES (true, '{
  "sourceFetch": {
    "origins": [
      "https://927251094806098001.discordsays.com"
    ]
  },
  "pricing": {
    "translation": { "xuPerPage": 5 }
  },
  "features": {
    "browse": true,
    "translation": true
  }
}'::jsonb)
ON CONFLICT (id) DO NOTHING;

-- +goose Down
DROP TABLE IF EXISTS app_settings;
