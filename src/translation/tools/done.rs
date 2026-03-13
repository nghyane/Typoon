use crate::agent;

pub fn def() -> agent::ToolDef {
    agent::ToolDef::new(
        "done",
        "Signal that the translation job is complete.\n\n\
            Behavior:\n\
            - Call ONLY after every required bubble ID has at least one translate() call.\n\
            - Prefer sending final translate() calls and done() in the same message.\n\
            - Before calling, verify coverage against the required ID list.\n\n\
            When to use: once, when nothing remains untranslated.\n\
            When NOT to use: do not call after partial work or after only viewing/searching.",
        serde_json::json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {}
        }),
    )
    .strict()
}
