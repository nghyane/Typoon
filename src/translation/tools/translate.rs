use crate::agent;

#[derive(serde::Deserialize)]
pub struct Args {
    pub id: String,
    pub source_text: String,
    pub translated_text: String,
}

pub fn def() -> agent::ToolDef {
    agent::ToolDef::new(
        "translate",
        "Submit the final translation for exactly one bubble.\n\n\
            Behavior:\n\
            - Call once for every required bubble ID. Batch many translate() calls in one message.\n\
            - id must exactly match a required bubble ID from the task.\n\
            - source_text = cleaned/corrected OCR text for that bubble.\n\
            - translated_text = final localized text only — no notes or explanations.\n\
            - To revise: call translate() again with the same id (latest wins).\n\n\
            When to use: whenever you are ready to submit a bubble translation.\n\
            When NOT to use: do not skip any listed bubble, even if short or SFX.\n\n\
            Examples:\n\
            - {\"id\":\"b0\",\"source_text\":\"おはよう\",\"translated_text\":\"Chào buổi sáng.\"}\n\
            - {\"id\":\"p0_b3\",\"source_text\":\"What?!\",\"translated_text\":\"Cái gì?!\"}",
        serde_json::json!({
            "type": "object",
            "required": ["id", "source_text", "translated_text"],
            "additionalProperties": false,
            "properties": {
                "id": {
                    "type": "string",
                    "description": "Bubble ID exactly as listed in the task (e.g. b0, p0_b1)"
                },
                "source_text": {
                    "type": "string",
                    "description": "Cleaned/corrected source text for this bubble"
                },
                "translated_text": {
                    "type": "string",
                    "description": "Final translated text — no notes or explanations"
                }
            }
        }),
    )
    .strict()
}
