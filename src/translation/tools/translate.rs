use crate::llm::ToolDef;

#[derive(serde::Deserialize)]
pub struct TranslationItem {
    pub id: String,
    pub translated_text: String,
}

#[derive(serde::Deserialize)]
pub struct Args {
    pub translations: Vec<TranslationItem>,
}

pub fn def() -> ToolDef {
    ToolDef::new(
        "translate",
        "Submit translations for one or more bubbles in a single call.\n\n\
            Behavior:\n\
            - Each item: id (must match a required bubble ID), translated_text (final localized text).\n\
            - Submit as many bubbles as possible per call to minimize round-trips.\n\
            - To revise: include the same id again (latest wins).\n\
            - Do not skip any listed bubble, even if short or SFX.",
        serde_json::json!({
            "type": "object",
            "required": ["translations"],
            "additionalProperties": false,
            "properties": {
                "translations": {
                    "type": "array",
                    "description": "Array of bubble translations",
                    "items": {
                        "type": "object",
                        "required": ["id", "translated_text"],
                        "additionalProperties": false,
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "Bubble ID exactly as listed (e.g. b0, p0_b1)"
                            },
                            "translated_text": {
                                "type": "string",
                                "description": "Final translated text — no notes or explanations"
                            }
                        }
                    }
                }
            }
        }),
    )
    .strict()
}
