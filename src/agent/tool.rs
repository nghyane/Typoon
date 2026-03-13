/// Provider-agnostic tool definitions.
///
/// `parameters` stays as `serde_json::Value` (JSON Schema) because fully typing
/// JSON Schema would be over-engineering. The key abstraction is that `ToolDef`
/// is provider-agnostic — each provider wraps it differently:
/// - OpenAI: `{"type":"function","function":{"name":..,"parameters":..}}`
/// - Anthropic: `{"name":..,"input_schema":..}`

/// A tool definition exposed to the LLM.
#[derive(Debug, Clone)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub strict: bool,
    pub parameters: serde_json::Value,
}

impl ToolDef {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            strict: false,
            parameters,
        }
    }

    pub fn strict(mut self) -> Self {
        self.strict = true;
        self
    }
}

/// Provider-agnostic tool response returned by handlers.
#[derive(Debug, Clone)]
pub enum ToolResponse {
    /// Plain text response.
    Text(String),
    /// Image content response (text label + data URI).
    ImageContent { text: String, data_uri: String },
}
