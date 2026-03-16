/// Generic agent framework — trait + reusable loop.
///
/// Each agent implements `Agent` with its own tools, dispatch logic, and output type.
/// The `run()` function drives any agent through the standard tool-calling loop.
pub mod context;
pub mod knowledge;
pub mod translation;

use std::future::Future;

use anyhow::Result;

use crate::llm::{ContentPart, Message, Provider, ToolCallMsg, ToolDef, ToolResponse};

/// Agent trait — defines prompt, tools, dispatch, and termination.
///
/// Each agent holds its own mutable state and borrowed resources.
/// The generic `run()` loop drives any `Agent` impl without per-agent loop code.
///
/// `dispatch` returns `impl Future + Send` so agents can be spawned on tokio.
pub trait Agent: Send {
    type Output;

    /// Agent identifier for structured logging.
    fn name(&self) -> &'static str;

    fn system_prompt(&self) -> String;
    fn user_message(&self) -> Message;
    fn tools(&self) -> Vec<ToolDef>;

    /// Handle a single tool call. Mutate internal state as needed.
    fn dispatch<'a>(
        &'a mut self,
        call: &'a ToolCallMsg,
    ) -> impl Future<Output = ToolResponse> + Send + 'a;

    /// Called when the provider responds with text and no tool calls.
    /// Context agent uses this to capture the final answer.
    fn on_text(&mut self, _text: Option<&str>) {}

    /// Should the loop stop early (e.g., all bubbles translated)?
    fn is_done(&self) -> bool {
        false
    }

    /// If the provider stopped calling tools but the agent isn't done,
    /// return a retry prompt to push the conversation forward.
    /// Return `None` to accept current state and stop.
    fn retry_prompt(&mut self) -> Option<String> {
        None
    }

    /// Extract final output. Consumes the agent.
    fn into_output(self) -> Self::Output;
}

/// Internal safety cap — prevents infinite loops from LLM bugs.
/// Normal agents terminate via `is_done()` or no tool calls well before this.
const SAFETY_MAX_TURNS: usize = 50;

/// Generic agent loop — written once, drives any `Agent` impl.
pub async fn run<A: Agent>(provider: &dyn Provider, mut agent: A) -> Result<A::Output> {
    let max_turns = SAFETY_MAX_TURNS;
    let agent_name = agent.name();
    let mut messages = vec![Message::system(agent.system_prompt()), agent.user_message()];
    let tools = agent.tools();

    for turn in 0..max_turns {
        tracing::info!("[{agent_name}] turn {}", turn + 1);
        let t = std::time::Instant::now();
        let resp = provider.call(&messages, &tools).await?;
        tracing::info!(
            "[{agent_name}] LLM responded in {:.1}s ({} tool calls)",
            t.elapsed().as_secs_f64(),
            resp.tool_calls.len(),
        );

        if resp.tool_calls.is_empty() {
            agent.on_text(resp.text.as_deref());
            if agent.is_done() {
                break;
            }
            match agent.retry_prompt() {
                Some(prompt) => {
                    tracing::info!("[{agent_name}] retrying...");
                    messages.push(Message::user_text(prompt));
                    continue;
                }
                None => break,
            }
        }

        let tool_calls = resp.tool_calls;
        messages.push(Message::Assistant {
            text: resp.text,
            tool_calls: tool_calls.clone(),
        });

        for tc in &tool_calls {
            let result = agent.dispatch(tc).await;
            messages.push(tool_response_to_message(&tc.id, result));
        }

        if agent.is_done() {
            tracing::info!("[{agent_name}] done after {} turns", turn + 1);
            break;
        }
    }

    Ok(agent.into_output())
}

fn tool_response_to_message(tool_call_id: &str, response: ToolResponse) -> Message {
    match response {
        ToolResponse::Text(text) => Message::tool_result_text(tool_call_id, text),
        ToolResponse::ImageContent { text, data_uri } => Message::tool_result_parts(
            tool_call_id,
            vec![ContentPart::Text(text), ContentPart::Image { data_uri }],
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::provider::CallResponse;
    use std::pin::Pin;
    use std::sync::Mutex;

    /// Mock provider that returns pre-programmed responses.
    struct MockProvider {
        responses: Mutex<Vec<CallResponse>>,
    }

    impl MockProvider {
        fn new(responses: Vec<CallResponse>) -> Self {
            Self {
                responses: Mutex::new(responses),
            }
        }
    }

    impl Provider for MockProvider {
        fn call<'a>(
            &'a self,
            _messages: &'a [Message],
            _tools: &'a [ToolDef],
        ) -> Pin<Box<dyn std::future::Future<Output = Result<CallResponse>> + Send + 'a>> {
            Box::pin(async {
                let mut responses = self.responses.lock().unwrap();
                if responses.is_empty() {
                    Ok(CallResponse {
                        tool_calls: vec![],
                        text: None,
                    })
                } else {
                    Ok(responses.remove(0))
                }
            })
        }
    }

    // ── Collecting agent: dispatches "add" tool, counts items ──

    struct CollectAgent {
        items: Vec<String>,
        target: usize,
    }

    impl Agent for CollectAgent {
        type Output = Vec<String>;

        fn name(&self) -> &'static str {
            "test"
        }
        fn system_prompt(&self) -> String {
            "test".into()
        }
        fn user_message(&self) -> Message {
            Message::user_text("go")
        }
        fn tools(&self) -> Vec<ToolDef> {
            vec![ToolDef::new("add", "add item", serde_json::json!({}))]
        }

        fn dispatch<'a>(
            &'a mut self,
            call: &'a ToolCallMsg,
        ) -> impl Future<Output = ToolResponse> + Send + 'a {
            async move {
                if call.name == "add" {
                    self.items.push(call.arguments.clone());
                }
                ToolResponse::Text("ok".into())
            }
        }

        fn is_done(&self) -> bool {
            self.items.len() >= self.target
        }

        fn into_output(self) -> Vec<String> {
            self.items
        }
    }

    #[tokio::test]
    async fn test_agent_loop_basic() {
        let provider = MockProvider::new(vec![
            CallResponse {
                tool_calls: vec![
                    ToolCallMsg {
                        id: "c1".into(),
                        name: "add".into(),
                        arguments: "item1".into(),
                    },
                    ToolCallMsg {
                        id: "c2".into(),
                        name: "add".into(),
                        arguments: "item2".into(),
                    },
                ],
                text: None,
            },
            CallResponse {
                tool_calls: vec![ToolCallMsg {
                    id: "c3".into(),
                    name: "add".into(),
                    arguments: "item3".into(),
                }],
                text: None,
            },
        ]);

        let agent = CollectAgent {
            items: vec![],
            target: 3,
        };
        let result = run(&provider, agent).await.unwrap();
        assert_eq!(result, vec!["item1", "item2", "item3"]);
    }

    #[tokio::test]
    async fn test_agent_loop_early_stop() {
        // Provider sends 2 items but target is 1 — loop should stop after first turn
        let provider = MockProvider::new(vec![CallResponse {
            tool_calls: vec![ToolCallMsg {
                id: "c1".into(),
                name: "add".into(),
                arguments: "only_one".into(),
            }],
            text: None,
        }]);

        let agent = CollectAgent {
            items: vec![],
            target: 1,
        };
        let result = run(&provider, agent).await.unwrap();
        assert_eq!(result, vec!["only_one"]);
    }

    #[tokio::test]
    async fn test_agent_loop_safety_cap() {
        // Provider always returns a tool call — loop should stop at SAFETY_MAX_TURNS
        let mut responses = Vec::new();
        for i in 0..SAFETY_MAX_TURNS + 10 {
            responses.push(CallResponse {
                tool_calls: vec![ToolCallMsg {
                    id: format!("c{i}"),
                    name: "add".into(),
                    arguments: format!("item{i}"),
                }],
                text: None,
            });
        }
        let provider = MockProvider::new(responses);

        let agent = CollectAgent {
            items: vec![],
            target: 999, // never satisfied
        };
        let result = run(&provider, agent).await.unwrap();
        assert_eq!(result.len(), SAFETY_MAX_TURNS);
    }

    // ── Text-capturing agent (like ContextAgent) ──

    struct TextAgent {
        answer: Option<String>,
    }

    impl Agent for TextAgent {
        type Output = String;

        fn name(&self) -> &'static str {
            "test"
        }
        fn system_prompt(&self) -> String {
            "answer".into()
        }
        fn user_message(&self) -> Message {
            Message::user_text("question")
        }
        fn tools(&self) -> Vec<ToolDef> {
            vec![ToolDef::new("search", "search", serde_json::json!({}))]
        }

        fn dispatch<'a>(
            &'a mut self,
            _call: &'a ToolCallMsg,
        ) -> impl Future<Output = ToolResponse> + Send + 'a {
            async { ToolResponse::Text("search results".into()) }
        }

        fn on_text(&mut self, text: Option<&str>) {
            self.answer = text.map(|s| s.to_string());
        }

        fn into_output(self) -> String {
            self.answer.unwrap_or_default()
        }
    }

    #[tokio::test]
    async fn test_agent_on_text() {
        let provider = MockProvider::new(vec![
            // Turn 1: tool call
            CallResponse {
                tool_calls: vec![ToolCallMsg {
                    id: "c1".into(),
                    name: "search".into(),
                    arguments: "{}".into(),
                }],
                text: None,
            },
            // Turn 2: text response (no tool calls) — final answer
            CallResponse {
                tool_calls: vec![],
                text: Some("The answer is 42".into()),
            },
        ]);

        let agent = TextAgent { answer: None };
        let result = run(&provider, agent).await.unwrap();
        assert_eq!(result, "The answer is 42");
    }

    // ── Retry agent ──

    struct RetryAgent {
        items: Vec<String>,
        retry_count: usize,
    }

    impl Agent for RetryAgent {
        type Output = Vec<String>;

        fn name(&self) -> &'static str {
            "test"
        }
        fn system_prompt(&self) -> String {
            "retry test".into()
        }
        fn user_message(&self) -> Message {
            Message::user_text("go")
        }
        fn tools(&self) -> Vec<ToolDef> {
            vec![ToolDef::new("add", "add", serde_json::json!({}))]
        }

        fn dispatch<'a>(
            &'a mut self,
            call: &'a ToolCallMsg,
        ) -> impl Future<Output = ToolResponse> + Send + 'a {
            async move {
                self.items.push(call.arguments.clone());
                ToolResponse::Text("ok".into())
            }
        }

        fn is_done(&self) -> bool {
            self.items.len() >= 2
        }

        fn retry_prompt(&mut self) -> Option<String> {
            if self.retry_count >= 2 {
                return None;
            }
            self.retry_count += 1;
            Some("please continue".into())
        }

        fn into_output(self) -> Vec<String> {
            self.items
        }
    }

    #[tokio::test]
    async fn test_agent_retry() {
        let provider = MockProvider::new(vec![
            // Turn 1: provider returns 1 item
            CallResponse {
                tool_calls: vec![ToolCallMsg {
                    id: "c1".into(),
                    name: "add".into(),
                    arguments: "first".into(),
                }],
                text: None,
            },
            // Turn 2: provider returns no tool calls — triggers retry
            CallResponse {
                tool_calls: vec![],
                text: Some("thinking...".into()),
            },
            // Turn 3: after retry prompt, provider returns second item
            CallResponse {
                tool_calls: vec![ToolCallMsg {
                    id: "c2".into(),
                    name: "add".into(),
                    arguments: "second".into(),
                }],
                text: None,
            },
        ]);

        let agent = RetryAgent {
            items: vec![],
            retry_count: 0,
        };
        let result = run(&provider, agent).await.unwrap();
        assert_eq!(result, vec!["first", "second"]);
    }
}
