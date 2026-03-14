use super::{TranslateContext, TranslateRequest};

/// Composable prompt builder for translation agent system/user prompts.
///
/// Each section is a separate method for testability and future provider override.
pub struct PromptBuilder<'a> {
    req: &'a TranslateRequest,
    ctx: &'a TranslateContext<'a>,
}

impl<'a> PromptBuilder<'a> {
    pub fn new(req: &'a TranslateRequest, ctx: &'a TranslateContext<'a>) -> Self {
        Self { req, ctx }
    }

    // ═══════════════════════════════════════════════════════════════════
    // System prompt sections
    // ═══════════════════════════════════════════════════════════════════

    pub fn system_prompt(&self) -> String {
        let mut sys = String::new();
        sys.push_str(&self.role_header());
        sys.push_str(&self.output_contract());
        sys.push_str(&self.workflow());
        sys.push_str(self.source_policy());
        sys.push_str(self.target_policy());
        sys
    }

    pub fn role_header(&self) -> String {
        let comic_type = match self.req.source_lang.as_str() {
            "ja" => "manga",
            "ko" => "manhwa",
            "zh" => "manhua",
            _ => "comic",
        };

        format!(
            "You are ComicScan's tool-only translation agent for {comic_type} ({} → {}).\n\n",
            self.req.source_lang, self.req.target_lang
        )
    }

    pub fn output_contract(&self) -> String {
        "\
## Output contract
- Your ONLY valid outputs are function/tool calls. No prose, no markdown, no explanations.
- Every assistant message MUST contain one or more tool calls.
- Do NOT send a single tool call when you have multiple ready; combine them in one message.
- Do NOT ask questions in prose. Call a tool (view_page, search_glossary) instead.\n\n"
            .to_string()
    }

    pub fn workflow(&self) -> String {
        let research_policy = if self.req.target_lang == "vi" {
            "\
2. Research (REQUIRED for Vietnamese — 1 message):
   - view_page(): view 2-3 key pages to determine speaker age, relationships, and setting
     for correct xưng hô. Batch all view_page() calls in one message.
   - get_context(): call if a character, relationship, or term from earlier chapters is unclear.
     Ask a specific question mentioning the names/terms you need, e.g.
     \"What are the established Vietnamese names and relationships for Max and Joy?\"
     Max 1 call. The sub-agent searches prior translations AND chapter notes
     (character, relationship, event, setting notes saved by add_note in past chapters).
   - search_glossary(): only for terms NOT in the provided glossary above."
        } else {
            "\
2. Research (optional — at most 1 message):
   - get_context(): only if a character, relationship, or term from earlier chapters is unclear.
     Ask a specific question mentioning the names/terms you need. Max 1 call.
   - view_page(): only if text alone cannot resolve speaker identity or relationship.
     Batch all view_page() calls in one message.
   - search_glossary(): only for terms NOT in the provided glossary above."
        };

        format!("\
## Workflow
1. Analyze: read all OCR text to understand dialogue flow, characters, tone, and relationships.
   Infer speakers from names, honorifics, speech register, terms of address.
{research_policy}
3. Translate ALL bubbles: call translate() with ALL bubble IDs in as few calls as possible.
   - Put 30+ bubbles per translate() call. Do NOT split into small groups.
   - Include EVERY required ID — never skip, even for short text or SFX.
   - translated_text = final localized text only, no notes or explanations.
   - To revise: call translate() again for the same ID (latest wins).
4. Finish: in the same message as your final translate() calls, call add_note() to record
   observations for future chapters. Save each distinct observation as a separate note:
   - character: new character introductions or descriptions
   - relationship: who knows whom, power dynamics, xưng hô pairs
   - event: significant plot points
   - setting: locations, workplace, school, etc.

IMPORTANT — efficiency:
- Target: 2-4 total messages. Batch aggressively.
- Do NOT trickle translations across many turns.
- Combine tool calls in a single message when possible.\n\n")
    }

    /// Source-language notes for the system prompt.
    pub fn source_policy(&self) -> &'static str {
        match self.req.source_lang.as_str() {
            "ja" => "\
## Source: Japanese manga
- Reading order: right-to-left. Bubble order follows this.
- Japanese often omits subject — infer from context/visuals.
- Keigo levels signal character relationships — reflect in translation register.
- Honorifics (-san, -kun, -chan, -sama, -sensei, -senpai): preserve or adapt per target.
- SFX onomatopoeia (ドキドキ, ゴゴゴ): translate or transliterate per target norms.\n\n",

            "ko" => "\
## Source: Korean manhwa
- Reading order: left-to-right. Webtoon format (vertical scroll).
- Speech levels (존댓말/반말) signal power dynamics — reflect in translation register.
- Honorifics (-님, -씨, 선배, 형/오빠/언니/누나): adapt per target conventions.
- Onomatopoeia (쿵, 두근두근): translate or adapt.\n\n",

            "zh" => "\
## Source: Chinese manhua
- Reading order: left-to-right (modern).
- Chengyu (成语): translate meaning, not literally.
- Cultivation/xianxia terms (修炼, 气, 丹田, 境界): use glossary if available.
- Honorifics (先生, 大人, 小姐): contextual — adapt per target.\n\n",

            _ => "",
        }
    }

    /// Target-language notes for the system prompt.
    pub fn target_policy(&self) -> &'static str {
        match self.req.target_lang.as_str() {
            "vi" => "\
## Target: Vietnamese
- Xưng hô is chosen from the speaker–addressee RELATIONSHIP, not literal source pronouns.
- Infer relationships from text cues first: names, honorifics, speech register, terms of address.
- If you viewed a page image, use visual cues (age, clothing, setting) to refine classification: \
child, teen/student, adult peer, younger→older, subordinate→superior, \
civilian→officer, employee→manager, teacher↔student, family, hostile speaker, etc.
- Adults in uniform / workplace / official / police / military / medical / office scenes: \
default to tôi/anh/chị/em/ông/bà as appropriate. NEVER cậu/tớ for adults.
- cậu/tớ: ONLY for young peers or intimate youthful speech clearly from context.
- Teacher↔student: thầy/cô ↔ em. Adult staff in school/work: adult pronouns, not cậu/tớ.
- Rough/aggressive: tao/mày. Formal/respectful: tôi/ngài.
- Keep xưng hô consistent within the same scene unless the relationship shifts.
- If you viewed a page and the image contradicts a text-only assumption, trust the image.
- Prefer natural Vietnamese; use Hán-Việt for formal/cultivation terms when natural.\n\n",

            "en" => "\
## Target: English
- Natural, fluent English. Avoid translationese.
- Preserve character voice: tough characters sound tough, polite ones sound polite.
- Honorifics: keep -san/-kun/-senpai for manga audience, or naturalize — be consistent.
- SFX: use English equivalents where natural (THUD, CRACK), keep original for iconic ones.\n\n",

            _ => "",
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // User prompt sections
    // ═══════════════════════════════════════════════════════════════════

    pub fn user_prompt(&self) -> String {
        let mut prompt = String::new();
        prompt.push_str(&self.task_header());
        prompt.push_str(&self.context_summary());
        prompt.push_str(&self.glossary_section());
        prompt.push_str(&self.notes_section());
        prompt.push_str(&self.previous_translations());
        prompt.push_str(&self.workflow_hint());
        prompt.push_str(&self.bubble_list());
        prompt
    }

    pub fn task_header(&self) -> String {
        let all_ids: Vec<&str> = self.req.all_bubbles().map(|b| b.id.as_str()).collect();
        let mut s = format!(
            "Task: Translate every listed bubble from {} to {} using tool calls only.\n\n",
            self.req.source_lang, self.req.target_lang
        );
        s.push_str(&format!(
            "Completion requirements:\n- Total bubbles: {}\n- Required IDs: {}\n- Each ID must get a translate() call.\n\n",
            all_ids.len(),
            all_ids.join(", ")
        ));
        s
    }

    pub fn context_summary(&self) -> String {
        let mut s = String::from("Available context:\n");
        let is_single_page = self.req.pages.len() == 1;
        if is_single_page && !self.ctx.page_images.is_empty() {
            s.push_str("- Page image: attached below (inspect directly)\n");
        } else if !self.ctx.page_images.is_empty() {
            s.push_str(&format!(
                "- Page images: {} pages available via view_page(index)\n",
                self.ctx.page_images.len()
            ));
        }
        if self.ctx.glossary.is_some() {
            s.push_str("- Glossary search: available via search_glossary()\n");
        }
        if self.ctx.context_store.is_some() && self.ctx.context_agent.is_some() {
            s.push_str("- Prior chapter context: available via get_context(question)\n");
        }
        s.push('\n');
        s
    }

    pub fn glossary_section(&self) -> String {
        if self.req.glossary.is_empty() {
            return String::new();
        }
        let mut s = String::from("Glossary (canon — use these):\n");
        for entry in &self.req.glossary {
            s.push_str(&format!("  {} → {}", entry.source_term, entry.target_term));
            if let Some(notes) = &entry.notes {
                s.push_str(&format!("  ({})", notes));
            }
            s.push('\n');
        }
        s.push('\n');
        s
    }

    pub fn notes_section(&self) -> String {
        if self.req.notes.is_empty() {
            return String::new();
        }
        let mut s = String::from("Known characters and relationships from previous chapters:\n");
        for n in &self.req.notes {
            s.push_str(&format!("  [{}] {}\n", n.note_type, n.content));
        }
        s.push('\n');
        s
    }

    pub fn previous_translations(&self) -> String {
        if self.req.context.is_empty() {
            return String::new();
        }
        let mut s = String::from("Previous translations for context:\n");
        for c in &self.req.context {
            s.push_str(&format!("  {} → {}\n", c.source_text, c.translated_text));
        }
        s.push('\n');
        s
    }

    pub fn workflow_hint(&self) -> String {
        String::new()
    }

    pub fn bubble_list(&self) -> String {
        let mut s = String::new();
        for page in &self.req.pages {
            if self.req.pages.len() > 1 {
                let img_note = if page.page_index < self.ctx.page_images.len() {
                    "available via view_page"
                } else {
                    "no image"
                };
                s.push_str(&format!("── Page {} ({}) ──\n", page.page_index, img_note));
            }
            for bubble in &page.bubbles {
                s.push_str(&format!("[{}] \"{}\"\n", bubble.id, bubble.source_text));
            }
            s.push('\n');
        }
        s
    }
}
