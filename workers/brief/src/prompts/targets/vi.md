## Target: Vietnamese

---

### Context agent policy

This section is injected into the context agent (storyboard vision pass).
It tells the agent how to make decisions for Vietnamese output.

**Naming by source script — Chinese (汉字)**

Apply Hán-Việt ONLY to genuine proper nouns and canonical system terms.
Do NOT apply Hán-Việt to descriptive compound words that have a natural
Vietnamese equivalent — translate those by meaning instead.

**Use Hán-Việt for:**
- Personal names: 叶川 → Diệp Xuyên, 周妍 → Chu Nghiên, 萧炎 → Tiêu Viêm.
- Sect / faction / place names: 青云宗 → Thanh Vân Tông, 天都城 → Thiên Đô Thành.
- Cultivation realm titles that are the world's canonical tier names:
  炼气期 → Luyện Khí Kỳ, 大师境 → Đại Sư Cảnh, 元婴期 → Nguyên Anh Kỳ.
- Technique / skill names that are proper titles: 天罡剑法 → Thiên Cương Kiếm Pháp.
- Titles / honorifics: 宗主 → Tông Chủ, 长老 → Trưởng Lão, 师尊 → Sư Tôn.

**Translate by meaning (NOT Hán-Việt) for:**
- Descriptive item names where the meaning is the point:
  杀猪刀 → dao mổ lợn (not "Sát Trư Đao").
  亡灵兵 → lính bất tử / lính vong linh (not "Vong Linh Binh").
  飞剑 → kiếm bay (not "Phi Kiếm") — unless it is a named technique.
- Common nouns that happen to be written in hanzi: translate normally.
- Any word that a Vietnamese reader would immediately understand better
  as a translated phrase than as a Hán-Việt compound.

**Rule of thumb:** if a Vietnamese reader hearing the Hán-Việt form would
ask "what does that mean?", translate by meaning instead. If it sounds
like a title or world-specific term they would accept as-is, use Hán-Việt.

**Other source scripts**
- Japanese (kana/kanji): keep Hepburn romaji as-is.
  田中ゆき → Tanaka Yuki. Do NOT Sino-Vietnamize Japanese kanji.
- Korean (hangul): keep Revised Romanization.
  준호 → Jun-ho. 민지 → Min-ji.
- Latin-script source (en/es/pt/…): keep original spelling.
- When a glossary entry already exists (prior chapter, series memory):
  use it verbatim. Consistency beats re-derivation.

**Honorific suffix integration**

Never leave a raw honorific suffix attached to a name in Vietnamese output.
Integrate into xưng hô:

- zh 姐/chị → chị + given name (周妍姐 → chị Nghiên)
- zh 哥/anh → anh + given name
- zh 弟/妹 → em + given name
- zh 师兄/师姐 → sư huynh / sư tỷ (+ name when natural)
- zh 前辈 → tiền bối; 晚辈 → vãn bối
- ja -san/-sama → anh/chị/ông/bà per relationship
- ja -kun/-chan → em or name-only per relationship
- ja -senpai → senpai (keep) or anh/chị
- ko -hyung/-oppa → anh; -noona/-unnie → chị; -sunbae → đàn anh/đàn chị

**ADDRESS decisions**

For every confirmed speaker→listener pair, emit one ADDRESS line.
Choose the xưng hô pair that fits relationship + scene tone:

- Hostile / contemptuous: tao ↔ mày (only with strong visual evidence)
- Superior → subordinate formal: tôi / ta ↔ ngươi / anh / chị
- Classical / xianxia elevated: ta ↔ ngươi
- Adult peers / workplace: tôi ↔ anh / chị
- Younger → older (respectful): em ↔ anh / chị / chú / cô / bác
- Older → younger: anh / chị ↔ em
- Parent → child: mẹ / ba / bố ↔ con
- Teacher → student: thầy / cô ↔ em
- Close young peers: cậu ↔ tớ (ONLY clearly youthful + casual)
- Romantic young adults: anh ↔ em
- Do not default to "bạn" for dramatic dialogue.
- When uncertain, omit the pair rather than guess wrong.

**BRIEF language**

Write the BRIEF section in Vietnamese.

---

### Translator mechanics

This section is injected into the translator's system prompt.
The chapter brief (glossary, address table, character voices) is the
primary guide. These rules are the fallback when the brief is absent
or incomplete.

- Output must contain zero Han characters. 汉字 renders as blank space.
  If brief glossary is missing an entry: for proper names and realm titles
  use Hán-Việt; for descriptive compound words translate by meaning.
  Never leave source hanzi in the output.
- Particles (à, nhé, thôi, mà, đấy, chứ, nhỉ, ạ) only when natural
  and character-appropriate. Never force them.
- Keep lines compact for speech bubbles. Split long sentences naturally.
- **Character budget**: each bubble header includes `chars=N` — the maximum
  number of characters (including spaces) that fit in the bubble at a
  readable font size. Your translation MUST stay within this budget.
  If the literal translation exceeds it, cut ruthlessly: drop redundant
  phrases, use shorter synonyms, omit filler. Illegible tiny font is
  worse than a slightly loose translation.
- Natural spoken Vietnamese. Not literal source-language word order.
- When brief has no ADDRESS entry for a pair and speaker is confirmed:
  use neutral adult phrasing (tôi / anh / chị) as safe default.
- When speaker is unknown: omit pronouns or rephrase to avoid them.
- SFX: punchy Vietnamese onomatopoeia. No Han characters in SFX either.
