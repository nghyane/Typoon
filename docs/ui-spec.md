# Typoon Web UI Spec

## App Overview

**Typoon** — local manga/manhwa translation tool with web UI.
Scans speech bubbles, translates via LLM, re-renders with target language text.

**Platform:** Web app (localhost), single user
**Style:** Agent's choice — design as you see fit for a manga translation tool.

---

## Screen 1 — Dashboard (Project List)

**Purpose:** Overview of all manga projects and their translation progress.

**Layout:**
- Top bar: app name "Typoon" left, worker status indicator right, "Add Project" button
- Main area: grid of project cards (2-3 columns)

**Worker Status Indicator (top right):**
- Green dot + "Workers running" when `typoon work` is active
- Gray dot + "Workers stopped" when idle

**Project Card:**
- Cover image placeholder (manga panel thumbnail if available, else gradient placeholder)
- Title (bold, white)
- Source → Target language badge e.g. "KO → VI"
- Progress bar: filled green portion = done chapters / total
- Stats row: "12 / 20 chapters" + "3 running"
- Status badges: counts of done / running / error chapters

**Actions per card:**
- Click → go to Project Detail screen
- "Redo" button (icon only, destructive confirmation dialog)
- Context menu: Rename, Delete

**Empty state:**
- Centered illustration
- Text: "No projects yet"
- Button: "Import folder" / "Pull from URL"

---

## Screen 2 — Project Detail

**Purpose:** Per-chapter status, trigger actions, view rendered output.

**Layout:**
- Breadcrumb: Dashboard > Project Title
- Project header: title, language pair, source URL if any, "Redo All" button
- Chapter table below

**Chapter Table columns:**
- # (chapter number, e.g. "ch1", "ch2.5")
- Status badge: Done / Scanning / Translating / Rendering / Error / Idle
- Stage: current stage label (visible when not done/idle)
- Pages: rendered page count when done
- Actions: "View" (if done), "Redo" (icon)

**Status badge designs:**
- Done: solid green pill "✓ Done"
- Scanning: yellow pill with spinner "⟳ Scanning"
- Translating: yellow pill with spinner "⟳ Translating"
- Rendering: yellow pill with spinner "⟳ Rendering"
- Error: red pill "✗ Error" — hover shows error message tooltip
- Idle: gray outline pill "○ Idle"

**Error row:**
- Row highlighted with subtle red left border
- Error stage shown in Stage column
- Hover tooltip shows full error message

**Actions bar above table:**
- "Add Chapter" dropdown: "Import folder" / "Pull from URL"
- "Redo All" button (red, confirmation required)
- Filter: All / Done / Running / Error / Idle

---

## Screen 3 — Chapter Viewer

**Purpose:** Browse rendered pages of a completed chapter.

**Layout:**
- Breadcrumb: Dashboard > Project > Chapter
- Page strip (horizontal scroll) at top — thumbnail previews
- Main viewer: large rendered page image centered
- Navigation: prev/next arrows, page counter "3 / 18"

**Page strip:**
- Horizontal scrollable row of page thumbnails
- Active page highlighted with blue border
- Click to jump to page

**Main viewer:**
- Single page image, fit to viewport height
- Click to zoom (lightbox overlay)
- Keyboard navigation: arrow keys

---

## Screen 4 — Add Project Modal

**Purpose:** Import local folder or pull from remote URL.

**Layout:** Modal dialog, centered, dark overlay background.

**Two tabs:**

**Tab 1 — Import Folder:**
- Folder path input (or drag-and-drop zone)
- Project name input (auto-filled from folder name)
- Source language select (KO / JA / ZH / EN)
- Target language select (VI / EN / etc.)
- "Import" button

**Tab 2 — Pull from URL:**
- URL input (e.g. comix.to manga URL)
- Discover button → loads chapter list below
- Chapter range selector: "From ch__ to ch__" or multi-select list
- Target language select
- "Pull" button

**After submit:**
- Modal closes
- Toast notification: "X chapters enqueued. Run workers to process."
- New project card appears in dashboard (idle state)

---

## Screen 5 — Redo Confirmation Dialog

**Purpose:** Confirm destructive redo action.

**Layout:** Small modal, centered.

**Content:**
- Warning icon (red)
- Title: "Reset and re-run?"
- Description: "This will delete all scan results, translations, and rendered pages for [N] chapter(s). Source images will be kept."
- Optional chapter range (if triggered from project-level)
- Two buttons: "Cancel" (secondary) / "Reset & Re-run" (red, primary)

---

## Screen 6 — Worker Status Panel (slide-over or bottom bar)

**Purpose:** Live view of what workers are currently doing.

**Trigger:** Click worker status indicator in top bar.

**Layout:** Slide-over panel from right, or expandable bottom bar.

**Content:**
- Worker status: Running / Stopped
- Active tasks list:
  - Each row: project name + chapter + stage + elapsed time
  - e.g. "Solo Leveling  ch5  Translating  0:42"
  - e.g. "Tycoon  ch2  Scanning  0:08"
- Pending queue count: "12 tasks pending"
- Failed tasks (attempts >= 3): listed with error summary
- "Start Workers" button if stopped

**Polling:** refreshes every 3 seconds automatically.

---

## Global Components

**Toast notifications:**
- Bottom-right corner
- Success (green), Error (red), Info (gray)
- Auto-dismiss after 4 seconds

**Top bar:**
- Always visible
- Left: "Typoon" wordmark
- Right: worker status dot + label, settings icon

**Empty/loading states:**
- Skeleton loaders for tables and cards while fetching
- Spinner for in-progress actions

---

## Image Gen Prompts (per screen)

### Prompt — Screen 1 Dashboard
> Web app dashboard for a manga translation tool called Typoon. Grid of project cards with cover image placeholders, project titles, language badges like "KO→VI", progress bars, chapter count stats. Top navigation bar with app name and worker status indicator. 2-3 column card grid.

### Prompt — Screen 2 Project Detail
> Project detail view for a manga translation web app. Breadcrumb navigation at top. Large project title with language pair badge. Data table with columns: chapter, status badge, stage, pages, actions. Status badges show Done / Translating / Error / Idle states. One row shows an error state. Clean table design.

### Prompt — Screen 3 Chapter Viewer
> Manga reader web app screen. Horizontal thumbnail strip at top with small page previews, one highlighted as active. Main area shows a large manga page image with translated speech bubbles. Left and right navigation arrows. Page counter "3 / 18" below. Full-height page view.

### Prompt — Screen 4 Add Project Modal
> Modal dialog for adding a new manga project. Two tabs: "Import Folder" and "Pull from URL". Pull from URL tab is active: URL text input, language selector "KO → VI", chapter list with checkboxes showing chapters 1-10, confirm button. Clean form layout.

### Prompt — Screen 5 Worker Status Panel
> Slide-over panel showing worker status for a translation pipeline. Header "Worker Status" with active indicator. List of active tasks showing manga title, chapter number, current stage (Scanning/Translating/Rendering), elapsed time. Pending queue count below. Start/Stop workers button at bottom.
