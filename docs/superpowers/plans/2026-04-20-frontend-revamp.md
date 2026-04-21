# Frontend Revamp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign `static/index.html` with FREED brand colors, adaptive light/dark theme, collapsible sidebar, top bar, and Modern SaaS component system — without touching any JavaScript logic.

**Architecture:** Single-file HTML overhaul. All CSS variables replaced with FREED brand tokens. Layout gains a 56px top bar and a collapsible sidebar (220px ↔ 56px). A minimal theme + sidebar JS block is appended (UI state only — no API/logic changes). Hardcoded dark-mode rgba values in JS inline style strings are replaced with CSS variable references for light-mode readability.

**Tech Stack:** Vanilla HTML/CSS/JS · Inter font (already loaded) · Chart.js (already loaded) · CSS custom properties for theming

**Design spec:** `docs/superpowers/specs/2026-04-20-frontend-revamp-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `static/index.html` | Modify | All CSS, HTML structure, theme/sidebar JS |

---

## Task 1: Replace CSS Token System

**Files:**
- Modify: `static/index.html:13-26` (`:root` block)

Replace the entire `:root` block (lines 13–26) with the FREED brand token system. Also add legacy aliases so existing JS inline styles using `--border-color`, `--text-color`, `--bg-surface` continue to work. Add `body.dark {}` block immediately after.

- [ ] **Step 1: Replace `:root` block**

Find and replace the entire existing `:root { ... }` block with:

```css
:root {
    /* FREED Brand */
    --primary: #02416E;
    --primary-hover: #023356;
    --primary-bg: #E9F3FA;
    --accent: #F16E20;
    --accent-bg: rgba(241, 110, 32, 0.10);

    /* Surfaces */
    --bg: #F6FAFE;
    --surface: #FFFFFF;
    --surface-2: #E9F3FA;
    --border: #D6E6F3;

    /* Text — all pass WCAG AA on their surfaces */
    --text: #0F1923;
    --text-muted: #5B7A93;

    /* Semantic */
    --success: #16a34a;
    --success-bg: rgba(22, 163, 74, 0.10);
    --warning: #b45309;
    --warning-bg: rgba(180, 83, 9, 0.10);
    --danger: #dc2626;
    --danger-bg: rgba(220, 38, 38, 0.10);

    /* Speaker colors */
    --agent-color: #02416E;
    --customer-color: #d97706;

    /* Legacy aliases — used by JS inline styles, do not remove */
    --border-color: #D6E6F3;
    --text-color: #0F1923;
    --bg-surface: #FFFFFF;
    --surface-hover: #E9F3FA;
    --border-hover: #A8C8E8;
    --primary-glow: rgba(2, 65, 110, 0.15);
    --glass: rgba(255, 255, 255, 0.6);
}

body.dark {
    --primary: #5BAEE0;
    --primary-hover: #4A9FD4;
    --primary-bg: rgba(91, 174, 224, 0.12);
    --accent: #F16E20;
    --accent-bg: rgba(241, 110, 32, 0.15);

    --bg: #08151F;
    --surface: #0F2338;
    --surface-2: #162D47;
    --border: #1E3A52;

    --text: #E8F4FF;
    --text-muted: #7B9CB5;

    --success: #4ade80;
    --success-bg: rgba(74, 222, 128, 0.12);
    --warning: #fbbf24;
    --warning-bg: rgba(251, 191, 36, 0.12);
    --danger: #f87171;
    --danger-bg: rgba(248, 113, 113, 0.12);

    --agent-color: #5BAEE0;
    --customer-color: #fbbf24;

    /* Legacy aliases — dark mode values */
    --border-color: #1E3A52;
    --text-color: #E8F4FF;
    --bg-surface: #0F2338;
    --surface-hover: #162D47;
    --border-hover: #2A4A6A;
    --primary-glow: rgba(91, 174, 224, 0.20);
    --glass: rgba(15, 35, 56, 0.8);
}
```

- [ ] **Step 2: Verify page loads without errors**

Open `http://localhost:8001` in browser. The page should render — colors will be wrong until layout tasks complete, but no JS errors in console.

- [ ] **Step 3: Commit**

```bash
cd "/Users/rajatkumawat/Projects/Transcription Service"
git add static/index.html
git commit -m "feat: replace CSS token system with FREED brand palette"
```

---

## Task 2: Global CSS Reset & Typography

**Files:**
- Modify: `static/index.html` — `body` rule, remove `font-family: 'Outfit'` references, update base typography

- [ ] **Step 1: Update `body` rule**

Replace the existing `body { ... }` CSS block with:

```css
body {
    font-family: 'Inter', ui-sans-serif, system-ui, sans-serif;
    background-color: var(--bg);
    color: var(--text);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    transition: background-color 200ms ease, color 200ms ease;
}
```

- [ ] **Step 2: Update `h2.section-title`**

Replace:
```css
h2.section-title {
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
    font-size: 2rem;
    margin-bottom: 30px;
    color: #fff;
}
```
With:
```css
h2.section-title {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 1.4rem;
    margin-bottom: 24px;
    color: var(--text);
}
```

- [ ] **Step 3: Replace all `font-family: 'Outfit'` in CSS**

Find every CSS occurrence of `font-family: 'Outfit'` or `font-family: 'Outfit', sans-serif` and replace with `font-family: 'Inter', sans-serif`. There are approximately 8 occurrences in the CSS section.

- [ ] **Step 4: Add base transition to `:root`**

Add after the `body.dark {}` block:

```css
*, *::before, *::after {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

/* Smooth theme transitions on key properties */
body, .sidebar, .top-bar, .main-content,
.panel, .kpi-card, .chart-box, .history-card,
.upload-card, .nav-item, .btn {
    transition: background-color 200ms ease,
                border-color 200ms ease,
                color 200ms ease;
}
```

- [ ] **Step 5: Commit**

```bash
git add static/index.html
git commit -m "feat: update typography to Inter, add theme transitions"
```

---

## Task 3: Add Anti-FOUC Script + Top Bar HTML

**Files:**
- Modify: `static/index.html` — `<body>` tag area, add `.top-bar` div, restructure layout

- [ ] **Step 1: Add no-flash script immediately after `<body>` tag**

Find the line:
```html
<body>
```
Replace with:
```html
<body>
<script>
    (function() {
        var t = localStorage.getItem('theme');
        if (t === 'dark') document.body.classList.add('dark');
        var s = localStorage.getItem('sidebar-state');
        if (s === 'collapsed') document.body.classList.add('sidebar-collapsed');
    })();
</script>
```

- [ ] **Step 2: Remove blob divs**

Find and delete these two lines from HTML:
```html
    <div class="blob blob-1"></div>
    <div class="blob blob-2"></div>
```

- [ ] **Step 3: Add `.top-bar` HTML before `.main-content`**

Find `<!-- Main Content -->` and replace with:
```html
<!-- Top Bar -->
<div class="top-bar">
    <span class="top-bar-title" id="top-bar-title">Global Analytics</span>
    <button class="theme-toggle" id="theme-toggle" title="Toggle theme" onclick="toggleTheme()">
        <svg class="icon-sun" width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <circle cx="12" cy="12" r="5"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/>
        </svg>
        <svg class="icon-moon" width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24" style="display:none">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z"/>
        </svg>
    </button>
</div>

<!-- Main Content -->
```

- [ ] **Step 4: Add sidebar collapse button to sidebar HTML**

Find the sidebar HTML opening:
```html
    <!-- Sidebar -->
    <div class="sidebar">
        <div class="brand">FREED AI</div>
```
Replace with:
```html
    <!-- Sidebar -->
    <div class="sidebar" id="sidebar">
        <div class="sidebar-header">
            <div class="brand">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="var(--accent)" style="flex-shrink:0">
                    <path d="M22 2L11 13M22 2L15 22l-4-9-9-4 20-7z"/>
                </svg>
                <span class="brand-text">FREED AI</span>
            </div>
            <button class="sidebar-toggle" id="sidebar-toggle" onclick="toggleSidebar()" title="Toggle sidebar">
                <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"/>
                </svg>
            </button>
        </div>
        <div class="nav-label">Navigation</div>
```

- [ ] **Step 5: Add `title` attributes to nav items for collapsed tooltips**

Update each nav item to add a `title` attribute:
```html
<div class="nav-item active" id="nav-overview" onclick="switchView('overview', this)" title="Overview">
```
```html
<div class="nav-item" id="nav-upload" onclick="switchView('upload', this)" title="New Process">
```
```html
<div class="nav-item" id="nav-history" onclick="switchView('history', this);" title="Job History">
```

- [ ] **Step 6: Wrap main content correctly**

Find:
```html
    <!-- Main Content -->
    <div class="main-content">
```
Replace with:
```html
    <!-- Main + Top Bar wrapper -->
    <div class="content-wrapper">
<!-- Main Content -->
<div class="main-content">
```

And find the closing `</div>` of `main-content` (just before `</body>`'s `<script>` block won't help — find the closing of `.main-content` which wraps all views). Add `</div>` after it to close `.content-wrapper`.

- [ ] **Step 7: Commit**

```bash
git add static/index.html
git commit -m "feat: add top bar, sidebar toggle button, remove blobs"
```

---

## Task 4: Layout CSS — Sidebar, Top Bar, Content Wrapper

**Files:**
- Modify: `static/index.html` — CSS section, sidebar styles, add top bar + content wrapper CSS

- [ ] **Step 1: Replace `.sidebar` CSS block**

Find the entire `.sidebar { ... }` block and replace with:

```css
.sidebar {
    width: 220px;
    min-width: 220px;
    background: var(--surface);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    padding: 16px 12px;
    z-index: 20;
    height: 100vh;
    position: sticky;
    top: 0;
    transition: width 200ms ease, min-width 200ms ease, padding 200ms ease;
    overflow: hidden;
}

body.sidebar-collapsed .sidebar {
    width: 56px;
    min-width: 56px;
    padding: 16px 8px;
}
```

- [ ] **Step 2: Replace `.brand` CSS and add new sidebar-header styles**

Replace the existing `.brand { ... }` block with:

```css
.sidebar-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 24px;
    gap: 8px;
}

.brand {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.95rem;
    font-weight: 800;
    color: var(--primary);
    letter-spacing: -0.02em;
    white-space: nowrap;
    overflow: hidden;
}

.brand-text {
    opacity: 1;
    transition: opacity 150ms ease;
    white-space: nowrap;
}

body.sidebar-collapsed .brand-text { opacity: 0; width: 0; }

.sidebar-toggle {
    background: none;
    border: none;
    cursor: pointer;
    color: var(--text-muted);
    padding: 4px;
    border-radius: 6px;
    display: flex;
    align-items: center;
    flex-shrink: 0;
    transition: color 150ms, background 150ms;
}

.sidebar-toggle:hover { color: var(--text); background: var(--surface-2); }

body.sidebar-collapsed .sidebar-toggle svg {
    transform: rotate(180deg);
}

.sidebar-toggle svg { transition: transform 200ms ease; }

.nav-label {
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--text-muted);
    padding: 0 8px;
    margin-bottom: 8px;
    opacity: 1;
    transition: opacity 150ms ease;
    white-space: nowrap;
}

body.sidebar-collapsed .nav-label { opacity: 0; }
```

- [ ] **Step 3: Replace `.nav-item` CSS**

Replace the `.nav-item`, `.nav-item:hover`, and `.nav-item.active` blocks with:

```css
.nav-item {
    padding: 10px 12px;
    border-radius: 8px;
    color: var(--text-muted);
    font-weight: 500;
    font-size: 0.875rem;
    cursor: pointer;
    transition: background 150ms ease, color 150ms ease;
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 10px;
    white-space: nowrap;
    overflow: hidden;
}

.nav-item:hover {
    background: var(--surface-2);
    color: var(--text);
}

.nav-item.active {
    background: var(--primary);
    color: #fff;
}

.nav-item span {
    opacity: 1;
    transition: opacity 150ms ease;
}

body.sidebar-collapsed .nav-item span { opacity: 0; width: 0; overflow: hidden; }
body.sidebar-collapsed .nav-item { justify-content: center; }
```

- [ ] **Step 4: Add `.top-bar` and `.content-wrapper` CSS**

Add after the sidebar CSS:

```css
/* Layout wrappers */
.app-shell {
    display: flex;
    height: 100vh;
    overflow: hidden;
}

.content-wrapper {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    min-width: 0;
}

.top-bar {
    height: 56px;
    min-height: 56px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 32px;
    z-index: 10;
}

.top-bar-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text);
}

.theme-toggle {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    cursor: pointer;
    color: var(--text-muted);
    padding: 7px;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: color 150ms, background 150ms;
}

.theme-toggle:hover { color: var(--text); background: var(--border); }
```

- [ ] **Step 5: Update `.main-content` CSS**

Replace the existing `.main-content { ... }` block with:

```css
.main-content {
    flex: 1;
    overflow-y: auto;
    padding: 28px 32px;
}
```

- [ ] **Step 6: Update the outer `body > div` layout**

The current layout has `body { display: flex }` which puts sidebar + main-content as flex children. With the new `.content-wrapper`, wrap the existing body flex in an `.app-shell`. 

In HTML, find:
```html
    <!-- Sidebar -->
    <div class="sidebar" id="sidebar">
```
And ensure the overall structure is:
```html
<div class="app-shell">
    <!-- Sidebar -->
    <div class="sidebar" id="sidebar">
        ...
    </div>
    <!-- Content Wrapper (top bar + main content) -->
    <div class="content-wrapper">
        <!-- Top Bar -->
        <div class="top-bar">...</div>
        <!-- Main Content -->
        <div class="main-content">
            ...all view sections...
        </div>
    </div>
</div>
```

Update the body CSS to remove `display: flex` since `.app-shell` handles that:
```css
body {
    font-family: 'Inter', ui-sans-serif, system-ui, sans-serif;
    background-color: var(--bg);
    color: var(--text);
    min-height: 100vh;
    transition: background-color 200ms ease, color 200ms ease;
}
```

- [ ] **Step 7: Remove old blob CSS**

Find and delete the `.blob`, `.blob-1`, `.blob-2`, and `@keyframes float` CSS blocks entirely.

- [ ] **Step 8: Open browser and verify layout structure**

Load `http://localhost:8001`. Expected:
- Sidebar visible on left, 220px wide
- Top bar visible across full width with "Global Analytics" title
- Main content below top bar
- No blobs, no broken layout

- [ ] **Step 9: Commit**

```bash
git add static/index.html
git commit -m "feat: layout redesign — collapsible sidebar, top bar, app shell"
```

---

## Task 5: Theme Toggle + Sidebar Collapse JS

**Files:**
- Modify: `static/index.html` — add UI state JS block near end of `<script>` section

- [ ] **Step 1: Add theme + sidebar JS near the top of the `<script>` block**

Find the `<script>` opening tag and add immediately after it:

```javascript
// ── UI State: Theme & Sidebar ──────────────────────────────────────────
function toggleTheme() {
    const isDark = document.body.classList.toggle('dark');
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
    updateThemeIcon(isDark);
}

function updateThemeIcon(isDark) {
    const sun = document.querySelector('.icon-sun');
    const moon = document.querySelector('.icon-moon');
    if (!sun || !moon) return;
    sun.style.display = isDark ? 'none' : 'block';
    moon.style.display = isDark ? 'block' : 'none';
}

function toggleSidebar() {
    const collapsed = document.body.classList.toggle('sidebar-collapsed');
    localStorage.setItem('sidebar-state', collapsed ? 'collapsed' : 'expanded');
}

// Apply saved states (backup to the inline no-flash script)
window.addEventListener('DOMContentLoaded', function() {
    const isDark = document.body.classList.contains('dark');
    updateThemeIcon(isDark);
});
// ── End UI State ────────────────────────────────────────────────────────
```

- [ ] **Step 2: Update `switchView()` to also update top bar title**

Find `function switchView(viewId, navElement) {` and add at the top of the function body:

```javascript
// Update top bar title
const titleMap = {
    'overview': 'Global Analytics',
    'upload': 'New Transcription',
    'history': 'Job History',
    'dashboard': 'Call Analysis'
};
const titleEl = document.getElementById('top-bar-title');
if (titleEl && titleMap[viewId]) titleEl.textContent = titleMap[viewId];
```

- [ ] **Step 3: Verify toggle works**

Load `http://localhost:8001`. Click the theme toggle — page should switch between light (white cards) and dark (dark navy cards). Click the sidebar `←` button — sidebar should collapse to 56px icons. Refresh — state should persist.

- [ ] **Step 4: Commit**

```bash
git add static/index.html
git commit -m "feat: theme toggle and sidebar collapse with localStorage persistence"
```

---

## Task 6: Button, Badge, and Card Components

**Files:**
- Modify: `static/index.html` — CSS section, button/badge/card/input CSS

- [ ] **Step 1: Replace `.btn` CSS**

Find the existing `.btn { ... }` and all `.btn:hover`, `.btn-secondary` blocks. Replace entirely with:

```css
.btn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 9px 18px;
    background: var(--primary);
    color: #fff;
    border: none;
    border-radius: 8px;
    font-family: 'Inter', sans-serif;
    font-size: 0.875rem;
    font-weight: 600;
    cursor: pointer;
    transition: background 150ms ease, transform 100ms ease;
    white-space: nowrap;
}

.btn:hover { background: var(--primary-hover); }
.btn:active { transform: scale(0.98); }
.btn:disabled { opacity: 0.5; cursor: not-allowed; }

.btn-secondary {
    background: transparent;
    color: var(--text);
    border: 1px solid var(--border);
}

.btn-secondary:hover { background: var(--surface-2); }

.btn-ghost {
    background: transparent;
    color: var(--text-muted);
    border: none;
    padding: 6px 10px;
}

.btn-ghost:hover { color: var(--text); background: var(--surface-2); }

.btn-danger {
    background: var(--danger-bg);
    color: var(--danger);
    border: 1px solid rgba(220, 38, 38, 0.25);
}

.btn-danger:hover { background: rgba(220, 38, 38, 0.18); }
```

- [ ] **Step 2: Replace `.badge` CSS**

Find the existing `.badge { ... }` block and replace with:

```css
.badge {
    display: inline-flex;
    align-items: center;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    background: var(--surface-2);
    color: var(--text-muted);
    border: 1px solid var(--border);
    white-space: nowrap;
}

.badge-primary { background: var(--primary-bg); color: var(--primary); border-color: rgba(2,65,110,0.2); }
.badge-success { background: var(--success-bg); color: var(--success); border-color: rgba(22,163,74,0.2); }
.badge-warning { background: var(--warning-bg); color: var(--warning); border-color: rgba(180,83,9,0.2); }
.badge-danger  { background: var(--danger-bg);  color: var(--danger);  border-color: rgba(220,38,38,0.2); }
.badge-accent  { background: var(--accent-bg);  color: var(--accent);  border-color: rgba(241,110,32,0.2); }
```

- [ ] **Step 3: Update `.kpi-card` CSS**

Find `.kpi-card { ... }` block and replace with:

```css
.kpi-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 20px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    cursor: pointer;
    transition: border-color 200ms ease, box-shadow 200ms ease;
}

.kpi-card:hover {
    border-color: rgba(2, 65, 110, 0.4);
    box-shadow: 0 2px 12px rgba(2, 65, 110, 0.08);
}

body.dark .kpi-card:hover {
    border-color: rgba(91, 174, 224, 0.4);
    box-shadow: 0 2px 12px rgba(91, 174, 224, 0.08);
}

.kpi-label {
    font-size: 0.72rem;
    font-weight: 500;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    line-height: 1.3;
}

.kpi-val {
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--primary);
    line-height: 1.1;
}
```

- [ ] **Step 4: Add form input CSS**

Add after the badge CSS:

```css
.form-input {
    width: 100%;
    padding: 10px 14px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text);
    font-family: 'Inter', sans-serif;
    font-size: 0.875rem;
    transition: border-color 150ms ease, box-shadow 150ms ease;
    outline: none;
}

.form-input::placeholder { color: var(--text-muted); }
.form-input:focus {
    border-color: var(--primary);
    box-shadow: 0 0 0 3px var(--primary-bg);
}
```

- [ ] **Step 5: Commit**

```bash
git add static/index.html
git commit -m "feat: redesign button, badge, kpi-card, form-input components"
```

---

## Task 7: Overview View — Charts & KPI Grid

**Files:**
- Modify: `static/index.html` — `.charts-container`, `.chart-box`, KPI tooltip, Chart.js defaults

- [ ] **Step 1: Replace `.charts-container` and `.chart-box` CSS**

Find the existing `.charts-container { ... }` and `.chart-box { ... }` blocks and replace with:

```css
.charts-container {
    display: grid;
    gap: 20px;
    margin-bottom: 20px;
}

.chart-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
}

.chart-box h3 {
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: var(--text-muted);
    margin-bottom: 16px;
}
```

- [ ] **Step 2: Update KPI tooltip styles**

Find the `#kpi-tooltip` inline style in HTML:
```html
<div id="kpi-tooltip"
    style="display:none; position:fixed; z-index:9999; background:rgba(9,9,11,0.95); border:1px solid rgba(255,255,255,0.12); border-radius:12px; padding:16px 20px; max-width:420px; max-height:350px; overflow-y:auto; box-shadow:0 20px 40px rgba(0,0,0,0.5); backdrop-filter:blur(12px); font-size:0.85rem; color:var(--text);">
```
Replace with:
```html
<div id="kpi-tooltip"
    style="display:none; position:fixed; z-index:9999; background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:16px 20px; max-width:420px; max-height:350px; overflow-y:auto; box-shadow:0 8px 24px rgba(0,0,0,0.15); font-size:0.85rem; color:var(--text);">
```

- [ ] **Step 3: Update Chart.js defaults in JS**

Find the `// --- Chart Defaults ---` section in JS and replace with:

```javascript
// --- Chart Defaults ---
Chart.defaults.color = getComputedStyle(document.body).getPropertyValue('--text-muted').trim() || '#5B7A93';
Chart.defaults.font.family = 'Inter';
Chart.defaults.scale.grid.color = getComputedStyle(document.body).getPropertyValue('--border').trim() || '#D6E6F3';
Chart.defaults.plugins.tooltip.backgroundColor = getComputedStyle(document.body).getPropertyValue('--surface').trim() || '#fff';
Chart.defaults.plugins.tooltip.titleColor = getComputedStyle(document.body).getPropertyValue('--text').trim() || '#0F1923';
Chart.defaults.plugins.tooltip.bodyColor = getComputedStyle(document.body).getPropertyValue('--text-muted').trim() || '#5B7A93';
Chart.defaults.plugins.tooltip.borderColor = getComputedStyle(document.body).getPropertyValue('--border').trim() || '#D6E6F3';
Chart.defaults.plugins.tooltip.borderWidth = 1;
Chart.defaults.plugins.tooltip.padding = 12;
Chart.defaults.plugins.tooltip.cornerRadius = 8;
```

- [ ] **Step 4: Update chart color palettes in `initOverview`**

Find the chart initialization calls (the `new Chart(...)` calls for `chart-sentiment`, `chart-probability`, `chart-painpoints`, `chart-keywords`, `chart-funnel`). Update their `backgroundColor` arrays from the current purple/cyan palette to:

```javascript
// Primary series palette for all charts
const chartPalette = [
    '#02416E', '#F16E20', '#5BAEE0', '#4ade80',
    '#fbbf24', '#a3b8cc', '#6d8fa8', '#e8a87c'
];
```

Add this constant near the top of `initOverview()` and use `chartPalette` for `backgroundColor` in each chart dataset.

- [ ] **Step 5: Verify overview charts render correctly in both themes**

Toggle dark mode. Charts should update colors on next `initOverview()` call (triggered by clicking Overview nav item).

- [ ] **Step 6: Commit**

```bash
git add static/index.html
git commit -m "feat: redesign overview charts, kpi tooltip, chart color palette"
```

---

## Task 8: Upload View — Drop Zone & Chain Panel

**Files:**
- Modify: `static/index.html` — CSS and HTML for upload view

- [ ] **Step 1: Replace `.upload-card` CSS**

Find the existing `.upload-card { ... }` block and replace:

```css
.upload-card {
    max-width: 520px;
    margin: 0 auto;
    border: 2px dashed var(--border);
    border-radius: 14px;
    padding: 48px 32px;
    text-align: center;
    cursor: pointer;
    transition: border-color 150ms ease, background 150ms ease;
    background: var(--surface);
}

.upload-card:hover,
.upload-card.drag-over {
    border-color: var(--primary);
    background: var(--surface-2);
}

.upload-card h3 {
    font-size: 1.2rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 8px;
}

.upload-card p {
    color: var(--text-muted);
    font-size: 0.875rem;
    line-height: 1.6;
}
```

- [ ] **Step 2: Update upload mode tab switcher HTML**

Find:
```html
<div style="display:flex; gap:8px; max-width:800px; margin:0 auto 20px; padding:4px; background:var(--bg-surface); border:1px solid var(--border-color); border-radius:10px;">
    <button id="mode-single-btn" onclick="setUploadMode('single')" style="flex:1; padding:10px 16px; border:none; background:var(--primary); color:#fff; border-radius:8px; font-weight:600; cursor:pointer; transition:all 0.15s;">Single Call</button>
    <button id="mode-chain-btn" onclick="setUploadMode('chain')" style="flex:1; padding:10px 16px; border:none; background:transparent; color:var(--text-muted); border-radius:8px; font-weight:600; cursor:pointer; transition:all 0.15s;">Chain of Calls</button>
</div>
```
Replace with:
```html
<div class="upload-mode-tabs">
    <button id="mode-single-btn" class="upload-mode-btn active" onclick="setUploadMode('single')">Single Call</button>
    <button id="mode-chain-btn" class="upload-mode-btn" onclick="setUploadMode('chain')">
        <svg width="14" height="14" fill="none" stroke="currentColor" viewBox="0 0 24 24" style="flex-shrink:0">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1"/>
        </svg>
        Chain of Calls
    </button>
</div>
```

- [ ] **Step 3: Add upload mode tab CSS**

Add:
```css
.upload-mode-tabs {
    display: flex;
    gap: 4px;
    max-width: 520px;
    margin: 0 auto 24px;
    padding: 4px;
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 10px;
}

.upload-mode-btn {
    flex: 1;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    padding: 9px 16px;
    border: none;
    background: transparent;
    color: var(--text-muted);
    border-radius: 8px;
    font-family: 'Inter', sans-serif;
    font-size: 0.875rem;
    font-weight: 600;
    cursor: pointer;
    transition: background 150ms ease, color 150ms ease;
}

.upload-mode-btn.active {
    background: var(--primary);
    color: #fff;
}

.upload-mode-btn:not(.active):hover {
    background: var(--surface);
    color: var(--text);
}
```

- [ ] **Step 4: Update `setUploadMode()` to use new CSS class**

Find `function setUploadMode(mode) {` and update to use the `.active` class on the buttons instead of inline style manipulation:

```javascript
function setUploadMode(mode) {
    const singleBtn = document.getElementById('mode-single-btn');
    const chainBtn = document.getElementById('mode-chain-btn');
    const singlePanel = document.getElementById('mode-single-panel');
    const chainPanel = document.getElementById('mode-chain-panel');
    if (mode === 'chain') {
        chainBtn.classList.add('active'); singleBtn.classList.remove('active');
        chainPanel.style.display = 'block'; singlePanel.style.display = 'none';
    } else {
        singleBtn.classList.add('active'); chainBtn.classList.remove('active');
        singlePanel.style.display = 'block'; chainPanel.style.display = 'none';
    }
}
```

- [ ] **Step 5: Fix chain panel inline dark-mode styles**

These specific inline style strings in HTML use hardcoded dark values. Find and replace each:

**Chain card container** (find):
```html
<div style="background:var(--bg-surface); border:1px solid var(--border-color); border-radius:12px; padding:24px; margin-bottom:20px;">
```
Replace with:
```html
<div style="background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:24px; margin-bottom:20px;">
```

**Chain label input** (find):
```html
style="padding:10px 14px; background:rgba(0,0,0,0.3); border:1px solid var(--border-color); border-radius:8px; color:var(--text-color); font-family:inherit; font-size:0.9rem;"
```
Replace both input instances with:
```html
style="padding:10px 14px; background:var(--surface); border:1px solid var(--border); border-radius:8px; color:var(--text); font-family:inherit; font-size:0.9rem; outline:none;"
```

**Chain drop zone** (find):
```html
style="border:2px dashed var(--border-color); border-radius:10px; padding:28px; text-align:center; cursor:pointer; background:rgba(0,0,0,0.2); transition:all 0.15s;"
```
Replace with:
```html
style="border:2px dashed var(--border); border-radius:10px; padding:28px; text-align:center; cursor:pointer; background:var(--surface-2); transition:all 0.15s;"
```

**Chain drop zone title** (find):
```html
<div style="font-family:'Outfit'; font-size:1.05rem; margin-bottom:6px; color:var(--text-color);">Drop audio files here</div>
```
Replace with:
```html
<div style="font-family:'Inter'; font-size:1rem; font-weight:600; margin-bottom:6px; color:var(--text);">Drop audio files here</div>
```

**Chain dragover handler** in JS (find):
```javascript
chainDrop.addEventListener('dragover', e => { e.preventDefault(); chainDrop.style.background = 'rgba(56,189,248,0.08)'; });
chainDrop.addEventListener('dragleave', () => { chainDrop.style.background = 'rgba(0,0,0,0.2)'; });
```
Replace with:
```javascript
chainDrop.addEventListener('dragover', e => { e.preventDefault(); chainDrop.style.background = 'var(--primary-bg)'; chainDrop.style.borderColor = 'var(--primary)'; });
chainDrop.addEventListener('dragleave', () => { chainDrop.style.background = 'var(--surface-2)'; chainDrop.style.borderColor = 'var(--border)'; });
```

**Chain drop handler** in JS (find):
```javascript
chainDrop.style.background = 'rgba(0,0,0,0.2)';
```
Replace with:
```javascript
chainDrop.style.background = 'var(--surface-2)';
```

- [ ] **Step 6: Fix "Add More" button inline style**

Find:
```html
style="padding: 8px 16px; font-size: 0.9rem; background: rgba(255,255,255,0.05); color: var(--text-color); border: 1px solid rgba(255,255,255,0.1);"
```
Replace with:
```html
class="btn btn-secondary" style="padding: 8px 14px; font-size: 0.875rem;"
```

- [ ] **Step 7: Fix chain file row inline styles in JS**

Find:
```javascript
row.style.cssText = 'display:flex; align-items:center; gap:12px; padding:10px 14px; background:rgba(0,0,0,0.3); border:1px solid var(--border-color); border-radius:8px;';
```
Replace with:
```javascript
row.style.cssText = 'display:flex; align-items:center; gap:12px; padding:10px 14px; background:var(--surface-2); border:1px solid var(--border); border-radius:8px;';
```

- [ ] **Step 8: Fix chain progress row inline styles in JS**

Find:
```javascript
row.style.cssText = 'padding:12px 16px; background:var(--bg-surface); border:1px solid var(--border-color); border-radius:8px; display:flex; flex-direction:column; gap:8px;';
```
Replace with:
```javascript
row.style.cssText = 'padding:12px 16px; background:var(--surface); border:1px solid var(--border); border-radius:8px; display:flex; flex-direction:column; gap:8px;';
```

- [ ] **Step 9: Fix single upload queue item inline styles in JS**

Find:
```javascript
item.style.cssText = 'background: var(--bg-surface); border: 1px solid var(--border-color); border-radius: 8px; padding: 15px; display: flex; flex-direction: column; gap: 10px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);';
```
Replace with:
```javascript
item.style.cssText = 'background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 15px; display: flex; flex-direction: column; gap: 10px;';
```

- [ ] **Step 10: Verify upload view in both themes**

Load upload view, switch themes — drop zone, inputs, chain panel should all look correct in light and dark.

- [ ] **Step 11: Commit**

```bash
git add static/index.html
git commit -m "feat: redesign upload view, fix inline dark-mode rgba values"
```

---

## Task 9: History View — Row List

**Files:**
- Modify: `static/index.html` — `.history-card`, `.history-grid` CSS, `_buildSingleCard()` and `_buildChainCard()` inline styles

- [ ] **Step 1: Replace `.history-grid` CSS**

Find the `.history-grid { ... }` block and replace with:

```css
.history-grid {
    display: flex;
    flex-direction: column;
    gap: 0;
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
    background: var(--surface);
}
```

- [ ] **Step 2: Replace `.history-card` CSS**

Find the `.history-card { ... }` and related sub-element blocks and replace with:

```css
.history-card {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 14px 20px;
    cursor: pointer;
    transition: background 150ms ease;
    display: flex;
    align-items: center;
    gap: 16px;
    position: relative;
}

.history-card:last-child { border-bottom: none; }
.history-card:hover { background: var(--surface-2); }

.history-card .date {
    font-size: 0.78rem;
    color: var(--text-muted);
    white-space: nowrap;
    flex-shrink: 0;
    width: 80px;
}

.history-card .filename {
    font-weight: 500;
    font-size: 0.875rem;
    color: var(--text);
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    margin-bottom: 0;
    word-break: normal;
}

.history-card .badges {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-shrink: 0;
}
```

- [ ] **Step 3: Fix `_buildSingleCard()` inline styles**

The `body` element in `_buildSingleCard` has `height:100%; display:flex; flex-direction:column; cursor:pointer;` — since layout is now row-based, update:

Find:
```javascript
body.style.cssText = 'height:100%; display:flex; flex-direction:column; cursor:pointer;';
```
Replace with:
```javascript
body.style.cssText = 'flex:1; display:flex; align-items:center; gap:16px; min-width:0; cursor:pointer;';
```

Also update badges margin:
```javascript
badges.style.marginTop = 'auto';
```
Replace with:
```javascript
badges.style.marginTop = '0';
```

- [ ] **Step 4: Fix `_buildChainCard()` — update chain accent color**

Find:
```javascript
card.style.borderLeft = '3px solid #a78bfa';
```
Replace with:
```javascript
card.style.borderLeft = '3px solid var(--accent)';
```

Find:
```javascript
tag.style.cssText = 'position:absolute; top:10px; right:10px; z-index:10; font-size:0.7rem; color:#a78bfa; letter-spacing:0.08em; text-transform:uppercase; font-weight:700;';
```
Replace with:
```javascript
tag.style.cssText = 'font-size:0.7rem; color:var(--accent); letter-spacing:0.08em; text-transform:uppercase; font-weight:700; flex-shrink:0;';
```

Update chain card body structure for row layout:
```javascript
// After creating dateDiv, labelDiv, badges — adjust layout
// Find this block in _buildChainCard:
card.appendChild(dateDiv); card.appendChild(labelDiv); card.appendChild(badges);
card.appendChild(open); card.appendChild(memberPanel);
```
Replace with:
```javascript
const chainBody = document.createElement('div');
chainBody.style.cssText = 'flex:1; display:flex; align-items:center; gap:16px; min-width:0;';
chainBody.appendChild(dateDiv); chainBody.appendChild(labelDiv); chainBody.appendChild(badges);
card.appendChild(tag); card.appendChild(chainBody);
card.appendChild(open); card.appendChild(memberPanel);
```

- [ ] **Step 5: Fix chain expand button inline style**

Find:
```javascript
expandBtn.style.cssText = 'padding:6px 12px; background:rgba(255,255,255,0.05); color:var(--text-color); border:1px solid rgba(255,255,255,0.1); border-radius:6px; cursor:pointer; font-size:0.85rem;';
```
Replace with:
```javascript
expandBtn.style.cssText = 'padding:6px 12px; background:var(--surface-2); color:var(--text-muted); border:1px solid var(--border); border-radius:6px; cursor:pointer; font-size:0.85rem; font-family:inherit;';
```

- [ ] **Step 6: Fix member panel inline style**

Find:
```javascript
memberPanel.style.cssText = 'display:none; margin-top:10px; padding:10px; background:rgba(0,0,0,0.25); border-radius:6px;';
```
Replace with:
```javascript
memberPanel.style.cssText = 'display:none; margin-top:10px; padding:10px; background:var(--surface-2); border:1px solid var(--border); border-radius:8px; width:100%;';
```

- [ ] **Step 7: Verify history view**

Load Job History. Should show a clean list with single call rows and chain rows (orange left border). Members expand panel should show on click.

- [ ] **Step 8: Commit**

```bash
git add static/index.html
git commit -m "feat: redesign history view as row list, fix chain card colors"
```

---

## Task 10: Call Analysis Panel + Collapsible Details

**Files:**
- Modify: `static/index.html` — `.panel`, `.panel-header`, `.panel-content`, `.dashboard-grid`, collapsible panel CSS

- [ ] **Step 1: Replace `.panel`, `.panel-header`, `.panel-content` CSS**

Find and replace:

```css
.panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

.panel-header {
    background: var(--surface-2);
    border-bottom: 1px solid var(--border);
    padding: 14px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-shrink: 0;
}

.panel-header h3 {
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: var(--text-muted);
    margin: 0;
}

.panel-content {
    overflow-y: auto;
    flex: 1;
    padding: 0;
}
```

- [ ] **Step 2: Update `.dashboard-grid` CSS**

Replace the existing `.dashboard-grid { ... }` with:

```css
.dashboard-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    height: calc(100vh - 56px - 56px - 24px);
}
```

- [ ] **Step 3: Restyle transcript toggle button**

Find the `.toggle-rephrase-btn` CSS block and replace with:

```css
.toggle-rephrase-btn {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 5px 10px;
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text-muted);
    font-size: 0.78rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 150ms ease;
    font-family: 'Inter', sans-serif;
}

.toggle-rephrase-btn.active {
    background: var(--primary-bg);
    border-color: rgba(2, 65, 110, 0.3);
    color: var(--primary);
}

body.dark .toggle-rephrase-btn.active {
    border-color: rgba(91, 174, 224, 0.3);
}
```

- [ ] **Step 4: Restyle `.emotion-panel` (`<details>` panels)**

Find the `.emotion-panel { ... }` and related CSS and replace:

```css
.emotion-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    margin-bottom: 12px;
    overflow: hidden;
}

.emotion-panel > summary {
    cursor: pointer;
    list-style: none;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 13px 18px;
    user-select: none;
    transition: background 150ms ease;
}

.emotion-panel > summary:hover {
    background: var(--surface-2);
}

.emotion-panel > summary::-webkit-details-marker { display: none; }

.emotion-panel[open] > summary {
    border-bottom: 1px solid var(--border);
    background: var(--surface-2);
}

.emotion-panel[open] > summary .panel-chevron {
    transform: rotate(180deg);
}

.panel-chevron {
    transition: transform 200ms ease;
    color: var(--text-muted);
    flex-shrink: 0;
}
```

- [ ] **Step 5: Update chevron SVGs in JS-generated panel summaries**

In the `buildSummaryHtml` / panel-building JS sections, find every occurrence of:
```javascript
<svg style="width:18px;height:18px;flex-shrink:0;color:var(--text-muted)" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/></svg>
```
Add `class="panel-chevron"` to each SVG:
```javascript
<svg class="panel-chevron" style="width:18px;height:18px;" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/></svg>
```

- [ ] **Step 6: Restyle `.meta-toggle-btn` (metadata toggle)**

Find `.meta-toggle-btn { ... }` and replace:

```css
.meta-toggle-btn {
    background: var(--surface-2);
    border: 1px solid var(--border);
    cursor: pointer;
    padding: 4px 10px;
    border-radius: 6px;
    color: var(--text-muted);
    font-size: 0.78rem;
    font-weight: 500;
    font-family: 'Inter', sans-serif;
    transition: color 150ms, background 150ms;
    display: inline-flex;
    align-items: center;
    gap: 4px;
    margin-left: 8px;
}

.meta-toggle-btn:hover { color: var(--primary); background: var(--primary-bg); }

.meta-toggle-btn .chevron {
    display: inline-block;
    transition: transform 300ms cubic-bezier(0.4, 0, 0.2, 1);
}

.meta-toggle-btn.open .chevron { transform: rotate(180deg); }
```

- [ ] **Step 7: Update stat-card CSS**

Find `.stat-card { ... }` and replace:

```css
.stat-card {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 16px;
    text-align: center;
}

.stat-card span {
    display: block;
    font-size: 0.72rem;
    font-weight: 500;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.4px;
    margin-bottom: 4px;
}

.stat-card strong {
    font-size: 1rem;
    font-weight: 700;
    color: var(--primary);
}
```

- [ ] **Step 8: Fix `.emotion-metric-card` background**

Find `.emotion-metric-card { ... }` and replace `background: rgba(255, 255, 255, 0.03)` with `background: var(--surface-2)`.

Full replacement:
```css
.emotion-metric-card {
    text-align: center;
    padding: 10px 6px;
    border-radius: 8px;
    background: var(--surface-2);
    border: 1px solid var(--border);
}
```

- [ ] **Step 9: Verify call analysis view**

Open any transcription from history. Both panels should render correctly. Toggle transcript view, open metadata, expand emotion/audit/checklist panels.

- [ ] **Step 10: Commit**

```bash
git add static/index.html
git commit -m "feat: redesign call analysis panels, details/summary, stat cards"
```

---

## Task 11: Transcript Timeline & Speaker Segments

**Files:**
- Modify: `static/index.html` — `.timeline-entry`, `.speaker-label`, `.transcript-text`, speaker segment CSS

- [ ] **Step 1: Replace timeline CSS**

Find the `.timeline-entry { ... }` block and replace with:

```css
.timeline-entry {
    padding: 12px 20px;
    border-bottom: 1px solid var(--border);
    display: flex;
    gap: 14px;
    align-items: flex-start;
    transition: background 150ms ease;
}

.timeline-entry:last-child { border-bottom: none; }
.timeline-entry:hover { background: var(--surface-2); }

.speaker-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    flex-shrink: 0;
    width: 72px;
    padding-top: 2px;
}

.speaker-label.agent { color: var(--agent-color); }
.speaker-label.customer { color: var(--customer-color); }

.transcript-text {
    font-size: 0.875rem;
    color: var(--text);
    line-height: 1.6;
    flex: 1;
}

.transcript-timestamp {
    font-size: 0.72rem;
    color: var(--text-muted);
    font-family: ui-monospace, 'Geist Mono', monospace;
    flex-shrink: 0;
    padding-top: 2px;
}
```

- [ ] **Step 2: Fix hardcoded `#e4e4e7` filename color in chain tooltip JS**

Find in the chain dashboard JS:
```javascript
html += `<span style="font-weight:500; color:#e4e4e7;">📞 ${m.filename}</span>`;
```
Replace with:
```javascript
html += `<span style="font-weight:500; color:var(--text);">📞 ${m.filename}</span>`;
```

- [ ] **Step 3: Fix hardcoded `#6ee7b7` and `#a78bfa` in KPI tooltip JS**

Find all hardcoded colors in the KPI tooltip build strings:
- `color:#6ee7b7` → `color:var(--success)`
- `color:#a78bfa` → `color:var(--primary)`
- `border-bottom:1px solid rgba(255,255,255,0.05)` → `border-bottom:1px solid var(--border)`

There are approximately 8 occurrences across the KPI tooltip builder. Replace all.

- [ ] **Step 4: Commit**

```bash
git add static/index.html
git commit -m "feat: redesign transcript timeline, fix hardcoded colors in tooltips"
```

---

## Task 12: Final Polish — Readability Pass

**Files:**
- Modify: `static/index.html` — any remaining hardcoded colors, contrast fixes

- [ ] **Step 1: Fix `.history-card .filename` hardcoded `color: #fff`**

Find in CSS:
```css
.history-card .filename {
    ...
    color: #fff;
    ...
}
```
The `color: #fff` should already be replaced from Task 9 Step 2. Verify it reads `color: var(--text)`.

- [ ] **Step 2: Find and replace remaining hardcoded dark colors in CSS**

Search for any remaining hardcoded colors in the CSS section that will be unreadable in light mode:
- `color: #fff` in non-button contexts → `color: var(--text)`
- `color: #a1a1aa` → `color: var(--text-muted)`
- `color: #f4f4f5` → `color: var(--text)`
- `background: rgba(24, 24, 27` → `background: var(--surface)`

Run this search in the file to find them:
```
grep -n "color: #fff\|color:#fff\|#a1a1aa\|#f4f4f5\|rgba(24, 24, 27\|rgba(9,9,11\|rgba(15,15,20" static/index.html
```

- [ ] **Step 3: Fix `.kv-item`, `.kv-key`, `.kv-val` colors**

Find `.kv-key { ... }` and update:
```css
.kv-key {
    font-size: 0.72rem;
    font-weight: 500;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
}

.kv-val {
    font-size: 0.9rem;
    color: var(--text);
    line-height: 1.5;
}

.kv-item { border-left: 2px solid var(--border-hover); padding-left: 14px; }
```

- [ ] **Step 4: Fix summary list colors**

Find `.summary-list li { ... }` and update `color: #d1d5db` → `color: var(--text)`.
Find `.summary-list li::before { ... }` and update `color: var(--primary)` (already correct).

- [ ] **Step 5: Verify contrast in light mode with browser DevTools**

Open `http://localhost:8001` in light mode. Use DevTools Accessibility checker to verify:
- KPI values readable on white cards ✅
- Nav items readable ✅
- Chart axis labels readable ✅
- Badge text readable on tinted backgrounds ✅

- [ ] **Step 6: Verify contrast in dark mode**

Toggle to dark mode. Verify:
- Text on dark surfaces ✅
- Orange accent elements don't appear as body text ✅
- Muted text passes AA (≥4.5:1) ✅

- [ ] **Step 7: Remove Outfit from Google Fonts link**

Find:
```html
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
```
Replace with:
```html
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
```

- [ ] **Step 8: Final full-page functional test**

Test all navigation paths:
1. Overview → charts load, KPI tooltips work
2. New Transcription → drop zone works, Single/Chain mode toggle works
3. History → cards load, single + chain rows display, Members toggle works
4. Open a call → transcript + panels render, theme toggle works mid-session
5. Collapse sidebar → layout adjusts correctly, tooltips appear on icon hover
6. Refresh page → theme and sidebar state restored

- [ ] **Step 9: Final commit**

```bash
git add static/index.html
git commit -m "feat: complete frontend revamp — FREED brand, adaptive theme, collapsible sidebar

- FREED brand palette: #02416E navy + #F16E20 orange
- Adaptive light/dark theme with localStorage persistence
- Collapsible sidebar (220px ↔ 56px) with icon tooltips
- Top bar with dynamic page title and theme toggle
- Modern SaaS component system: buttons, badges, cards, panels
- History view redesigned as row list (single + chain rows)
- Call chaining integrated: orange accent, proper light-mode styles
- All JS logic, IDs, API calls, and redirections unchanged
- Removed blobs, glassmorphism, Outfit font, hardcoded dark rgba values"
```
