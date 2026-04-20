# Frontend Revamp Design Spec
**Date:** 2026-04-20  
**Project:** FREED AI Transcription Service Dashboard  
**Scope:** Full CSS + layout overhaul of `static/index.html` (2,774 lines)  
**Constraint:** Zero JavaScript logic changes — all IDs, function names, event handlers, API calls, and redirections preserved exactly.

---

## 1. Design Direction

**Style:** Modern SaaS — Notion/Clerk inspired. Clean, confident, minimal decoration.  
**Theme:** Adaptive — light by default, dark mode toggle in top-right of top bar.  
**Brand:** FREED organisation colors extracted from freed.care.

---

## 2. Color System

### Brand Colors (from freed.care)
| Token | Light Mode | Dark Mode | Usage |
|-------|-----------|-----------|-------|
| `--primary` | `#02416E` | `#5BAEE0` | Buttons, active nav, links, key values |
| `--primary-hover` | `#023356` | `#4A9FD4` | Button hover states |
| `--primary-bg` | `#E9F3FA` | `rgba(91,174,224,0.12)` | Focus rings, soft fills |
| `--accent` | `#F16E20` | `#F16E20` | Icons only, chart series, left-border decorations |
| `--accent-bg` | `rgba(241,110,32,0.10)` | `rgba(241,110,32,0.15)` | Badge tints for accent |

### Readability Rule
`--accent` (#F16E20) is **never used as text color on light backgrounds** — contrast ratio ~3.5:1 fails WCAG AA. It is used exclusively for: SVG icons, chart bar/pie series, 3px left-border accents on chain rows, and decorative elements. On dark backgrounds it's fine (>7:1 contrast).

### Surface & Text
| Token | Light | Dark |
|-------|-------|------|
| `--bg` | `#F6FAFE` | `#08151F` |
| `--surface` | `#FFFFFF` | `#0F2338` |
| `--surface-2` | `#E9F3FA` | `#162D47` |
| `--border` | `#D6E6F3` | `#1E3A52` |
| `--text` | `#0F1923` | `#E8F4FF` |
| `--text-muted` | `#5B7A93` | `#7B9CB5` |

Contrast checks:
- `--text` on `--surface`: ≥ 15:1 in both modes ✅
- `--text-muted` on `--surface`: ≥ 4.5:1 in both modes ✅ (WCAG AA)
- White on `--primary` (#02416E): ≥ 9:1 ✅
- `--primary` (#5BAEE0) on dark `--surface`: ≥ 5:1 ✅

### Semantic Colors
| Token | Light | Dark | Use |
|-------|-------|------|-----|
| `--success` | `#16a34a` | `#4ade80` | Enrollment, compliance clean, checklist pass |
| `--success-bg` | `rgba(22,163,74,0.10)` | `rgba(74,222,128,0.12)` | |
| `--warning` | `#b45309` | `#fbbf24` | Partial, hesitation (dark mode uses amber, light uses darker amber for AA) |
| `--warning-bg` | `rgba(180,83,9,0.10)` | `rgba(251,191,36,0.12)` | |
| `--danger` | `#dc2626` | `#f87171` | Compliance flags, drop-off, not-done |
| `--danger-bg` | `rgba(220,38,38,0.10)` | `rgba(248,113,113,0.12)` | |

---

## 3. Typography

Single font family: **Inter** (already loaded). Drop `Outfit` entirely.

| Element | Weight | Size | Notes |
|---------|--------|------|-------|
| Brand mark "FREED AI" | 800 | 1rem | Navy `--primary`, tight tracking |
| Page title (top bar) | 600 | 1rem | `--text` |
| Section headings | 600 | 1.4rem | `--text` |
| Panel headers | 600 | 0.8rem | Uppercase, `0.6px` letter-spacing, `--text-muted` |
| Body text | 400 | 0.875rem | `--text`, `1.6` line-height |
| Muted labels | 500 | 0.72rem | Uppercase, `--text-muted` |
| KPI values | 700 | 1.75rem | `--primary` |
| Timestamps/mono | 400 | 0.78rem | `ui-monospace, 'Geist Mono', monospace` |

---

## 4. Spacing Scale

Base: 4px. Scale: `4 · 8 · 12 · 16 · 20 · 24 · 32 · 40 · 48 · 64px`  
All padding, gap, and margin values snap to this scale. No arbitrary values.

---

## 5. Layout

### Sidebar (Collapsible)
- **Expanded width:** 220px
- **Collapsed width:** 56px (icon only)
- **Background:** `--surface` with 1px `--border` right edge
- **Toggle:** `[←]`/`[→]` chevron at top-right of sidebar header
- **Collapsed icons:** Show tooltips via `title` attribute
- **Nav items:**
  - Default: `--text-muted`, transparent bg
  - Hover: `--surface-2` bg, `--text` color
  - Active: `--primary` bg, `#fff` text, `8px` radius
  - Padding: `10px 14px`, gap between icon and label: `10px`
- **Bottom section:** No theme toggle here (moved to top bar)
- **Persistence:** `localStorage` key `sidebar-state`

### Top Bar (56px height)
- **Background:** `--surface`, `1px --border` bottom
- **Left:** Dynamic page title — updates as views switch via `switchView()`
- **Right:** Theme toggle — sun icon (light mode) / moon icon (dark mode)
- **Theme toggle:** Pill button, `--surface-2` bg, `--text-muted` icon, hover `--text`
- **Persistence:** `localStorage` key `theme`; applied before first paint (no FOUC)

### Main Content
- `padding: 24px 32px`
- No ambient blobs (remove `.blob` divs from HTML)
- `overflow-y: auto`, `height: calc(100vh - 56px)` (accounts for top bar)

---

## 6. Per-View Layouts

### View: Global Analytics (Overview)
- KPI grid: `repeat(auto-fit, minmax(160px, 1fr))`, `gap: 16px`
- Chart containers: `--surface` bg, `1px --border`, `border-radius: 12px`, `padding: 20px`
- Chart palette: primary `#02416E`, accent `#F16E20`, series `#5BAEE0` / `#4ade80` / `#fbbf24`

### View: New Transcription (Upload)
- Mode tab switcher: max-width `600px`, centered, `--surface-2` bg container
- Active tab: `--primary` bg, white text
- Drop zone: `max-width: 520px`, centered, `2px dashed --border`, `border-radius: 14px`
- Drag-over: border → `--primary`, bg → `--surface-2`
- Queue rows: clean list, `--border` bottom, status pill, `--primary` progress bar
- Chain panel: all existing IDs preserved, form fields restyled

### View: Job History
- Replaces card grid with a unified row list
- **Single call row:** icon · filename · engine badge · duration · date · View button
- **Chain row:** chain icon · label · "Chain · N Calls" badge · duration · date · View Combined + Members buttons
- Chain rows: `3px solid --accent` left border (FREED orange — visible on both modes)
- Members expand panel: `--surface-2` bg, numbered rows, individual calls clickable
- Hover: `--surface-2` bg on row
- Empty state: centered `--text-muted` message

### View: Call Analysis (Dashboard)
- 2-column panel layout preserved
- Panel header: `--surface-2` bg strip, `border-bottom: 1px solid --border`
- `dash-title`, `dash-engine`, `dash-timeline`, `dash-analytics` IDs untouched
- Transcript toggle button: clean pill style
- Collapsible `<details>` panels (Audit, Checklist, Emotion, Sarcasm, Triggers): chevron rotates 180° on open

---

## 7. Components

### Cards / KPI Cards
```
background: var(--surface)
border: 1px solid var(--border)
border-radius: 10px
padding: 16px 20px
transition: border-color 200ms, box-shadow 200ms
```
Hover: `border-color: --primary (40% opacity)`, `box-shadow: 0 2px 12px rgba(2,65,110,0.08)`

### Buttons
| Variant | Bg | Text | Border |
|---------|----|------|--------|
| Primary | `--primary` | `#fff` | none |
| Secondary | transparent | `--text` | `1px solid --border` |
| Ghost | transparent | `--text-muted` | none |
| Danger | `--danger-bg` | `--danger` | `1px solid --danger (30%)` |

All: `border-radius: 8px`, `padding: 9px 18px`, `font-weight: 600`, `transition: all 150ms ease`

### Badges / Pills
`border-radius: 20px` · `padding: 3px 10px` · `font-size: 0.72rem` · `font-weight: 600`

### Form Inputs (Chain mode)
```
background: var(--surface)
border: 1px solid var(--border)
border-radius: 8px
padding: 10px 14px
color: var(--text)
```
Focus: `border-color: --primary`, `box-shadow: 0 0 0 3px var(--primary-bg)`

---

## 8. Interactions & Animations

- **View fade:** `opacity 0→1, translateY 10→0, 300ms ease` (existing, keep)
- **Sidebar collapse:** `width 200ms ease`, labels `opacity 150ms`
- **Theme switch:** `background/color/border-color 200ms ease` on `:root`
- **Hover states:** `150ms ease` uniformly
- **Details chevron:** `rotate(180deg) 200ms ease`
- **Drop zone drag:** border + bg `150ms ease`

---

## 9. Dark Mode

Toggle class `.dark` on `<body>`. All tokens redefined under `body.dark { }`.  
Applied via `localStorage` before DOM paint to prevent flash:
```html
<script>
  if (localStorage.getItem('theme') === 'dark') document.body.classList.add('dark');
</script>
```
Placed immediately after `<body>` tag — before any layout rendering.

---

## 10. What Does NOT Change

- All JavaScript functions: `switchView()`, `initOverview()`, `loadHistoryGrid()`, `renderDashboard()`, `renderChainDashboard()`, `setUploadMode()`, `submitChain()`, `pollChainJob()`, `toggleMetaStats()`, `toggleTranscriptView()`, etc.
- All element IDs referenced by JS
- All `fetch()` API calls
- Chart.js canvas elements and initialization logic (only Chart.defaults colors updated)
- `<details>` open/close behavior
- Drag-and-drop file handlers
- Queue polling and progress update logic

---

## 11. Removed

- `.blob`, `.blob-1`, `.blob-2` div elements and their CSS
- `backdrop-filter: blur()` glassmorphism on all surfaces
- `font-family: 'Outfit'` (replaced with Inter)
- Static inline `style=""` attributes in HTML markup migrated to CSS classes; JS-generated inline styles (dynamically set via `.style.cssText`, `.style.display`, etc.) left as-is to avoid breaking functionality
- Purple `#8b5cf6` and cyan `#38bdf8` accent colors
- Gradient text effects on brand/headings
