
# RealityStream Development Notes

## Overview
RealityStream is a machine learning pipeline and web interface for exploring parameterized datasets (bees, blinks, industries, GDC, etc.).  
This document captures technical implementation notes, common issues, and UI/UX decisions relevant to development.

---

## Recent UI Enhancements

### Parambase Dropdown
- **Copy shareable link button**:  
  - Copies the current `location.href` (including the hash state) to the clipboard.  
  - Shows a **temporary “Copied!” status flash** next to the button for feedback.  
  - Implemented using `navigator.clipboard.writeText(location.href)`.  
- **Open YAML button**:  
  - Dynamically points to the currently selected `.yaml` file.  
  - Updates automatically on dropdown selection changes.  
  - Opens in a new tab (`target="_blank"`, `rel="noopener"`).  

**UI layout example:**
```

\[Base parameters:] \[Dropdown ▼] \[Copy shareable link] \[Open YAML]

```

### Technical Implementation
- Implemented in **`cloud/run/js/param-input.js`** inside `loadBaseParamsSelect()`.
- DOM elements created:
  - `<label id="parambase-label">Base parameters:</label>`
  - `<button id="parambase-copy">Copy shareable link</button>`
  - `<a id="parambase-yaml" target="_blank">Open YAML</a>`
- Idempotent logic: prevents duplicate elements if re-run.
- Event handlers:
  - Copy button → copies URL + flashes status.
  - Dropdown change → updates YAML href + hash state + localStorage.

---

## Reset Button
- **Purpose**: Quickly clear user state when debugging or testing.
- **Functionality**:
  - Clears dropdown selection.
  - Removes `parambase/value` and `parambase/url` from localStorage.
  - Clears the URL hash completely.
  - Reloads the page back to default.
- **Implementation**:
  - Added next to the dropdown UI, styled consistently with other buttons.
  - Event handler resets hash/localStorage and calls `location.reload()`.
- **Visibility Decision**:
  - Dev convenience tool, safe for production.  
  - Could be hidden via CSS if not needed in end-user builds.

---

## GDC Integration
- **Default parambase**:  
  - `Google Data Commons (GDC)` moved to first row of `parameters/parameter-paths.csv`.  
  - Default selection logic ensures it loads automatically when no `#parambase` is in the URL.
- **YAML link**:  
  - `parameters-gdc.yaml` added as selectable option in the dropdown.  
  - Loads GDC features/targets via `dcid` integration in Colab.

---

## Common Issues & Fixes

### CSV Loading
- **UTF-8 BOM Characters**:  
  - Some CSV files may contain a BOM prefix.  
  - Fix: Strip using `csvText.replace(/^\uFEFF/, '')` before parsing.
- **Cache Invalidation**:  
  - Browser often caches CSV aggressively.  
  - Fix: Use `fetch(url, { cache: 'no-store' })` during development.

### State Management
- **Priority order**:  
  - URL hash → localStorage → default first CSV row.  
- **Custom Path**:  
  - If `custom` is selected, URL hash stores `yaml=` parameter for reproducibility.  
- **Copy Links**:  
  - Always produce shareable links that fully restore the dropdown state.

---

## Debugging Tips
- Use Chrome/Firefox DevTools → **Network tab** → verify `parameter-paths.csv` loads with `cache-control: no-store`.  
- Check **Console logs**: `loadBaseParamsSelect()` prints key steps.  
- Use `performance.getEntriesByType('resource')` in console to confirm CSV requests.  
- If dropdown misbehaves, clear `localStorage` and reload.

---

## File Structure
- `/parameters/parameter-paths.csv` → master list of YAML configs.  
- `/cloud/run/js/param-input.js` → UI logic for dropdown, copy, reset, YAML link.  
- `/docs/dev-notes.md` → This file.  
