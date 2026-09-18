# Tarot Guanche — AI Interpreter backend

Small Node/Express service that powers the "AI Interpreter" section of
tarotguanche.com. It receives a photo of a spread, identifies the cards
with Claude's vision model, validates them against the real 80-card deck,
and writes an interpretation with Claude's text model — strictly scoped
to Tarot Guanche readings.

## Why a backend at all?

The `ANTHROPIC_API_KEY` must never be embedded in the website's HTML/JS —
anyone could copy it out of the page source and run up charges on your
account. This service holds the key server-side; the website only ever
talks to *this* server, never directly to Anthropic.

## Setup

```bash
npm install
cp .env.example .env
# edit .env: paste your Anthropic API key, set ALLOWED_ORIGIN to your real domain
npm start
```

Runs on `http://localhost:3001` by default. One endpoint, two modes:

```
POST /api/interpret-reading

# Text mode (default on the site):
{ "mode": "text", "textSpread": "Saqué 3 cartas: El Saltador derecha, La Maguada invertida...",
  "lang": "es" | "en", "question": "optional theme, e.g. 'mi trabajo'" }

# Photo mode:
{ "mode": "photo", "image": "data:image/jpeg;base64,...",
  "lang": "es" | "en", "question": "optional theme" }
```

`question` is always optional in both modes. It's passed to the model
explicitly labeled as a thematic focus, never as instructions — the
system prompt is written so the model ignores anything in that field
that isn't a theme/question about the reading itself.

## Deploying

This is a standard Node app — it needs a host that runs Node, not plain
static hosting (shared hosting that only serves HTML files won't work
for this part; the `index.html` itself can still live anywhere, including
your existing Bluehost hosting).

### Option A — Bluehost / cPanel (if available on your plan)

1. In cPanel, look under **Software** for an icon called **Setup Node.js App**.
   Not every Bluehost plan includes this (it depends on your hosting tier) —
   if you don't see it, skip to Option B below.
2. Click **Create Application**:
   - **Node.js version**: pick the latest available (18.x or 20.x)
   - **Application mode**: Production
   - **Application root**: e.g. `tarot-ai-backend`
   - **Application URL**: a subdomain or path, e.g. `api.tarotguanche.com`
     or `tarotguanche.com/api` — a **subdomain is simpler** here since it
     keeps the API cleanly separate from your static site
   - **Application startup file**: `server.js`
3. Upload the contents of this folder into that application root via
   cPanel's **File Manager** (or FTP) — everything except `.env`.
4. Back in **Setup Node.js App**, open your app and find the
   **Environment Variables** section. Add `ANTHROPIC_API_KEY` and
   `ALLOWED_ORIGIN` there (values from `.env.example`) — this replaces
   uploading a real `.env` file, which is safer.
5. Click **Run NPM Install** in the app's control panel.
6. Click **Restart**. cPanel will show the actual URL it's serving on —
   that's what goes into `API_BASE` in `index.html`.

Note: cPanel runs your app through Passenger, which supplies its own port —
`server.js` already accounts for this, so no changes needed on your side.

### Option B — a small external Node host (if Bluehost doesn't offer it)

Many Bluehost shared plans don't include the Node.js Selector. If yours
doesn't, the simplest fix is to run *just this backend* on a small free/cheap
Node host — Render or Railway are the easiest to get started with — while
`index.html` keeps living on Bluehost exactly as it does now:

1. Push this folder to a GitHub repo (or upload directly if the platform allows it).
2. On Render/Railway: create a new **Web Service** from that repo.
   - Build command: `npm install`
   - Start command: `npm start`
   - Add the two environment variables from `.env.example` in their dashboard.
3. They'll give you a live URL (e.g. `https://tarot-ai.onrender.com`) —
   that's your `API_BASE`.

Either way, once you have the live backend URL, open `index.html`, find:
```js
var API_BASE = 'https://YOUR-BACKEND-DOMAIN.example.com';
```
and replace it with your real one, then re-upload `index.html` to Bluehost
as usual.

## Cost control

- Vision step uses Haiku (cheap) — just OCR + orientation, no interpretation.
- Text step uses Sonnet, and only receives the validated card data (a few
  hundred words), not the photo — keeps that call small too.
- `express-rate-limit` caps each IP to 30 requests / 15 min by default —
  tune `server.js` if you expect more legitimate traffic than that, or
  less (to control cost from casual visitors).
- Exact current per-token pricing: https://docs.claude.com (search "pricing").

## Files

- `server.js` — the API
- `cards.json` — all 78 cards' names + upright/reversed meanings (ES/EN),
  extracted from your site's data. Re-export this if you ever update a
  card's wording on the site, so the two stay in sync.
- `spreads.json` — position meanings for the 1-card, 3-card, and Celtic
  Cross (10-card) layouts. Any other card count falls back to a generic
  "card N of N" framing in the prompt — add more named spreads here if
  you want richer position language for a spread size you use often.
