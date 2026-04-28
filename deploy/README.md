# Deploying the API (free tier friendly)

This folder contains a **CPU-only** Docker image (`Dockerfile.cpu`) for hosting the FastAPI app on Render, Fly.io, or Railway. Models are fetched from **Google Drive during the image build** (`download_models.py`).

**Important:** Free tiers are tight on RAM (~512 MB on Render). Loading two YOLO weights may **fail with out-of-memory** on the smallest plans. If the deploy crashes on startup, upgrade to a **512 MB+ or 1 GB** instance or test locally with Docker first.

## What you get in the browser

- `https://<your-service>.onrender.com/docs` — Swagger UI (try `/health`, `/predict`)
- Not the PyQt desktop GUI — only the REST API.

---

## Option A — Render (often the simplest)

1. Push this repo to GitHub (branch with API + `deploy/` merged).
2. Go to [render.com](https://render.com) → sign up (free).
3. **New** → **Blueprint** → connect the repository → Render reads [`render.yaml`](../render.yaml) at the repo root.
4. Confirm the **Web Service** uses:
   - Dockerfile path: `deploy/Dockerfile.cpu`
   - Docker context: `.`
5. Deploy and wait (first build can take **15–25 minutes**: PyTorch + weights).
6. Open the service URL + `/docs`.

**Manual Web Service (without Blueprint):** New → Web Service → Docker → set Dockerfile path to `deploy/Dockerfile.cpu`, root directory `.`.

---

## Option B — Fly.io

1. Install [flyctl](https://fly.io/docs/hands-on/install-flyctl/).
2. Edit [`fly.toml`](../fly.toml): change `app = "mammography-positioning-api"` to a **unique** name.
3. From the repo root:

```bash
fly launch --dockerfile deploy/Dockerfile.cpu --no-deploy
fly deploy
```

4. Open `https://<app-name>.fly.dev/docs`.

Free allowance changes over time; check [Fly pricing](https://fly.io/docs/about/pricing/).

---

## Option C — Railway

1. [railway.app](https://railway.app) → New Project → Deploy from GitHub.
2. Select the repo → **Settings**:
   - **Dockerfile path:** `deploy/Dockerfile.cpu`
   - **Root directory:** `/` (repository root)
3. Deploy. Railway sets `PORT` automatically; the image already uses `CMD` with `$PORT`.

---

## Local smoke test (Docker)

```bash
docker build -f deploy/Dockerfile.cpu -t mmg-api .
docker run --rm -p 8000:8000 -e PORT=8000 mmg-api
```

Then open `http://localhost:8000/docs`.

### Calling with uploaded DICOMs (Bruno / curl)

Use **POST `/predict/upload`** — `multipart/form-data` with two files and optional form fields:

- **mlo_dicom** — file (MLO `.dcm`)
- **cc_dicom** — file (CC `.dcm`)
- **laterality** — form field, default `L`
- **pixel_spacing** — form field, default `0.085`
- **threshold_mm** — form field, default `10.0`

**curl example:**

```bash
curl -X POST "https://YOUR_HOST/predict/upload" \
  -F "mlo_dicom=@/path/to/mlo.dcm" \
  -F "cc_dicom=@/path/to/cc.dcm" \
  -F "laterality=L" \
  -F "pixel_spacing=0.085" \
  -F "threshold_mm=10.0"
```

In **Bruno**: method **POST**, URL `.../predict/upload`, **Body → multipart/form-data**, add two file fields named exactly `mlo_dicom` and `cc_dicom`, plus optional text fields above.

---

## Environment variables (optional)

| Variable | Purpose |
|----------|---------|
| `MLO_MODEL_PATH` | Override path to MLO `.pt` |
| `CC_MODEL_PATH` | Override path to CC `.pt` |

Defaults: `/app/weights/*.pt` inside the container.
