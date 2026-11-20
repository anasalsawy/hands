# Hands, hosted: the web + mobile app and a cloud workspace for the cards.
FROM node:22-bookworm-slim AS ui
WORKDIR /src
COPY desktop/package.json desktop/package-lock.json desktop/
RUN npm ci --prefix desktop
COPY desktop desktop
RUN npm run build --prefix desktop

FROM node:22-bookworm-slim AS engine
WORKDIR /src
COPY engine/package.json engine/package-lock.json engine/
RUN npm ci --prefix engine

FROM python:3.12-slim-bookworm
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HANDS_CLOUD=1 HANDS_DATA_DIR=/data HANDS_SUPERVISED=0
RUN apt-get update && apt-get install -y --no-install-recommends \
      git ca-certificates chromium fonts-liberation fonts-noto-color-emoji curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=engine /usr/local/bin/node /usr/local/bin/node
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt && pip install playwright
COPY . .
COPY --from=engine /src/engine/node_modules engine/node_modules
COPY --from=ui /src/desktop/dist ui
RUN python scripts/install_browser_sdk.py || echo "browser SDK not installed: Browser cards will be unavailable"
RUN useradd -m -u 1000 hands && mkdir -p /data && chown -R hands /data /app
USER hands
EXPOSE 8800
CMD ["python", "hands_app.py", "--cloud", "--ui-dir", "/app/ui"]
