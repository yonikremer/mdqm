# The mock internal API (task D04: `src/mdqm/testing/mock_api.py`) as a container, for the `api`
# compose profile. It is the only image this repo builds; everything else is a published image.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Copy the package first so the dependency layer is cached while the source changes.
COPY pyproject.toml README.md ./
COPY src ./src

# `[dev]` brings fastapi + uvicorn, which the mock imports. Needs network access at build time on
# the developer machine only; the running container talks to nothing outside itself.
RUN pip install --no-cache-dir ".[dev]"

ENV MDQM_MOCK_API_HOST=0.0.0.0 \
    MDQM_MOCK_API_PORT=8000

EXPOSE 8000

# `--serve` keeps the app on a real port; the API key comes from MDQM_MOCK_API_KEY, and with no key
# every request is refused (fail closed).
CMD ["python", "-m", "mdqm.testing.mock_api", "--serve"]
