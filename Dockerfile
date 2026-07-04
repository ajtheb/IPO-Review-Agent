FROM python:3.13-slim

# System dependencies:
#   - default-jre-headless: required by tabula-py (PDF table extraction)
#   - google-chrome-stable: required by Selenium for the SEBI document search tab
#   - wget/gnupg/ca-certificates: needed to add Google's apt repo for Chrome
#   - build-essential/libxml2-dev/libxslt1-dev: needed to build lxml from source
RUN apt-get update && apt-get install -y --no-install-recommends \
    default-jre-headless \
    wget \
    gnupg \
    ca-certificates \
    build-essential \
    libxml2-dev \
    libxslt1-dev \
    && wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /usr/share/keyrings/google-chrome-keyring.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome-keyring.gpg] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 8501

# Render (and most PaaS providers) inject $PORT at runtime; default to 8501
# for local `docker run`. Shell form so ${PORT} is expanded.
CMD streamlit run app.py \
    --server.port=${PORT:-8501} \
    --server.address=0.0.0.0 \
    --server.headless=true
