# ClipFinder Azure Functions - Service Bus indexing consumers
FROM mcr.microsoft.com/azure-functions/python:4-python3.11

ENV AzureWebJobsScriptRoot=/home/site/wwwroot \
    AzureFunctionsJobHost__Logging__Console__IsEnabled=true \
    PYTHONPATH=/home/site/wwwroot/backend

WORKDIR /home/site/wwwroot

COPY backend/ ./backend/
COPY functions/host.json ./
COPY functions/function_app.py ./
COPY functions/requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*