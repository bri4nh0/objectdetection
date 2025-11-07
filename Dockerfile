FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install --upgrade pip && pip install -r requirements.txt || pip install -r requirements_enhanced.txt
# default command runs demo smoke
CMD ["python", "scripts/run_demo.py"]
