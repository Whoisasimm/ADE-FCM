FROM python:3.9-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-11-jre-headless \
    curl \
    wget \
    procps \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

WORKDIR /opt/ade-fcm

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM base AS spark-base

ENV SPARK_VERSION=3.3.4 \
    HADOOP_VERSION=3 \
    SPARK_HOME=/opt/spark

RUN curl -fsSL "https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz" \
    | tar -xzC /opt/ && \
    ln -s "/opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}" "${SPARK_HOME}"

ENV PATH="${SPARK_HOME}/bin:${SPARK_HOME}/sbin:${PATH}" \
    SPARK_MASTER_PORT=7077 \
    SPARK_MASTER_WEBUI_PORT=8080 \
    SPARK_WORKER_PORT=8888 \
    SPARK_WORKER_WEBUI_PORT=8081

FROM spark-base AS ade-fcm

COPY . /opt/ade-fcm
RUN pip install --no-cache-dir -e /opt/ade-fcm

EXPOSE 7077 8080 8081 8888 4040

FROM ade-fcm AS jupyter

RUN pip install --no-cache-dir jupyterlab notebook

EXPOSE 8888

FROM ade-fcm AS training

ENTRYPOINT ["spark-submit", "--master", "spark://spark-master:7077", \
    "--packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.4", \
    "/opt/ade-fcm/train.py"]

FROM ade-fcm AS evaluation

ENTRYPOINT ["spark-submit", "--master", "spark://spark-master:7077", \
    "/opt/ade-fcm/evaluate.py"]
