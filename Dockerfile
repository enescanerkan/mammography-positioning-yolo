FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV TZ=Asia/Istanbul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN echo "PYTHONUNBUFFERED=1" >> /etc/environment && \
    echo "OMP_NUM_THREADS=1" >> /etc/environment

WORKDIR /root

RUN apt-get update && apt-get install -y --no-install-recommends \
        nano \
        cmake \
        unzip \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
        libsm6 \
        libxext6 \
        swig \
        wget \
        bzip2 \
        g++ \
        make \
        git \
        iputils-ping \
        telnet \
 && rm -rf /var/lib/apt/lists/*

RUN pip install python-gdcm==3.0.24.1

RUN groupadd -g 2000 runner && \
    useradd -m -s /bin/bash -u 2000 -g runner runner

ENV LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"

WORKDIR /docker

COPY ./requirements.txt /docker/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /docker/requirements.txt

COPY . /docker/

RUN python download_models.py --output_dir /docker/weights

RUN chown -R runner:runner /docker
RUN chmod -R 700 /docker

USER runner

EXPOSE 5008

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5008"]
