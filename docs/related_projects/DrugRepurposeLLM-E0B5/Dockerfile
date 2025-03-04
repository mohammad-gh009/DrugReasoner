FROM continuumio/miniconda3

WORKDIR /app

ENV PATH /opt/conda/bin:$PATH

RUN apt-get update && apt-get install -yq curl wget jq vim

COPY environment.yml /app/environment.yml
RUN conda env create -f /app/environment.yml

SHELL ["conda", "run", "-n", "python_env", "/bin/bash", "-c"]
RUN pip install git+https://github.com/bp-kelley/descriptastorus "flaml[automl]"

COPY . /app

EXPOSE 9999
CMD [ "/bin/bash", "-c", "source activate python_env && cd agents && nohup jupyter lab --ip=0.0.0.0 --port=9999 --no-browser --allow-root --NotebookApp.token='' > jupyter.log 2>&1 & bash" ]
