FROM zenmldocker/zenml

WORKDIR /ml

COPY . /ml

RUN pip install -r /ml/requirements.txt

# replace the file name of the pipeline you want to run
CMD ["python", "run_XGB_pipeline.py"]