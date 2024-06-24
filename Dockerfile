FROM zenmldocker/zenml

WORKDIR /ml

COPY . /ml

RUN pip install -r /ml/requirements.txt

CMD ["python", "run_SVM_pipeline.py"]