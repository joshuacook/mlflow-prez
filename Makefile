pdf:
	pandoc presentation.md -t beamer -o presentation.pdf

mlflow-server:
	mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0