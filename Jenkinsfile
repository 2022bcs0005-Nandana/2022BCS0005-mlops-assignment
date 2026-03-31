pipeline {
    agent {
        docker {
            image 'python:3.10'
        }
    }

    stages {
        stage('Checkout') {
            steps {
                echo 'Checking out code...'
                checkout scm
            }
        }

        stage('Install Dependencies') {
            steps {
                sh 'pip install --upgrade pip'
                sh 'pip install fastapi uvicorn scikit-learn mlflow "dvc[s3]" pandas numpy joblib boto3'
            }
        }

        stage('DVC Pull') {
            steps {
                sh 'dvc pull'
            }
        }

        stage('Train Model') {
            steps {
                sh 'python src/train.py --run_name "jenkins_run" --dataset_version "v2" --model_type "random_forest" --n_estimators 100 --feature_set "all"'
            }
        }

        stage('Generate Metrics') {
            steps {
                sh 'cat models/metrics.json'
            }
        }
    }
}