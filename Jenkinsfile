pipeline {
    agent any

    environment {
        PYTHON = 'python3'
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
                echo 'Installing dependencies...'
                sh 'pip install fastapi uvicorn scikit-learn mlflow "dvc[s3]" pandas numpy joblib boto3'
            }
        }

        stage('DVC Pull') {
            steps {
                echo 'Pulling data from DVC...'
                sh 'dvc pull'
            }
        }

        stage('Train Model') {
            steps {
                echo 'Training model...'
                sh 'python3 src/train.py --run_name "jenkins_run" --dataset_version "v2" --model_type "random_forest" --n_estimators 100 --feature_set "all"'
            }
        }

        stage('Generate Metrics') {
            steps {
                echo 'Metrics generated!'
                sh 'cat models/metrics.json'
            }
        }
    }

    post {
        success {
            echo 'Pipeline completed successfully!'
        }
        failure {
            echo 'Pipeline failed!'
        }
    }
}