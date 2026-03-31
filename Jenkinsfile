pipeline {
    agent any

    environment {
        PYTHON = 'python'
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
                bat 'pip install fastapi uvicorn scikit-learn mlflow dvc[s3] pandas numpy joblib boto3'
            }
        }

        stage('DVC Pull') {
            steps {
                echo 'Pulling data from DVC...'
                bat 'dvc pull'
            }
        }

        stage('Train Model') {
            steps {
                echo 'Training model...'
                bat 'python src/train.py --run_name "jenkins_run" --dataset_version "v2" --model_type "random_forest" --n_estimators 100 --feature_set "all"'
            }
        }

        stage('MLflow Logging') {
            steps {
                echo 'MLflow logging done during training!'
            }
        }

        stage('Generate Metrics') {
            steps {
                echo 'Metrics generated!'
                bat 'type models\\metrics.json'
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