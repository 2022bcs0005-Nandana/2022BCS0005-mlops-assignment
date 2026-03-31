pipeline {
    agent any

    environment {
        AWS_ACCESS_KEY_ID = credentials('aws-access-key-id')
        AWS_SECRET_ACCESS_KEY = credentials('aws-secret-access-key')
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Install Dependencies') {
            steps {
                sh '''
                apt-get update -y || true
                apt-get install -y python3 python3-pip || true
                pip3 install fastapi uvicorn scikit-learn mlflow "dvc[s3]" pandas numpy joblib boto3 --break-system-packages || pip3 install fastapi uvicorn scikit-learn mlflow "dvc[s3]" pandas numpy joblib boto3
                '''
            }
        }

        stage('DVC Pull') {
            steps {
                sh '''
                dvc remote modify myremote access_key_id $AWS_ACCESS_KEY_ID
                dvc remote modify myremote secret_access_key $AWS_SECRET_ACCESS_KEY
                dvc pull
                '''
            }
        }

        stage('Train Model') {
            steps {
                sh 'python3 src/train.py --run_name "jenkins_run" --dataset_version "v2" --model_type "random_forest" --n_estimators 100 --feature_set "all"'
            }
        }

        stage('Generate Metrics') {
            steps {
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