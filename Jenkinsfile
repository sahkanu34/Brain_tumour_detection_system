pipeline {
    agent any

    environment {
        DOCKERHUB_REPO = 'sahkanu37/brain_tumour_detection'
        IMAGE_TAG = "latest"
        DOCKERHUB_CREDENTIALS = credentials('docker-hub-credentials')
    }

    stages {

        stage('Checkout Code') {
            steps {
                git branch: 'main', url: 'https://github.com/sahkanu34/Brain_tumour_detection_system.git'
            }
        }

        stage('Install Dependencies') {
            steps {
                sh 'python3 -m venv venv && venv/bin/pip install -r requirements.txt'
            }
        }

        stage('Run Tests') {
            steps {
                echo 'Running unit tests...'
                sh 'venv/bin/pytest tests/' 
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    dockerImage = docker.build("${DOCKERHUB_REPO}:${IMAGE_TAG}")
                }
            }
        }

        stage('Push to Docker Hub') {
            steps {
                withCredentials([usernamePassword(credentialsId: 'docker-hub-credentials', usernameVariable: 'DOCKER_USER', passwordVariable: 'DOCKER_PASS')]) {
                    sh """
                        echo "$DOCKER_PASS" | docker login -u "$DOCKER_USER" --password-stdin
                        docker push ${DOCKERHUB_REPO}:${IMAGE_TAG}
                    """
                }
            }
        }
    }

    post {
        success {
            echo "✅ Successfully pushed to Docker Hub: ${DOCKERHUB_REPO}:${IMAGE_TAG}"
        }
        failure {
            echo "❌ Build or test failed. See logs."
        }
    }
}
