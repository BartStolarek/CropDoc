# Crop and State detection

## Introduction

This project is a part of the course "Algorithms in Machine Learning" at the University of New England, Armidale Australia. The goal of this project is to detect the crop and its state. The state of the crop can be 'healthy' or one of the various diseases or pest infestations associated with that crop. The dataset used for this project is the "Dataset for Crop Pest and Disease Detection" dataset which contains images of crops in different states. The dataset can be downloaded from [here](https://data.mendeley.com/datasets/bwh3zbpkpv/1).

## Installation

### Prerequisites

Before you begin, ensure you have the following installed:
- Docker
- Docker Compose

If you do not you can find instructions to install them [here](https://docs.docker.com/get-docker/).

### Project Structure

COSC320-GROUPD/
├── CropDoc/          # Backend (Python Flask)
├── frontend/         # Frontend (Next.js with Tailwind & NextUI)
├── Dockerfile.backend
├── Dockerfile.frontend
└── docker-compose.yml

### Docker Setup and Usage

#### Building and Starting the Application

To build and start both the frontend and backend containers:

```docker-compose up --build```

This command builds the Docker images and starts the containers. Use this when running for the first time or after making changes to the Dockerfiles.

#### Starting the Application
If you've already built the images and just want to start the containers:

```docker-compose up```

- Locally the frontend will be available at:
    - [http://localhost:3000](http://localhost:3000)
- Locally the backend will be available at:
    - [http://localhost:5000](http://localhost:5000)

#### Stopping the Application

To stop the running containers:

```docker-compose down```

#### Custom Backend Commands (Command Line Interface)

The backend application includes several custom commands that can be run using the 'manage.py' script. To execute these commands, you need to access the backend container first.

1. Access the backend container:

```docker-compose exec backend bash```

2. Once inside the container, you can run the following commands:

    - Hello World test:
    ```python manage.py hello```

    - Run formatters over the code:
    ```python manage.py format```

    -
    ```python manage.py train```

    ```python manage.py process```

    ```python manage.py runserver```

