# Crop and State detection

## Introduction

This project is a part of the course "Information Technology Project" at the University of New England, Armidale Australia. The goal of this project is to detect the crop and its state. The state of the crop can be 'healthy' or one of the various diseases or pest infestations associated with that crop. The dataset used for this project is the "Dataset for Crop Pest and Disease Detection" dataset which contains images of crops in different states. The dataset can be downloaded from [here](https://data.mendeley.com/datasets/bwh3zbpkpv/1).

## Prerequisites

Before you begin, ensure you have the following installed:
- Docker
- Docker Compose

or

- apptainer (if you are not using Docker) - see installation instructions below in the Apptainer section

If you do not have them installed you can find instructions to install them [here](https://docs.docker.com/get-docker/).

## Project Structure

The project is structured in to de-coupled frontend and backend components. The frontend is built using Next.js with Tailwind CSS and NextUI. The backend is built using Python Flask. The project is containerized using Docker and Docker Compose.

```
COSC320-GROUPD/
├── CropDoc/          # Backend (Python Flask)
├── frontend/         # Frontend (Next.js with Tailwind & NextUI)
├── Dockerfile.prod.backend
├── Dockerfile.prod.frontend
├── Dockerfile.dev.backend
├── Dockerfile.dev.frontend
├── docker-compose.prod.yml
└── docker-compose.dev.yml
```

# Docker
## Docker Setup

### Development Workflow/Mode

1. Start containers (first time or after Dockerfile/dependency changes):
    
    ```docker-compose -f docker-compose.dev.yml up --build```

2. For subsequent development sessions:
    
    ```docker-compose -f docker-compose.dev.yml up```

3. Make changes to the code. These changes should be reflected immediately due to volume mounting and hot reloading.

4. If you make changes that require a rebuild (like changing dependencies), stop the containers and use ```--build``` again.

### Production Mode

To start the application in production mode:

    ```docker-compose -f docker-compose.prod.yml up --build```


### Stopping the Application

To stop the running containers:

```
docker-compose -f docker-compose.dev.yml down  # For development mode
docker-compose -f docker-compose.prod.yml down  # For production mode
```

## Accessing and Using the Application

There are two methods to use the application, the most useful method is using the command line interface (CLI), the second method which is less useful but more user-friendly is using the frontend. 

To use the frontend, you can access it by navigating to the following URL in your browser for a locally hosted application
- [http://localhost:3000](http://localhost:3000)

To use the CLI, you need to make sure you have start the application in development mode, or production mode. Then you can open a new terminal, further instructions are below.

### Custom Backend Commands (Command Line Interface)

The backend application includes several custom commands that can be run using the 'manage.py' script. To execute these commands, you need to start the application (instructions above), open a new terminal and access the backend container first.

1. Access the backend container:

    ```docker-compose -f docker-compose.dev.yml exec backend bash```

    Note, change 'dev' to 'prod' if you are using the production mode.

2. Once inside the container, you can run the following commands:

    - Hello World test:
    
        ```python manage.py hello```

    - Run formatters over the code:
    
        ```python manage.py format```
    
    - Train the model:
    
        ```python manage.py train -c <config_file>```
    
    - Make a prediction using the model:

        ```python manage.py predict -c <config_file> -i <image_path>```

    - Test the model using dataset in the config file:
    
        ```python manage.py test -c <config_file_name>```


    - Predict many images using dataset in the config file:
    
        ```python manage.py predict_many -c <config_file>```


    - Produce evaluation metrics, plots and details:
    
        ```python manage.py evaluate -c <config_file>```  

    - For developers, unit tests can be run using the following command:
    
        ```python manage.py unittest```

        ```python manage.py unittest --coverage``` Add coverage flag to get coverage report.
        
        ```python manage.py unittest --filename <filename>``` Add filename flag to run a specific test file.

    - For developers, to run the formatter over the code:
    
        ```python manage.py format``` 
     

    Config file is a yml file that contains the configuration for the training process, located in CropDoc/app/config/ directory, just use the name of the file, not the full path.
    Optionally, you can add in additional key word (kwargs) arguments to the train command by adding the flag `-k` or `--kwargs`.

# Apptrainer
## Installing Apptainer (Alternative to Docker)

If you don't have access to Docker (e.g., on a university server), you can use Apptainer as an alternative. Here are the steps to install Apptainer:
1. Install the rpm2cpio tool (if not already available):

```
sudo apt-get update
sudo apt-get install rpm2cpio
```

2. Install Apptainer:

```
curl -s https://raw.githubusercontent.com/apptainer/apptainer/main/tools/install-unprivileged.sh | \
    bash -s - ~/apptainer
```

3. Add Appttainer to your PATH:
```
echo 'export PATH=$PATH:~/apptainer/bin' >> ~/.zshrc
source ~/.zshrc
```
Note: If you're using bash instead of zsh, replace .zshrc with .bashrc in the above commands.

4. Verify Installation:
```
apptainer --version
```

## Using Apptainer

If you're using Apptainer instead of Docker, follow these steps:
1. Ensure Apptainer is installed

2. Ensure build script and run script are executable:
```
chmod +x build_apptainer.sh
chmod +x run_apptainer.sh
chmod +x run_cropdoc.sh 
```

3. Build the containers if they are not built or if you have made changes to dependencies, packages or docker files:

```
./build_apptainer.sh
```
Note: You can add --backend or --frontend flags to build only the backend or frontend containers respectively.
Note: if prompted to overwrite the existing image, type 'y' and press Enter.

4. Run the Apptainer containers (you can run this script again to restart the containers when only the code has changed):

```
./run_apptainer.sh
```

5. Access the application:

- Frontend: [http://localhost:3000](http://localhost:3000)
- Backend: [http://localhost:5000](http://localhost:5000)

6. To stop the application, use Ctrl+C int he terminal where you ran the `run_apptainer.sh` script.

## Custom Backend Commands (Command Line Interface)

To run customer backend commands:

1. Open a new terminal window after using the `run_apptainer.sh` script.

2. Run the backend container interactively:

```
./run_cropdoc.sh
```

3. Once inside the container, you can run the same commands as listed in the Docker section

4. Exit the container when done:

```
exit
```

# API Documentation

CropDoc now includes automatically generated API documentation using Swagger UI. This provides an interactive interface to explore and test the API endpoints.

## Accessing the API Documentation

1. Ensure the CropDoc server is running (follow the "Running the server" instructions above).

2. Open a web browser and navigate to:
    
    ```http://localhost:5000/doc/```

Note: If you've configured a different host or port, adjust the URL accordingly.

3. You will see the Swagger UI, which provides:
- An overview of all API endpoints
- Detailed information about each endpoint, including parameters and response models
- The ability to test endpoints directly from the browser

## Using the API Documentation

- Browse through the available endpoints in the Swagger UI.
- Click on an endpoint to expand its details.
- You can see the expected parameters and response format for each endpoint.
- Use the "Try it out" button to send requests directly from the browser and see the responses.

This documentation is automatically updated when changes are made to the API, ensuring it always reflects the current state of the application.

For developers: When adding new endpoints or modifying existing ones, make sure to use appropriate Flask-RESTX decorators and models to ensure they are correctly reflected in the Swagger UI.






