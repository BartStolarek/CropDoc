# COSC320-GroupD

Hello World!


# CropDoc
## Setup Instructions

Venv or Conda environment setup is recommended for running the CropDoc application.

1. Make sure you're in 'CropDoc' Directory

```cd CropDoc```

2. Create a virtual environment

- for venv:
```python -m venv venv``` for venv
- for conda: ```conda create --name myenv python=3.10.11```

3. Activate the environment
    
    - for venv:
    ```source venv/bin/activate```(linux) ```venv\Scripts\activate```(windows)
    - for conda: ```conda activate myenv```

4. Install Requirements

- for venv: ```pip install -r requirements.txt```
- for conda: ```conda install --file requirements.txt```

### Additional: VSC Python Interpreter Setup

To setup python interpreter in visual studio code (so that installed packages dont cause any unresolved issues or warnings)

1. Open the command palette in VSC (Ctrl+Shift+P)

2. Type 'Python: Select Interpreter'

3. Click 'enter interpreter path'

4. Navigate to the 'CropDoc/venv/bin/' directory and select the python you'd like to use, i.e. 'python3.10'

5. Restart your terminal


## Running the server
1. Make sure you're in 'CropDoc' Directory

```cd CropDoc```

2. Activate the environment
    
    ```source venv/bin/activate```(linux)
        
    ```venv\Scripts\activate```(windows)

3. Run the server in development mode

```python manage.py runserver```
or
```flask run```

## 



## Maintenance

## Adding a new python library

Use a venv or conda virtual environment, and make sure it is active

1. Make sure you are in the 'CropDoc' directory

```cd CropDoc```

2. Install the library using pip

- for venv: ```pip install <library-name>```
- for conda: ```conda install <library-name>```

3. Update the requirements.txt file

- for venv: ```pip freeze > requirements.txt```
- for conda: ```conda list --export > requirements.txt```

4. Commit the changes

```git add requirements.txt```

```git commit -m "Added <library-name> to requirements.txt"```

```git push```


## Running formatters over code
1. Make sure you are in the 'CropDoc' directory

```cd CropDoc```

2. Run the formatter

```python manage.py format```


# Usage

There are two methods to use the CropDoc application. The first method is to use the command line interface, and the second method is to use the API.

## Command Line Interface

1. Make sure you are in the 'CropDoc' directory

```cd CropDoc```

2. Ensure your venv environment is activated, and all requirements are installed, if not please go to the setup instructions.


3. Select a command and use the correct flags.

Here is the markdown for the table:

| Command      | Description                                                   | Flags                                  |
|--------------|---------------------------------------------------------------|----------------------------------------|
| `python manage.py hello`      | Prints hello world.                                           | None                                   |
| `python manage.py format`     | Runs autoflake, isort, and yapf formatters over the project.  | None                                   |
| `python manage.py train`      | Runs the training script (Incomplete)                                     | None                                   |
| `python manage.py process`    | Processes data using a specified file and method (function).              | `-i, --input`  (required)<br>`-o, --output` (required)<br>`-f, --file` (required)<br>`-m, --method` (required)<br>`-k, --kwargs` (optional) |
| `python manage.py runserver`  | Runs the development server.                                  | `--debug` (flag)<br>`--host` (default: '0.0.0.0')<br>`--port` (default: '5000') |

## API Documentation

CropDoc now includes automatically generated API documentation using Swagger UI. This provides an interactive interface to explore and test the API endpoints.

### Accessing the API Documentation

1. Ensure the CropDoc server is running (follow the "Running the server" instructions above).

2. Open a web browser and navigate to:
    
    ```http://localhost:5000/doc/```

Note: If you've configured a different host or port, adjust the URL accordingly.

3. You will see the Swagger UI, which provides:
- An overview of all API endpoints
- Detailed information about each endpoint, including parameters and response models
- The ability to test endpoints directly from the browser

### Using the API Documentation

- Browse through the available endpoints in the Swagger UI.
- Click on an endpoint to expand its details.
- You can see the expected parameters and response format for each endpoint.
- Use the "Try it out" button to send requests directly from the browser and see the responses.

This documentation is automatically updated when changes are made to the API, ensuring it always reflects the current state of the application.

For developers: When adding new endpoints or modifying existing ones, make sure to use appropriate Flask-RESTX decorators and models to ensure they are correctly reflected in the Swagger UI.

# Additional Information

## Git Controls

### Git Graph

To view the git graph, you can use the following command:

```git log --all --decorate --oneline --graph```

## VSC Virtual Environment