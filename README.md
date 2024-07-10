# COSC320-GroupD

Hello World!


# CropDoc
## Setup Instructions

Venv or Conda environment setup is recommended for running the CropDoc application.

### Venv
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


