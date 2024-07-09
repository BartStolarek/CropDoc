# COSC320-GroupD

Hello World!


# CropDoc
## Setup Instructions
1. Make sure you're in 'CropDoc' Directory

```cd CropDoc```

2. Create a virtual environment

```python -m venv venv```

3. Activate the environment
    
    ```source venv/bin/activate```(linux)
        
    ```venv\Scripts\activate```(windows)

4. Install Requirements

```pip install -r requirements.txt```


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

1. Make sure you are in the 'CropDoc' directory

```cd CropDoc```

2. Install the library using pip

```pip install <library-name>```

3. Update the requirements.txt file

```pip freeze > requirements.txt```

4. Commit the changes

```git add requirements.txt```

```git commit -m "Added <library-name>"```

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


